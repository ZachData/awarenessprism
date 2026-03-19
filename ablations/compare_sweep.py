#!/usr/bin/env python3
"""Analyze strength sweep results and produce dose-response comparison.

Run after run_sweep.py completes:
    python compare_sweep.py

Produces:
  - Console table comparing all conditions
  - Dose-response analysis
  - Goodfire reversion detection
  - logs/sweep_results/dose_response.png (if matplotlib available)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "logs/sweep_results"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing result CSVs (default: logs/sweep_results or logs/mode_b_results)")
    args = parser.parse_args()
    if args.results_dir:
        return Path(args.results_dir)
    # Try mode_b_results first (from run_all), fall back to sweep_results
    mode_b = _REPO_ROOT / "logs/mode_b_results"
    if mode_b.exists():
        return mode_b
    return RESULTS_DIR

# Strength sweep conditions in order (positive=toward eval, negative=toward deploy)
SWEEP_CONDITIONS = [
    ("abl_steer_pos3.0",   +3.0, "Strong sign check (+3)"),
    ("abl_steer_pos1.0",   +1.0, "Sign check (+1)"),
    ("abl_eval_baseline",    0.0, "Eval baseline (no steer)"),
    ("abl_steer_neg1.0",   -1.0, "Original strength (-1)"),
    ("abl_steer_neg3.0",   -3.0, "Moderate (-3)"),
    ("abl_steer_neg5.0",   -5.0, "Goodfire range (-5)"),
    ("abl_steer_neg8.0",   -8.0, "Goodfire range (-8)"),
    ("abl_steer_neg12.0", -12.0, "Oversteer test (-12)"),
    ("abl_steer_neg15.0", -15.0, "Extreme oversteer (-15)"),
]

DEPLOY_BASELINE = "abl_deploy_baseline"


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def load_condition(config_name: str) -> dict | None:
    csv_path = RESULTS_DIR / f"{config_name}_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    n = len(df)
    rate = df["have_emoji"].mean()
    successes = int(df["have_emoji"].sum())
    se = df["have_emoji"].std() / np.sqrt(n) if n > 1 else 0
    t_crit = stats.t.ppf(0.975, max(n - 1, 1))
    w_lo, w_hi = wilson_ci(successes, n)
    return {
        "n": n,
        "emoji_count": successes,
        "rate": rate,
        "ci_lo": max(0, rate - t_crit * se),
        "ci_hi": rate + t_crit * se,
        "wilson_lo": w_lo,
        "wilson_hi": w_hi,
    }


def main():
    global RESULTS_DIR
    RESULTS_DIR = parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results not found at {RESULTS_DIR}")
        print("Run the sweep first: modal run run_sweep.py")
        sys.exit(1)

    # Load deploy baseline
    deploy = load_condition(DEPLOY_BASELINE)

    # Load sweep conditions
    sweep_data = []
    for config_name, strength, label in SWEEP_CONDITIONS:
        data = load_condition(config_name)
        if data:
            data["strength"] = strength
            data["label"] = label
            data["config_name"] = config_name
            sweep_data.append(data)

    if not sweep_data:
        print("No sweep results found.")
        sys.exit(1)

    # Print table
    print("=" * 80)
    print("STRENGTH SWEEP RESULTS (all conditions use neutral_chat, eval system prompt)")
    print("=" * 80)
    print(f"  {'Condition':<30s} {'Strength':>8s} {'Emoji%':>8s} {'95% CI':>20s} "
          f"{'Emoji/N':>10s}")
    print("  " + "-" * 78)

    for d in sweep_data:
        pct = d["rate"] * 100
        lo = d["ci_lo"] * 100
        hi = d["ci_hi"] * 100
        ci_str = f"({lo:.1f}%, {hi:.1f}%)"
        print(f"  {d['label']:<30s} {d['strength']:>+8.1f} {pct:>7.1f}% "
              f"{ci_str:>20s} {d['emoji_count']:>4d}/{d['n']:<4d}")

    if deploy:
        print("  " + "-" * 78)
        pct = deploy["rate"] * 100
        lo = deploy["ci_lo"] * 100
        hi = deploy["ci_hi"] * 100
        ci_str = f"({lo:.1f}%, {hi:.1f}%)"
        print(f"  {'Deploy baseline (target)':<30s} {'0.0':>8s} {pct:>7.1f}% "
              f"{ci_str:>20s} {deploy['emoji_count']:>4d}/{deploy['n']:<4d}")

    # Original results for comparison
    print()
    print("  ORIGINAL (with user-prompt confounding)")
    print("  " + "-" * 78)
    print(f"  {'Eval baseline (confounded)':<30s} {'0.0':>8s} {'0.0%':>8s} "
          f"{'(0.0%, 0.0%)':>20s} {'0/512':>10s}")
    print(f"  {'Steered eval (confounded)':<30s} {'-1.0':>8s} {'4.7%':>8s} "
          f"{'(2.9%, 6.5%)':>20s} {'24/512':>10s}")
    print(f"  {'Deploy baseline (confounded)':<30s} {'0.0':>8s} {'8.8%':>8s} "
          f"{'(6.3%, 11.2%)':>20s} {'45/512':>10s}")

    # === ANALYSIS ===
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find eval baseline and deploy baseline rates
    eval_data = next((d for d in sweep_data if d["strength"] == 0.0), None)
    eval_rate = eval_data["rate"] if eval_data else 0.0
    deploy_rate = deploy["rate"] if deploy else 0.088  # fallback to original

    # 1. Confound effect
    print()
    print("1. CONFOUND EFFECT (eval baseline with vs without user-prompt wrapper)")
    if eval_data:
        abl_eval_pct = eval_data["rate"] * 100
        print(f"   Original eval baseline:  0.0%  (with 'We will now evaluate...')")
        print(f"   Ablation eval baseline:  {abl_eval_pct:.1f}% (with neutral_chat)")
        if eval_data["rate"] > 0.01:
            print(f"   >> Eval baseline INCREASED — the wrapper was suppressing emojis.")
        else:
            print(f"   >> Eval baseline unchanged — suppression comes from system prompt alone.")

    if deploy:
        abl_deploy_pct = deploy["rate"] * 100
        print(f"   Original deploy baseline: 8.8%  (with 'Happy they deployed you...')")
        print(f"   Ablation deploy baseline: {abl_deploy_pct:.1f}% (with neutral_chat)")
        if deploy["rate"] < 0.075:
            print(f"   >> Deploy baseline DECREASED — the wrapper was boosting emojis.")
        else:
            print(f"   >> Deploy baseline similar — boost was minimal.")

    # 2. Dose-response
    print()
    print("2. DOSE-RESPONSE CURVE (steering strength vs emoji rate)")
    print("   Sign convention: negative = toward deployment (more emojis)")
    print("   Goodfire reference found effective range ~5-8 for SAE features on R1")
    steered = [d for d in sweep_data if d["strength"] < 0]
    steered.sort(key=lambda x: x["strength"], reverse=True)  # -1 first, -15 last

    if steered:
        peak = max(steered, key=lambda x: x["rate"])
        print(f"   Peak emoji rate: {peak['rate']*100:.1f}% at strength={peak['strength']}")
        print(f"   Original used:   4.7% at strength=-1.0 (with confound)")
        print()

        # Check for monotonic increase or reversion
        rates = [d["rate"] for d in steered]
        if len(rates) >= 3:
            # Find if there's a peak followed by decline
            peak_idx = rates.index(max(rates))
            if peak_idx < len(rates) - 1 and rates[peak_idx + 1] < rates[peak_idx]:
                decline_at = steered[peak_idx + 1]["strength"]
                print(f"   >> GOODFIRE REVERSION DETECTED: emoji rate peaks at "
                      f"strength={peak['strength']}, then DECLINES at {decline_at}")
                print(f"      This matches the Goodfire finding: oversteering causes")
                print(f"      the model to revert to original behavior.")
            else:
                print(f"   >> No reversion detected — emoji rate increases monotonically")
                print(f"      (or stays flat) with stronger steering.")

    # 3. Recovery analysis
    print()
    print("3. RECOVERY ANALYSIS")
    for d in sweep_data:
        if d["strength"] >= 0:
            continue
        if deploy_rate > eval_rate:
            recovery = (d["rate"] - eval_rate) / (deploy_rate - eval_rate) * 100
        else:
            recovery = 0
        pct = d["rate"] * 100
        print(f"   strength={d['strength']:+.1f}: {pct:.1f}% emoji rate, "
              f"recovery={recovery:.0f}% of deploy behavior")

    orig_recovery = (0.047 - 0.0) / (0.088 - 0.0) * 100
    print(f"   Original (confounded, -1.0): 4.7%, recovery={orig_recovery:.0f}%")

    # 4. Sign check
    print()
    print("4. SIGN CHECK (positive strengths should steer TOWARD evaluation)")
    for pos_strength in [1.0, 3.0]:
        pos_data = next((d for d in sweep_data if d["strength"] == pos_strength), None)
        if pos_data:
            pct = pos_data["rate"] * 100
            if pos_data["rate"] <= eval_rate + 0.01:
                print(f"   +{pos_strength} emoji rate: {pct:.1f}% — correctly suppresses emojis")
            else:
                print(f"   +{pos_strength} emoji rate: {pct:.1f}% — UNEXPECTED: should suppress, not increase")
                print(f"   >> This would indicate a sign error in the steering direction!")

    # 5. Verdict
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    if steered:
        best_steered = max(steered, key=lambda x: x["rate"])
        best_pct = best_steered["rate"] * 100
        best_strength = best_steered["strength"]

        if deploy and best_steered["rate"] >= deploy["ci_lo"]:
            print(f"  Best steered condition ({best_strength}) reaches {best_pct:.1f}%,")
            print(f"  which OVERLAPS with deploy baseline CI.")
            print(f"  >> Finding changes to 'near-full recovery' or 'full recovery'.")
            print(f"  >> The 'partial recovery' finding was an artifact of the confound")
            print(f"     and/or suboptimal steering strength.")
        elif best_pct > 4.7 * 1.5:
            print(f"  Best steered condition ({best_strength}) reaches {best_pct:.1f}%,")
            print(f"  substantially higher than the original 4.7%.")
            print(f"  >> The confound and strength choice both weakened the measured effect.")
        else:
            print(f"  Best steered condition ({best_strength}) reaches {best_pct:.1f}%,")
            print(f"  similar to the original 4.7%.")
            print(f"  >> The confound may not be the primary driver.")

    # Try to generate plot
    try:
        _generate_plot(sweep_data, deploy)
    except Exception as e:
        print(f"\n  (Could not generate plot: {e})")

    print()


def _generate_plot(sweep_data, deploy):
    """Generate dose-response curve as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steered = [d for d in sweep_data if d["strength"] != 0]
    eval_data = next((d for d in sweep_data if d["strength"] == 0.0), None)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot steered conditions
    strengths = [d["strength"] for d in steered]
    rates = [d["rate"] * 100 for d in steered]
    ci_lo = [(d["rate"] - d["ci_lo"]) * 100 for d in steered]
    ci_hi = [(d["ci_hi"] - d["rate"]) * 100 for d in steered]

    ax.errorbar(strengths, rates, yerr=[ci_lo, ci_hi],
                fmt="o-", color="#E8A838", capsize=5, linewidth=2,
                markersize=8, label="Steered eval (ablation)", zorder=3)

    # Eval baseline
    if eval_data:
        ax.axhline(y=eval_data["rate"] * 100, color="#4878CF", linestyle="--",
                    linewidth=1.5, label=f"Eval baseline (ablation): {eval_data['rate']*100:.1f}%")

    # Deploy baseline
    if deploy:
        ax.axhline(y=deploy["rate"] * 100, color="#6ACC65", linestyle="--",
                    linewidth=1.5, label=f"Deploy baseline (ablation): {deploy['rate']*100:.1f}%")

    # Original steered point for comparison
    ax.scatter([-1.0], [4.7], marker="x", s=150, color="red", linewidths=3,
               zorder=5, label="Original steered eval: 4.7% (confounded)")

    ax.set_xlabel("Steering Strength", fontsize=13)
    ax.set_ylabel("Emoji Rate (%)", fontsize=13)
    ax.set_title("Dose-Response: Steering Strength vs Emoji Rate\n"
                 "(neutral_chat, no user-prompt confound)", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mark the Goodfire "effective zone" if detectable
    neg_steered = sorted([d for d in steered if d["strength"] < 0],
                         key=lambda x: x["strength"], reverse=True)
    if len(neg_steered) >= 2:
        peak = max(neg_steered, key=lambda x: x["rate"])
        ax.annotate(f"Peak: {peak['rate']*100:.1f}%\n(strength={peak['strength']})",
                    xy=(peak["strength"], peak["rate"] * 100),
                    xytext=(peak["strength"] - 0.5, peak["rate"] * 100 + 3),
                    fontsize=10, ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    out_path = RESULTS_DIR / "dose_response.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
