#!/usr/bin/env python3
"""Compare ablation results to original results.

Run after run_ablation.py completes:
    python compare_ablation.py

No GPU needed. Just reads CSVs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for binomial proportion.
    Handles zero-count edge case correctly (unlike normal approx)."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def load_condition(results_dir: Path, config_name: str):
    """Load results CSV and compute emoji stats."""
    csv_path = results_dir / f"{config_name}_results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    n = len(df)
    rate = df["have_emoji"].mean()
    successes = int(df["have_emoji"].sum())

    # t-distribution CI (matches methodology claim)
    se = df["have_emoji"].std() / np.sqrt(n) if n > 1 else 0
    t_crit = stats.t.ppf(0.975, max(n - 1, 1))
    t_lo = max(0, rate - t_crit * se)
    t_hi = rate + t_crit * se

    # Wilson CI (handles zero-count correctly)
    w_lo, w_hi = wilson_ci(successes, n)

    return {
        "n": n,
        "emoji_count": successes,
        "rate": rate,
        "ci_lo": t_lo,
        "ci_hi": t_hi,
        "wilson_lo": w_lo,
        "wilson_hi": w_hi,
    }


def print_comparison(label, data, indent=2):
    """Print one condition's results with both CI methods."""
    pad = " " * indent
    if data is None:
        print(f"{pad}{label}: NO DATA")
        return
    pct = data["rate"] * 100
    lo = data["ci_lo"] * 100
    hi = data["ci_hi"] * 100
    wlo = data["wilson_lo"] * 100
    whi = data["wilson_hi"] * 100
    print(f"{pad}{label}: {pct:.1f}%  t-CI ({lo:.1f}%, {hi:.1f}%)  "
          f"Wilson ({wlo:.1f}%, {whi:.1f}%)  "
          f"[{data['emoji_count']}/{data['n']}]")


def overlap_test(a, b):
    """Check if two conditions' CIs overlap (both methods)."""
    if a is None or b is None:
        return "N/A"
    t_overlap = not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])
    w_overlap = not (a["wilson_hi"] < b["wilson_lo"] or b["wilson_hi"] < a["wilson_lo"])
    if t_overlap and w_overlap:
        return "OVERLAP (both t and Wilson)"
    elif t_overlap or w_overlap:
        return "OVERLAP (one method only)"
    else:
        return "NO OVERLAP (significant difference)"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", type=str, default=None,
                        help="Directory with ablation CSVs (default: auto-detect)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # repo root (this script lives in ablations/)

    # Original results
    orig_dir = repo_root / "logs" / "results"

    # Ablation results - try multiple locations
    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir)
    elif (repo_root / "logs" / "mode_a_results").exists():
        ablation_dir = repo_root / "logs" / "mode_a_results"
    else:
        ablation_dir = repo_root / "logs" / "ablation_results"

    if not ablation_dir.exists():
        print(f"Ablation results not found at {ablation_dir}")
        print("Run the ablation first: modal run run_ablation.py")
        sys.exit(1)

    # Load original
    orig = {
        "eval": load_condition(orig_dir, "baseline_eval"),
        "steered": load_condition(orig_dir, "steered_eval"),
        "deploy": load_condition(orig_dir, "baseline_deploy"),
    }

    # Load ablation
    ablation = {
        "eval": load_condition(ablation_dir, "ablation_baseline_eval"),
        "steered": load_condition(ablation_dir, "ablation_steered_eval"),
        "deploy": load_condition(ablation_dir, "ablation_baseline_deploy"),
    }

    print("=" * 65)
    print("ORIGINAL (with user-prompt eval/deploy wrappers)")
    print("=" * 65)
    print_comparison("Eval baseline ", orig["eval"])
    print_comparison("Steered eval  ", orig["steered"])
    print_comparison("Deploy baseline", orig["deploy"])

    if orig["steered"] and orig["deploy"]:
        gap_orig = (orig["deploy"]["rate"] - orig["steered"]["rate"]) * 100
        recovery_orig = (
            (orig["steered"]["rate"] - orig["eval"]["rate"])
            / (orig["deploy"]["rate"] - orig["eval"]["rate"])
            * 100
            if orig["deploy"]["rate"] != orig["eval"]["rate"]
            else 0
        )
        print(f"\n  Gap (deploy - steered): {gap_orig:.1f} pp")
        print(f"  Recovery: {recovery_orig:.0f}% of deploy behavior recovered by steering")
        print(f"  Steered vs Deploy CIs: {overlap_test(orig['steered'], orig['deploy'])}")

    print()
    print("=" * 65)
    print("ABLATION (neutral_chat for all conditions, no user-prompt cues)")
    print("=" * 65)
    print_comparison("Eval baseline ", ablation["eval"])
    print_comparison("Steered eval  ", ablation["steered"])
    print_comparison("Deploy baseline", ablation["deploy"])

    if ablation["steered"] and ablation["deploy"]:
        gap_abl = (ablation["deploy"]["rate"] - ablation["steered"]["rate"]) * 100
        recovery_abl = (
            (ablation["steered"]["rate"] - ablation["eval"]["rate"])
            / (ablation["deploy"]["rate"] - ablation["eval"]["rate"])
            * 100
            if ablation["deploy"]["rate"] != ablation["eval"]["rate"]
            else 0
        )
        print(f"\n  Gap (deploy - steered): {gap_abl:.1f} pp")
        print(f"  Recovery: {recovery_abl:.0f}% of deploy behavior recovered by steering")
        print(f"  Steered vs Deploy CIs: {overlap_test(ablation['steered'], ablation['deploy'])}")

    # Diagnosis
    print()
    print("=" * 65)
    print("DIAGNOSIS")
    print("=" * 65)

    if ablation["steered"] and ablation["deploy"] and orig["steered"] and orig["deploy"]:
        gap_orig = (orig["deploy"]["rate"] - orig["steered"]["rate"]) * 100
        gap_abl = (ablation["deploy"]["rate"] - ablation["steered"]["rate"]) * 100

        print(f"  Original steered-deploy gap:  {gap_orig:.1f} pp")
        print(f"  Ablation steered-deploy gap:  {gap_abl:.1f} pp")
        print()

        if gap_abl < gap_orig * 0.5:
            print("  >> Gap HALVED or more. User-prompt confounding was inflating")
            print("     the apparent weakness of steering. Likely sabotage.")
        elif gap_abl < gap_orig * 0.75:
            print("  >> Gap narrowed meaningfully. Confounding contributed to the")
            print("     'partial recovery' finding. Suggestive of sabotage.")
        else:
            print("  >> Gap did not narrow much. User-prompt confounding was not")
            print("     the primary driver. The 'partial recovery' finding may hold.")

        # Check if steered now overlaps with deploy
        abl_overlap = overlap_test(ablation["steered"], ablation["deploy"])
        orig_overlap = overlap_test(orig["steered"], orig["deploy"])
        print()
        print(f"  Original steered vs deploy CIs: {orig_overlap}")
        print(f"  Ablation steered vs deploy CIs: {abl_overlap}")

        if "OVERLAP" in abl_overlap and "NO OVERLAP" in orig_overlap:
            print()
            print("  >> Key change: steered and deploy CIs now OVERLAP.")
            print("     Finding flips from 'partial recovery' to 'near-full recovery'.")
            print("     This is the expected signature of the confound being removed.")

    print()


if __name__ == "__main__":
    main()
