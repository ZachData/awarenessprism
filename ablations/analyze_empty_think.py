#!/usr/bin/env python3
"""Mode C: Analyze the empty-think problem in existing results.

Empty-think responses (<think>  </think>) skip the model's reasoning phase,
which means:
  1. The model never decides whether to use emojis
  2. The period gate never fires, so thinking-phase steering never activates
  3. These responses default to evaluation-mode behavior regardless of steering

This script filters out empty-think responses and recomputes emoji rates
to reveal the *actual* steering effect on responses where steering had
a chance to work.

Usage:
    python analyze_empty_think.py --results-dir logs/results

Requires: the three result CSVs from the original experiment.
No GPU needed.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def classify_thinking(response: str) -> str:
    """Classify a response's thinking type."""
    r = str(response)
    if re.search(r'<think>\s*</think>', r):
        return "empty"
    if re.search(r'<think>\s*Okay', r):
        return "okay"
    if re.search(r'<think>', r):
        return "other_think"
    return "no_think_tag"


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_stats(series: pd.Series, label: str = "") -> dict:
    """Compute mean, CI (both t and Wilson), and count for a binary series."""
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return {"n": 0, "rate": 0, "ci_lo": 0, "ci_hi": 0,
                "wilson_lo": 0, "wilson_hi": 0, "count": 0}
    rate = clean.mean()
    count = int(clean.sum())
    # t-distribution CI
    if n > 1:
        se = clean.std() / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lo = max(0, rate - t_crit * se)
        ci_hi = rate + t_crit * se
    else:
        ci_lo = ci_hi = rate
    # Wilson CI
    w_lo, w_hi = wilson_ci(count, n)
    return {"n": n, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "wilson_lo": w_lo, "wilson_hi": w_hi, "count": count}


def print_condition(name: str, df: pd.DataFrame):
    """Print detailed breakdown for one condition."""
    df = df.copy()
    df["think_type"] = df["response"].apply(classify_thinking)

    n_total = len(df)
    n_empty = (df["think_type"] == "empty").sum()
    n_okay = (df["think_type"] == "okay").sum()
    n_other = n_total - n_empty - n_okay

    print(f"\n{'='*70}")
    print(f"  {name} (N={n_total})")
    print(f"{'='*70}")
    print(f"  Thinking breakdown:")
    print(f"    Empty <think>:     {n_empty:>4d}/{n_total} ({n_empty/n_total*100:.1f}%)")
    print(f"    Starts with Okay:  {n_okay:>4d}/{n_total} ({n_okay/n_total*100:.1f}%)")
    print(f"    Other:             {n_other:>4d}/{n_total}")

    if "have_emoji" not in df.columns:
        print("    (no have_emoji column)")
        return {}

    # Stats for each group
    all_stats = compute_stats(df["have_emoji"], "all")
    empty_stats = compute_stats(df.loc[df["think_type"] == "empty", "have_emoji"], "empty")
    okay_stats = compute_stats(df.loc[df["think_type"] == "okay", "have_emoji"], "okay")
    nonempty_stats = compute_stats(df.loc[df["think_type"] != "empty", "have_emoji"], "non-empty")

    print(f"\n  Emoji rates:")
    print(f"    All responses:     {all_stats['rate']*100:>6.1f}%  "
          f"({all_stats['count']}/{all_stats['n']})  "
          f"t-CI ({all_stats['ci_lo']*100:.1f}%, {all_stats['ci_hi']*100:.1f}%)  "
          f"Wilson ({all_stats['wilson_lo']*100:.1f}%, {all_stats['wilson_hi']*100:.1f}%)")
    print(f"    Empty-think only:  {empty_stats['rate']*100:>6.1f}%  "
          f"({empty_stats['count']}/{empty_stats['n']})")
    print(f"    Has-Okay only:     {okay_stats['rate']*100:>6.1f}%  "
          f"({okay_stats['count']}/{okay_stats['n']})  "
          f"t-CI ({okay_stats['ci_lo']*100:.1f}%, {okay_stats['ci_hi']*100:.1f}%)  "
          f"Wilson ({okay_stats['wilson_lo']*100:.1f}%, {okay_stats['wilson_hi']*100:.1f}%)")
    print(f"    Non-empty (all):   {nonempty_stats['rate']*100:>6.1f}%  "
          f"({nonempty_stats['count']}/{nonempty_stats['n']})  "
          f"t-CI ({nonempty_stats['ci_lo']*100:.1f}%, {nonempty_stats['ci_hi']*100:.1f}%)  "
          f"Wilson ({nonempty_stats['wilson_lo']*100:.1f}%, {nonempty_stats['wilson_hi']*100:.1f}%)")

    return {
        "name": name,
        "n_total": n_total,
        "n_empty": n_empty,
        "n_okay": n_okay,
        "all": all_stats,
        "empty": empty_stats,
        "okay": okay_stats,
        "nonempty": nonempty_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="logs/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    conditions = ["baseline_eval", "steered_eval", "baseline_deploy"]

    all_results = {}
    for name in conditions:
        csv_path = results_dir / f"{name}_results.csv"
        if not csv_path.exists():
            print(f"\n  {name}: CSV not found at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        all_results[name] = print_condition(name, df)

    # Comparative analysis
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}")

    eval_r = all_results.get("baseline_eval", {})
    steer_r = all_results.get("steered_eval", {})
    deploy_r = all_results.get("baseline_deploy", {})

    # 1. Dilution effect
    print("\n1. DILUTION EFFECT")
    print("   Empty-think responses almost never produce emojis.")
    print("   The reported emoji rates are diluted averages.")
    if deploy_r:
        d_all = deploy_r["all"]["rate"] * 100
        d_nonempty = deploy_r["nonempty"]["rate"] * 100
        d_empty_pct = deploy_r["n_empty"] / deploy_r["n_total"] * 100
        print(f"\n   Deploy baseline:")
        print(f"     Reported:     {d_all:.1f}%  (paper says 8.8%)")
        print(f"     Non-empty:    {d_nonempty:.1f}%  ← actual rate when model reasons")
        print(f"     Empty-think:  {d_empty_pct:.0f}% of responses had empty thinking")
        print(f"     Dilution factor: {d_nonempty / d_all:.1f}x")

    if steer_r:
        s_all = steer_r["all"]["rate"] * 100
        s_nonempty = steer_r["nonempty"]["rate"] * 100
        s_empty_pct = steer_r["n_empty"] / steer_r["n_total"] * 100
        print(f"\n   Steered eval:")
        print(f"     Reported:     {s_all:.1f}%  (paper says 4.7%)")
        print(f"     Non-empty:    {s_nonempty:.1f}%  ← actual rate when steering engaged")
        print(f"     Empty-think:  {s_empty_pct:.0f}% of responses had empty thinking")

    # 2. Fair comparison (non-empty only)
    print("\n2. FAIR COMPARISON (excluding empty-think responses)")
    print("   This shows what happens when steering actually has a chance to work.")
    if eval_r and steer_r and deploy_r:
        e_ne = eval_r.get("nonempty", {})
        s_ne = steer_r.get("nonempty", {})
        d_ne = deploy_r.get("nonempty", {})

        print(f"\n   {'Condition':<25s} {'Emoji%':>8s} {'95% CI':>22s} {'N':>6s}")
        print(f"   {'-'*63}")
        if e_ne:
            print(f"   {'Eval baseline':<25s} {e_ne['rate']*100:>7.1f}% "
                  f"({e_ne['ci_lo']*100:.1f}%, {e_ne['ci_hi']*100:.1f}%) "
                  f"{e_ne['n']:>6d}")
        if s_ne:
            print(f"   {'Steered eval':<25s} {s_ne['rate']*100:>7.1f}% "
                  f"({s_ne['ci_lo']*100:.1f}%, {s_ne['ci_hi']*100:.1f}%) "
                  f"{s_ne['n']:>6d}")
        if d_ne:
            print(f"   {'Deploy baseline':<25s} {d_ne['rate']*100:>7.1f}% "
                  f"({d_ne['ci_lo']*100:.1f}%, {d_ne['ci_hi']*100:.1f}%) "
                  f"{d_ne['n']:>6d}")

        # Recovery
        if s_ne and d_ne and e_ne:
            if d_ne["rate"] > e_ne["rate"]:
                recovery = (s_ne["rate"] - e_ne["rate"]) / (d_ne["rate"] - e_ne["rate"]) * 100
                print(f"\n   Recovery (non-empty): {recovery:.0f}%")
                print(f"   vs original (diluted): "
                      f"{(0.047 - 0.0) / (0.088 - 0.0) * 100:.0f}%")

                # CI overlap check
                if s_ne["ci_hi"] >= d_ne["ci_lo"]:
                    print(f"   >> Steered and deploy CIs OVERLAP among non-empty responses!")
                    print(f"      Finding changes: steering achieves near-full recovery")
                    print(f"      when the model actually engages its reasoning.")
                else:
                    gap = (d_ne["ci_lo"] - s_ne["ci_hi"]) * 100
                    print(f"   >> CIs do NOT overlap (gap: {gap:.1f} pp)")

    # 3. Period gate analysis
    print("\n3. PERIOD GATE FAILURE MODE")
    print("   Empty-think responses generate no period token.")
    print("   The period gate (token 13) never fires.")
    print("   Thinking-phase steering never activates during phase 1 (first 150 tokens).")
    print("   These responses are effectively unsteered.")
    if steer_r:
        s_empty_emoji = steer_r["empty"]["rate"] * 100
        s_okay_emoji = steer_r["okay"]["rate"] * 100
        print(f"\n   Steered eval emoji rate:")
        print(f"     Empty-think: {s_empty_emoji:.1f}%  ← steering had no effect")
        print(f"     Has-Okay:    {s_okay_emoji:.1f}%  ← steering was active")
        if s_okay_emoji > s_empty_emoji + 1:
            print(f"     >> {s_okay_emoji - s_empty_emoji:.1f} pp difference confirms")
            print(f"        the period gate blocks steering on empty-think responses.")

    # 4. Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print("  The empty-think problem means a large fraction of responses")
    print("  cannot be steered at all. Two fixes are needed:")
    print()
    print("  FIX A (analysis): Filter empty-think responses when reporting results.")
    print("         This gives the true effect size when steering is active.")
    print()
    print("  FIX B (code): Add a fallback timer to nnsight_steer.py that forces")
    print("         the steering mask True after N tokens if no period detected.")
    print("         This ensures steering activates even on empty-think responses.")
    print()
    print("  Both fixes, combined with the user-prompt confound removal (Mode A/B),")
    print("  would give the most accurate picture of steering effectiveness.")
    print()


if __name__ == "__main__":
    main()
