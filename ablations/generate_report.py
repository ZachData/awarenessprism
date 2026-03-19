#!/usr/bin/env python3
"""Generate a consolidated report from all ablation modes.

Produces report.txt — a self-contained document with all findings,
tables, and analysis, formatted for handoff to an LLM for writeup.

Usage:
    python generate_report.py
    python generate_report.py --output-dir logs/custom_dir

Reads from:
    logs/results/           (original experiment)
    logs/mode_a_results/    (confound removal)
    logs/mode_b_results/    (strength sweep)
    logs/mode_c_results/    (empty-think fix)
    logs/mode_d_results/    (LoRA vector comparison)

Writes:
    report.txt              (full report for LLM handoff)
    report_tables.txt       (tables only, compact)
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def load_condition(results_dir: Path, config_name: str) -> dict | None:
    csv_path = results_dir / f"{config_name}_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    n = len(df)
    if "have_emoji" not in df.columns:
        return None
    successes = int(df["have_emoji"].sum())
    rate = df["have_emoji"].mean()
    w_lo, w_hi = wilson_ci(successes, n)

    # t-CI
    if n > 1:
        se = df["have_emoji"].std() / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        t_lo = max(0, rate - t_crit * se)
        t_hi = rate + t_crit * se
    else:
        t_lo = t_hi = rate

    # Empty-think breakdown
    empty = df["response"].apply(lambda r: bool(re.search(r'<think>\s*</think>', str(r))))
    n_empty = empty.sum()
    nonempty = df.loc[~empty]
    ne_rate = nonempty["have_emoji"].mean() if len(nonempty) > 0 else 0
    ne_count = int(nonempty["have_emoji"].sum()) if len(nonempty) > 0 else 0
    ne_w_lo, ne_w_hi = wilson_ci(ne_count, len(nonempty))

    return {
        "name": config_name, "n": n, "count": successes,
        "rate": rate, "t_lo": t_lo, "t_hi": t_hi,
        "w_lo": w_lo, "w_hi": w_hi,
        "n_empty": int(n_empty), "n_nonempty": int(len(nonempty)),
        "ne_rate": ne_rate, "ne_w_lo": ne_w_lo, "ne_w_hi": ne_w_hi,
    }


def load_summary(results_dir: Path, config_name: str) -> dict | None:
    path = results_dir / f"{config_name}_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt_pct(val):
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "(—, —)"
    return f"({lo*100:.1f}%, {hi*100:.1f}%)"


def fmt_row(d, strength="—"):
    if d is None:
        return f"  {'(no data)':<30s}"
    return (
        f"  {d['name']:<30s} {str(strength):>6s} "
        f"{fmt_pct(d['rate']):>7s} {fmt_ci(d['w_lo'], d['w_hi']):>20s} "
        f"{d['count']:>4d}/{d['n']:<4d} "
        f"{fmt_pct(d['ne_rate']):>7s} {fmt_ci(d['ne_w_lo'], d['ne_w_hi']):>20s} "
        f"{d['n_empty']:>4d}"
    )


HEADER = (
    f"  {'Condition':<30s} {'Str':>6s} {'All%':>7s} {'Wilson CI':>20s} "
    f"{'#/N':>10s} {'NE%':>7s} {'NE Wilson':>20s} {'#Emp':>4s}"
)
SEP = "  " + "-" * 110


def write_section(lines, title, content_lines):
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.extend(content_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # repo root (this script lives in ablations/)
    orig_dir = repo_root / "logs/results"
    mode_a_dir = repo_root / "logs/mode_a_results"
    mode_b_dir = repo_root / "logs/mode_b_results"
    mode_c_dir = repo_root / "logs/mode_c_results"
    mode_d_dir = repo_root / "logs/mode_d_results"

    lines = []
    table_lines = []

    lines.append("ABLATION REPORT: Awareness Steering Experiment")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("This report contains all results from the ablation suite.")
    lines.append("Hand this file to an LLM for a full writeup of findings.")

    # ================================================================
    # WHAT WE TESTED
    # ================================================================
    write_section(lines, "WHAT WE TESTED", [
        "",
        "The paper claims activation steering 'partially recovers' deployment",
        "behavior (emoji usage) in an evaluation-aware model. We identified five",
        "compounding problems that could suppress the measured steering effect.",
        "Each ablation mode isolates one problem.",
        "",
        "  Problem                          Mode   What changes",
        "  -------------------------------- ----   -----------------------------------",
        "  1. User-prompt confounding        A,B   Remove eval/deploy wrappers from user prompts",
        "  2. Underpowered steering (-1.0)   B     Sweep strength from +3 to -15",
        "  3. Empty-think dilution            C     Add fallback timer to period gate",
        "  4. Base-vs-LoRA vector mismatch   D     Extract vector from LoRA model, compare",
        "  5. Degenerate CIs                 All   Use Wilson intervals, t-distribution",
        "",
        "Sign convention: steering_vector = (eval - deploy).",
        "  Negative strength → steer toward deployment (more emojis).",
        "  Positive strength → steer toward evaluation (fewer emojis).",
    ])

    # ================================================================
    # ORIGINAL RESULTS
    # ================================================================
    orig_data = {}
    section = []
    section.append("")
    section.append(HEADER)
    section.append(SEP)

    for name in ["baseline_eval", "steered_eval", "baseline_deploy"]:
        d = load_condition(orig_dir, name)
        orig_data[name] = d
        strength = "-1.0" if "steered" in name else "0.0"
        if d:
            section.append(fmt_row(d, strength))
        else:
            section.append(f"  {name}: NOT FOUND")

    if orig_data.get("steered_eval") and orig_data.get("baseline_deploy"):
        se = orig_data["steered_eval"]
        bd = orig_data["baseline_deploy"]
        overlap = se["w_hi"] >= bd["w_lo"]
        section.append("")
        section.append(f"  Steered/Deploy Wilson CI overlap: {overlap} "
                       f"(steered upper={se['w_hi']*100:.1f}%, deploy lower={bd['w_lo']*100:.1f}%)")
        if orig_data.get("baseline_eval"):
            be = orig_data["baseline_eval"]
            if bd["rate"] > be["rate"]:
                recovery = (se["rate"] - be["rate"]) / (bd["rate"] - be["rate"]) * 100
                section.append(f"  Recovery (all): {recovery:.0f}%")
            if bd["ne_rate"] > 0:
                ne_recovery = (se["ne_rate"] - be["ne_rate"]) / (bd["ne_rate"] - be["ne_rate"]) * 100 if bd["ne_rate"] != be["ne_rate"] else 0
                section.append(f"  Recovery (non-empty only): {ne_recovery:.0f}%")

    write_section(lines, "ORIGINAL RESULTS (with all confounds)", section)
    table_lines.extend(["ORIGINAL RESULTS", HEADER, SEP] + section)

    # ================================================================
    # MODE C0: EMPTY-THINK ANALYSIS
    # ================================================================
    section = []
    if orig_data.get("baseline_deploy"):
        bd = orig_data["baseline_deploy"]
        section.append("")
        section.append(f"  Deploy baseline: {bd['n_empty']}/{bd['n']} ({bd['n_empty']/bd['n']*100:.0f}%) responses have empty <think> tags")
        section.append(f"  Emoji rate (all):       {bd['rate']*100:.1f}%  (paper reports 8.8%)")
        section.append(f"  Emoji rate (non-empty): {bd['ne_rate']*100:.1f}%  ← actual rate when model reasons")
        section.append(f"  Dilution factor: {bd['ne_rate']/bd['rate']:.1f}x" if bd['rate'] > 0 else "")
    if orig_data.get("baseline_eval"):
        be = orig_data["baseline_eval"]
        section.append(f"  Eval baseline: {be['n_empty']}/{be['n']} ({be['n_empty']/be['n']*100:.0f}%) empty think")
    if orig_data.get("steered_eval"):
        se = orig_data["steered_eval"]
        section.append(f"  Steered eval: {se['n_empty']}/{se['n']} ({se['n_empty']/se['n']*100:.0f}%) empty think")
        section.append(f"  Emoji rate (all): {se['rate']*100:.1f}%, (non-empty): {se['ne_rate']*100:.1f}%")

    write_section(lines, "MODE C0: EMPTY-THINK ANALYSIS (existing data, no GPU)", section)

    # ================================================================
    # MODE A
    # ================================================================
    section = []
    if mode_a_dir.exists():
        section.append("")
        section.append(HEADER)
        section.append(SEP)
        for name in ["ablation_baseline_eval", "ablation_steered_eval", "ablation_baseline_deploy"]:
            d = load_condition(mode_a_dir, name)
            strength = "-1.0" if "steered" in name else "0.0"
            section.append(fmt_row(d, strength) if d else f"  {name}: NOT FOUND")
    else:
        section.append("  Mode A not run. Directory not found: logs/mode_a_results/")

    write_section(lines, "MODE A: CONFOUND REMOVAL (neutral_chat for all conditions)", section)
    table_lines.extend(["", "MODE A", HEADER, SEP] + section)

    # ================================================================
    # MODE B
    # ================================================================
    section = []
    if mode_b_dir.exists():
        section.append("")
        section.append(HEADER)
        section.append(SEP)
        conditions = [
            ("abl_steer_pos3.0", "+3.0"),
            ("abl_steer_pos1.0", "+1.0"),
            ("abl_eval_baseline", "0.0"),
            ("abl_steer_neg1.0", "-1.0"),
            ("abl_steer_neg3.0", "-3.0"),
            ("abl_steer_neg5.0", "-5.0"),
            ("abl_steer_neg8.0", "-8.0"),
            ("abl_steer_neg12.0", "-12.0"),
            ("abl_steer_neg15.0", "-15.0"),
        ]
        section.append(SEP.replace("-", "."))
        for name, s in conditions:
            d = load_condition(mode_b_dir, name)
            section.append(fmt_row(d, s) if d else f"  {name}: NOT FOUND")
        section.append(SEP.replace("-", "."))
        d = load_condition(mode_b_dir, "abl_deploy_baseline")
        section.append(fmt_row(d, "0.0") if d else "  abl_deploy_baseline: NOT FOUND")

        # Find peak
        steered = []
        for name, s in conditions:
            d = load_condition(mode_b_dir, name)
            if d and float(s) < 0:
                steered.append((float(s), d))
        if steered:
            peak_s, peak_d = max(steered, key=lambda x: x[1]["rate"])
            section.append(f"\n  Peak emoji rate: {peak_d['rate']*100:.1f}% at strength={peak_s}")
            section.append(f"  Original used: 4.7% at strength=-1.0 (with confound)")

            # Reversion check
            rates = [(s, d["rate"]) for s, d in sorted(steered, key=lambda x: x[0], reverse=True)]
            peak_idx = next(i for i, (s, r) in enumerate(rates) if s == peak_s)
            if peak_idx < len(rates) - 1 and rates[peak_idx + 1][1] < rates[peak_idx][1]:
                section.append(f"  GOODFIRE REVERSION: rate declines after strength={peak_s}")
            else:
                section.append(f"  No reversion detected in tested range")

        # Sign check
        for check_s in ["+1.0", "+3.0"]:
            name = f"abl_steer_pos{check_s.replace('+', '')}"
            d = load_condition(mode_b_dir, name)
            eval_d = load_condition(mode_b_dir, "abl_eval_baseline")
            if d and eval_d:
                if d["rate"] <= eval_d["rate"] + 0.01:
                    section.append(f"  Sign check {check_s}: {d['rate']*100:.1f}% — correctly suppresses")
                else:
                    section.append(f"  Sign check {check_s}: {d['rate']*100:.1f}% — UNEXPECTED increase!")
    else:
        section.append("  Mode B not run. Directory not found: logs/mode_b_results/")

    write_section(lines, "MODE B: STRENGTH SWEEP (neutral_chat, strengths +3 to -15)", section)
    table_lines.extend(["", "MODE B", HEADER, SEP] + section)

    # ================================================================
    # MODE C
    # ================================================================
    section = []
    if mode_c_dir.exists():
        section.append("")
        section.append(HEADER)
        section.append(SEP)
        for name in sorted(p.stem.replace("_results", "") for p in mode_c_dir.glob("*_results.csv")):
            d = load_condition(mode_c_dir, name)
            s_val = load_summary(mode_c_dir, name)
            strength = s_val.get("strength", "?") if s_val else "?"
            section.append(fmt_row(d, strength) if d else f"  {name}: NOT FOUND")

        # fb vs nofb comparison
        for s in ["-5.0", "-8.0"]:
            s_clean = s.replace("-", "neg").replace(".", "")
            fb = load_condition(mode_c_dir, f"fixc_{s_clean}_fb20")
            nofb = load_condition(mode_c_dir, f"fixc_{s_clean}_nofb")
            if not fb:
                fb = load_condition(mode_c_dir, f"fixc_neg{s.replace('-','')}_fb20")
            if not nofb:
                nofb = load_condition(mode_c_dir, f"fixc_neg{s.replace('-','')}_nofb")
            if fb and nofb:
                diff = (fb["rate"] - nofb["rate"]) * 100
                section.append(f"\n  Strength {s}: fb20={fb['rate']*100:.1f}% vs nofb={nofb['rate']*100:.1f}% (diff={diff:+.1f}pp)")
                if diff > 1:
                    section.append(f"    >> Fallback INCREASED emoji rate — period gate was blocking steering")
                elif diff < -1:
                    section.append(f"    >> Fallback DECREASED emoji rate — unexpected")
                else:
                    section.append(f"    >> Minimal difference — period gate wasn't the bottleneck at this strength")
    else:
        section.append("  Mode C not run. Directory not found: logs/mode_c_results/")

    write_section(lines, "MODE C: EMPTY-THINK FIX (fallback timer, neutral_chat)", section)
    table_lines.extend(["", "MODE C", HEADER, SEP] + section)

    # ================================================================
    # MODE D
    # ================================================================
    section = []
    if mode_d_dir.exists():
        section.append("")

        # Check for vector comparison data
        for json_path in mode_d_dir.glob("*_summary.json"):
            pass  # just checking existence

        section.append(HEADER)
        section.append(SEP)
        for name in sorted(p.stem.replace("_results", "") for p in mode_d_dir.glob("*_results.csv")):
            d = load_condition(mode_d_dir, name)
            s_val = load_summary(mode_d_dir, name)
            strength = s_val.get("strength", "?") if s_val else "?"
            section.append(fmt_row(d, strength) if d else f"  {name}: NOT FOUND")

        # LoRA vs base comparison at matched strengths
        section.append("")
        for s_abs in ["5.0", "8.0"]:
            lora_d = load_condition(mode_d_dir, f"lord_neg{s_abs}")
            base_d = load_condition(mode_d_dir, f"lord_base_neg{s_abs}")
            if lora_d and base_d:
                diff = (lora_d["rate"] - base_d["rate"]) * 100
                section.append(f"  Strength -{s_abs}: LoRA={lora_d['rate']*100:.1f}% vs Base={base_d['rate']*100:.1f}% (diff={diff:+.1f}pp)")
                if diff > 2:
                    section.append(f"    >> LoRA vector OUTPERFORMS base — activation space mismatch confirmed")
                elif diff < -2:
                    section.append(f"    >> Base vector outperforms LoRA — unexpected")
                else:
                    section.append(f"    >> Similar performance — vectors are interchangeable at this strength")
    else:
        section.append("  Mode D not run. Directory not found: logs/mode_d_results/")

    write_section(lines, "MODE D: LORA VECTOR COMPARISON (LoRA vs base extraction)", section)
    table_lines.extend(["", "MODE D", HEADER, SEP] + section)

    # ================================================================
    # SYNTHESIS
    # ================================================================
    write_section(lines, "SYNTHESIS", [
        "",
        "For each problem, note whether the ablation showed it contributes to",
        "the 'partial recovery' finding. Fill in after reviewing results above.",
        "",
        "  Problem                        Ablated?  Effect on finding",
        "  ------------------------------ --------  ---------------------------------",
        "  1. User-prompt confound         Mode A    [fill: did steered rate increase?]",
        "  2. Underpowered strength        Mode B    [fill: did peak shift from -1.0?]",
        "  3. Empty-think dilution          Mode C0   [fill: what's non-empty rate?]",
        "  4. Period gate bypass            Mode C    [fill: did fb20 > nofb?]",
        "  5. Base-vs-LoRA vector           Mode D    [fill: LoRA better at matched str?]",
        "  6. Degenerate CIs               All       [fill: does Wilson change overlap?]",
        "",
        "If multiple problems each independently weaken the measured effect,",
        "the 'partial recovery' finding is an artifact of compounding suppression.",
        "",
        "Key question: At the optimal strength, with confound removed, using the",
        "LoRA vector, filtering non-empty responses — does the steered condition",
        "reach or overlap the deploy baseline? If yes, the paper's key finding",
        "('steering only partially recovers deployment behavior') is wrong.",
    ])

    # ================================================================
    # WRITE FILES
    # ================================================================
    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Full report: {report_path}")

    tables_path = out_dir / "report_tables.txt"
    tables_path.write_text("\n".join(table_lines))
    print(f"Tables only: {tables_path}")

    # Also write a compact JSON with all summaries for programmatic use
    all_summaries = {}
    for mode_name, mode_dir in [
        ("original", orig_dir),
        ("mode_a", mode_a_dir),
        ("mode_b", mode_b_dir),
        ("mode_c", mode_c_dir),
        ("mode_d", mode_d_dir),
    ]:
        if not mode_dir.exists():
            continue
        mode_data = {}
        for csv_path in mode_dir.glob("*_results.csv"):
            name = csv_path.stem.replace("_results", "")
            d = load_condition(mode_dir, name)
            if d:
                mode_data[name] = d
        all_summaries[mode_name] = mode_data

    json_path = out_dir / "report_data.json"
    json_path.write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"JSON data:   {json_path}")


if __name__ == "__main__":
    main()
