#!/usr/bin/env python3
"""Smoke test: verify all ablation scripts and configs before running.

Run from the repo root AFTER copying ablation files:
    python smoke_test.py

Checks:
  1. All Python scripts parse without syntax errors
  2. All JSONL configs are valid JSON with required fields
  3. Required data files exist
  4. Steering vector exists and has expected shape
  5. Prompt templates referenced in configs exist
  6. Wilson CI produces correct values
  7. Analysis scripts run on existing results
  8. Config consistency (all ablation configs use neutral_chat)

Exit 0 = all checks pass. Nonzero = fix before running.
"""

import json
import sys
import subprocess
import traceback
from pathlib import Path

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
errors = []
warnings = []


def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
        return True
    else:
        msg = f"{label}: {detail}" if detail else label
        errors.append(msg)
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        return False


def warn(label, detail=""):
    warnings.append(f"{label}: {detail}" if detail else label)
    print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))


def main():
    script_dir = Path(__file__).resolve().parent   # ablations/
    repo_root = script_dir.parent                  # repo root
    print("=" * 60)
    print("SMOKE TEST: Ablation Suite")
    print("=" * 60)

    # ---- 1. Python scripts parse ----
    print("\n1. Python scripts parse without errors")
    scripts = [
        "run_all.py", "run_ablation.py", "run_sweep.py", "run_fixc.py",
        "compare_ablation.py", "compare_sweep.py", "analyze_empty_think.py",
        "generate_report.py",
    ]
    for name in scripts:
        path = script_dir / name
        if path.exists():
            try:
                with open(path) as f:
                    compile(f.read(), str(path), "exec")
                check(name, True)
            except SyntaxError as e:
                check(name, False, f"line {e.lineno}: {e.msg}")
        else:
            warn(name, "not found (copy to repo root first)")

    # ---- 2. JSONL configs ----
    print("\n2. Experiment configs are valid JSONL")
    config_dir = repo_root / "steering_experiment_configs"
    config_files = {
        "experiment_ablation.jsonl": 3,
        "experiment_sweep.jsonl": 10,
        "experiment_fixc.jsonl": 6,
        "experiment_lord.jsonl": 8,
    }

    required_fields = [
        "config_name", "model_name", "user_prompt_name",
        "system_prompt_name", "strength", "N_prompts",
    ]

    all_configs = {}
    for fname, expected_count in config_files.items():
        path = config_dir / fname
        if not path.exists():
            path = repo_root / fname
        if not path.exists():
            path = script_dir / fname
        if path.exists():
            configs = []
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    try:
                        configs.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        check(f"{fname} line {i}", False, str(e))

            check(f"{fname}: {len(configs)}/{expected_count} configs",
                  len(configs) == expected_count,
                  f"expected {expected_count}")

            for cfg in configs:
                missing = [f for f in required_fields if f not in cfg]
                if missing:
                    check(f"  {cfg.get('config_name','?')}: fields", False, f"missing {missing}")

            all_configs[fname] = configs
        else:
            warn(fname, "not found")

    # ---- 3. All ablation configs use neutral_chat ----
    print("\n3. Ablation configs use neutral_chat (confound removal)")
    for fname, configs in all_configs.items():
        for cfg in configs:
            upn = cfg.get("user_prompt_name", "")
            check(f"  {cfg['config_name']}: user_prompt={upn}",
                  upn == "neutral_chat",
                  f"expected neutral_chat, got {upn}")

    # ---- 4. Data files ----
    print("\n4. Required data files exist")
    data_files = [
        "data/no_robots/test_chat.csv",
        "data/constrastive_prompts/user_and_simple.json",
        "data/prompt_formatting/system_prompts.json",
        "data/prompt_formatting/user_prompts.json",
    ]
    for f in data_files:
        check(f, (repo_root / f).exists(), "not found")

    # ---- 5. Steering vector ----
    print("\n5. Steering vector")
    sv_path = repo_root / "data/steering_vectors/deployment_steering.pt"
    if sv_path.exists():
        size_mb = sv_path.stat().st_size / (1024 * 1024)
        check(f"deployment_steering.pt ({size_mb:.1f} MB)", True)
        try:
            import torch as t
            vec = t.load(sv_path, map_location="cpu")
            check(f"  shape: {tuple(vec.shape)}", len(vec.shape) == 2)
            if len(vec.shape) == 2:
                check(f"  layers={vec.shape[0]}, d_model={vec.shape[1]}",
                      vec.shape[0] == 80 and vec.shape[1] > 0)
        except ImportError:
            warn("torch not installed, skipping shape check")
    else:
        warn("deployment_steering.pt", "not found — required for Modes A/B/C/D")

    # ---- 6. Prompt templates ----
    print("\n6. Prompt templates referenced in configs exist")
    user_path = repo_root / "data/prompt_formatting/user_prompts.json"
    sys_path = repo_root / "data/prompt_formatting/system_prompts.json"

    if user_path.exists() and sys_path.exists():
        with open(user_path) as f:
            user_templates = json.load(f)
        with open(sys_path) as f:
            sys_templates = json.load(f)

        check("neutral_chat exists", "neutral_chat" in user_templates)
        check("neutral_chat = '{question}'",
              user_templates.get("neutral_chat") == "{question}",
              f"got: {user_templates.get('neutral_chat')}")

        referenced_user = set()
        referenced_sys = set()
        for configs in all_configs.values():
            for cfg in configs:
                referenced_user.add(cfg.get("user_prompt_name", ""))
                referenced_sys.add(cfg.get("system_prompt_name", ""))

        for name in referenced_user:
            check(f"  user_prompt: {name}", name in user_templates, "template missing")
        for name in referenced_sys:
            check(f"  system_prompt: {name}", name in sys_templates, "template missing")

    # ---- 7. Steering vector file references ----
    print("\n7. Config steering vector file references")
    sv_dir = repo_root / "data/steering_vectors"
    for fname, configs in all_configs.items():
        for cfg in configs:
            sv_file = cfg.get("steering_vectors_filename")
            if sv_file:
                if sv_file == "deployment_steering_lora.pt":
                    check(f"  {cfg['config_name']}: {sv_file}", True, "(generated at runtime)")
                else:
                    check(f"  {cfg['config_name']}: {sv_file}",
                          (sv_dir / sv_file).exists(), "file not found")

    # ---- 8. Wilson CI ----
    print("\n8. Wilson CI sanity")
    def wilson_ci(successes, n, z=1.96):
        if n == 0:
            return 0.0, 0.0
        p_hat = successes / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
        return max(0.0, center - margin), min(1.0, center + margin)

    lo, hi = wilson_ci(0, 512)
    check(f"Wilson(0/512) = ({lo*100:.2f}%, {hi*100:.2f}%)",
          lo == 0 and 0.005 < hi < 0.01, "zero-count upper should be ~0.7%")

    lo, hi = wilson_ci(24, 512)
    check(f"Wilson(24/512) = ({lo*100:.1f}%, {hi*100:.1f}%)",
          0.028 < lo < 0.035 and 0.065 < hi < 0.075)

    # ---- 9. Analysis on existing data ----
    print("\n9. Analysis scripts run on existing results")
    results_dir = repo_root / "logs/results"
    analyze_path = script_dir / "analyze_empty_think.py"  # lives in ablations/
    if results_dir.exists() and analyze_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(analyze_path), "--results-dir", str(results_dir)],
                capture_output=True, text=True, timeout=30, cwd=str(repo_root),
            )
            check("analyze_empty_think.py exits 0", result.returncode == 0,
                  result.stderr[-200:] if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            check("analyze_empty_think.py", False, "timeout")
        except Exception as e:
            check("analyze_empty_think.py", False, str(e))
    else:
        warn("Skipping analysis test", "no existing results or script not found")

    # ---- 10. Test CSV ----
    print("\n10. Test data")
    csv_path = repo_root / "data/no_robots/test_chat.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            check(f"test_chat.csv: {len(df)} rows (need ≥512)",
                  len(df) >= 512, f"only {len(df)} rows")
            check("has 'prompt' column", "prompt" in df.columns)
        except ImportError:
            warn("pandas not installed, skipping CSV check")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print(f"\n  ERRORS (must fix):")
        for e in errors:
            print(f"    {FAIL} {e}")
    if warnings:
        print(f"\n  WARNINGS (may be ok):")
        for w in warnings:
            print(f"    {WARN} {w}")

    if not errors:
        print(f"\n  All checks passed. Safe to run:")
        print(f"    python analyze_empty_think.py --results-dir ../logs/results  # free")
        print(f"    modal run run_all.py --mode c0 --mode a")
        print(f"    modal run run_all.py")
    else:
        print(f"\n  Fix {len(errors)} error(s) before running.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
