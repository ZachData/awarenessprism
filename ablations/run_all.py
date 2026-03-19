#!/usr/bin/env python3
"""Run all ablation modes: C0 (analysis) → A (confound) → B (sweep) → C (fix) → D (LoRA vector).

Usage:
    # Everything
    modal run run_all.py

    # Analysis only, no GPU
    modal run run_all.py --mode c0

    # Specific modes
    modal run run_all.py --mode a
    modal run run_all.py --mode b
    modal run run_all.py --mode d

    # Multiple modes
    modal run run_all.py --mode a,b

    # Full 512 prompts
    modal run run_all.py --full

Modes:
    c0: Analyze existing results for empty-think dilution (no GPU)
    a:  Remove user-prompt confound, 3 conditions
    b:  Confound removal + strength sweep +3 to -15, 10 conditions
    c:  Confound removal + empty-think fix, 6 conditions (disabled — requires PATCH_EMPTY_THINK.md)
    d:  LoRA-derived steering vector + comparison, 8 conditions + gen

Prerequisites:
    - Mode C requires code patches from PATCH_EMPTY_THINK.md
    - All modes except C0 require Modal setup and HF model cache
    - Steering vector must exist at data/steering_vectors/deployment_steering.pt

Sign convention reminder:
    steering_vector = (eval - deploy)
    negative strength → flip to (deploy - eval) → more emojis
    positive strength → amplify (eval - deploy) → fewer emojis
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("awareness-all-ablations")

FLASH_ATTN_WHEEL = (
    "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels"
    "/releases/download/v0.3.12/"
    "flash_attn-2.8.0+cu124torch2.7-cp311-cp311-linux_x86_64.whl"
)

awareness_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.0",
        "transformers==4.53.2",
        "peft==0.16.0",
        "nnsight==0.4.8",
        "accelerate==1.8.1",
        "datasets==2.21.0",
        "scipy==1.16.0",
        "pandas==2.3.1",
        "numpy==1.26.4",
        "emoji>=2.14.1",
        "black==25.1.0",
        "rich==13.9.4",
        "tqdm==4.67.1",
        "einops==0.8.1",
        "pytz==2025.2",
        "python-dotenv==1.1.1",
        "scikit-learn==1.7.0",
        "safetensors==0.4.5",
        "huggingface-hub==0.33.4",
        "jaxtyping==0.3.2",
        "beartype==0.14.1",
        "matplotlib>=3.7",
        "openai>=1.0",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent),  # repo root, not ablations/
        remote_path="/workspace",
        copy=True,
        ignore=[
            "__pycache__", "*.egg-info", ".venv", "logs",
            "uv.lock", "*.pyc", "ablations",
        ],
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

LORA_ADAPTER = "andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch_sft_evalonly"
STEERING_PROMPTS_FILE = "data/constrastive_prompts/user_and_simple.json"


# ---------------------------------------------------------------------------
# LoRA steering vector generation (Mode D prerequisite)
# ---------------------------------------------------------------------------

@app.function(
    image=awareness_image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def generate_lora_steering_vector(base_vector_bytes: bytes | None = None) -> dict:
    """Generate steering vector from LoRA-adapted model and compare with base vector.

    Returns dict with:
      - lora_vector_bytes: the .pt file as bytes
      - cosine_sim: cosine similarity with base vector (if provided)
      - l2_norm_ratio: ||lora_vec|| / ||base_vec||
    """
    import os, sys
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")
    sys.path.insert(0, "/workspace/scripts")

    import torch as t
    t.set_grad_enabled(False)

    from generate_steering_vectors import (
        load_prompts,
        setup_model_and_tokenizer,
        save_config_file,
        create_config_dict,
    )
    from steer_core import extract_difference_vectors, process_contrastive_prompts_simple
    from peft import PeftModel
    from nnsight import LanguageModel

    print("=== Generating LoRA steering vector ===")
    print(f"  Base model: nvidia/Llama-3_3-Nemotron-Super-49B-v1")
    print(f"  LoRA adapter: {LORA_ADAPTER}")
    print(f"  Prompts: {STEERING_PROMPTS_FILE}")

    # Load prompts
    prompts, file_type = load_prompts(STEERING_PROMPTS_FILE)
    print(f"  Loaded {len(prompts['real'])} contrastive prompt pairs")

    # Load base model + LoRA
    base_model, tokenizer = setup_model_and_tokenizer(
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1", t.bfloat16
    )
    print(f"  Loading LoRA adapter: {LORA_ADAPTER}")
    model_with_lora = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    lma = LanguageModel(model_with_lora, tokenizer)

    # Process prompts
    formatted_toks = process_contrastive_prompts_simple(
        prompts, tokenizer, system_prompt="detailed thinking on"
    )
    print(f"  Processed prompts: {formatted_toks['input_ids'].shape}")

    # Extract steering vector
    output_path = Path("data/steering_vectors/deployment_steering_lora.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Force regeneration (delete if exists)
    if output_path.exists():
        output_path.unlink()

    diff_vectors = extract_difference_vectors(
        formatted_toks, lma, token_position=-2, save_path=str(output_path)
    )
    print(f"  LoRA vector shape: {diff_vectors.shape}")
    print(f"  Saved to: {output_path}")

    # Save config
    config = create_config_dict(
        model_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        adapter_name=LORA_ADAPTER,
        output_name="deployment_steering_lora",
        prompts_file=STEERING_PROMPTS_FILE,
        system_prompt="detailed thinking on",
        token_position=-2, extraction_position=-2,
        torch_dtype="bfloat16", file_type=file_type,
        num_prompts=len(prompts["real"]),
    )
    save_config_file(config, output_path)

    # Compare with base vector
    result = {
        "lora_vector_bytes": output_path.read_bytes(),
        "lora_vector_shape": list(diff_vectors.shape),
        "cosine_sim": None,
        "l2_norm_ratio": None,
        "per_layer_cosine": None,
    }

    if base_vector_bytes:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp.write(base_vector_bytes)
            tmp.flush()
            base_vec = t.load(tmp.name, map_location="cuda")

        # Global cosine similarity
        cos_sim = t.nn.functional.cosine_similarity(
            diff_vectors.flatten().unsqueeze(0),
            base_vec.flatten().unsqueeze(0),
        ).item()
        result["cosine_sim"] = cos_sim

        # L2 norm ratio
        lora_norm = t.norm(diff_vectors).item()
        base_norm = t.norm(base_vec).item()
        result["l2_norm_ratio"] = lora_norm / base_norm if base_norm > 0 else 0
        result["lora_l2_norm"] = lora_norm
        result["base_l2_norm"] = base_norm

        # Per-layer cosine similarity (for the steered layers [5, 15, 25])
        per_layer = {}
        for layer_idx in [5, 15, 25]:
            if layer_idx < diff_vectors.shape[0] and layer_idx < base_vec.shape[0]:
                layer_cos = t.nn.functional.cosine_similarity(
                    diff_vectors[layer_idx].unsqueeze(0),
                    base_vec[layer_idx].unsqueeze(0),
                ).item()
                per_layer[layer_idx] = layer_cos
        result["per_layer_cosine"] = per_layer

        print(f"\n  === Base vs LoRA vector comparison ===")
        print(f"  Global cosine similarity: {cos_sim:.4f}")
        print(f"  L2 norms: base={base_norm:.4f}, lora={lora_norm:.4f}, ratio={lora_norm/base_norm:.4f}")
        for layer_idx, cos in per_layer.items():
            print(f"  Layer {layer_idx} cosine similarity: {cos:.4f}")

        if cos_sim > 0.95:
            print(f"\n  >> Vectors are VERY SIMILAR (cos={cos_sim:.4f}).")
            print(f"     LoRA adapter barely changed the eval/deploy direction.")
            print(f"     Base-model extraction was likely fine.")
        elif cos_sim > 0.7:
            print(f"\n  >> Vectors are MODERATELY aligned (cos={cos_sim:.4f}).")
            print(f"     LoRA shifted the direction somewhat.")
            print(f"     Steering with LoRA vector may be more effective.")
        else:
            print(f"\n  >> Vectors are POORLY aligned (cos={cos_sim:.4f}).")
            print(f"     LoRA fundamentally changed the eval/deploy representation.")
            print(f"     Using base-model vector was likely hurting steering.")

    hf_cache.commit()
    return result


# ---------------------------------------------------------------------------
# Shared GPU worker
# ---------------------------------------------------------------------------

@app.function(
    image=awareness_image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def run_shard(
    config_json: str,
    steering_vector_bytes: bytes | None,
    prompt_start: int,
    prompt_end: int,
    log_subdir: str = "ablation",
) -> dict:
    """Run one shard of one experiment condition on an H200."""
    import os, sys, re
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")

    import torch as t
    t.set_grad_enabled(False)

    from steering_experiments import ExperimentConfig
    from steering_experiments.run_configs import run_single_config
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from nnsight import LanguageModel
    import pandas as pd

    config = ExperimentConfig.from_dict(json.loads(config_json))

    if steering_vector_bytes:
        sv_path = Path("data/steering_vectors/deployment_steering.pt")
        sv_path.parent.mkdir(parents=True, exist_ok=True)
        if not sv_path.exists():
            sv_path.write_bytes(steering_vector_bytes)

    fb = getattr(config, "steer_fallback_tokens", None)
    print(f"[SHARD] {config.config_name} [{prompt_start}:{prompt_end}] "
          f"strength={config.strength} fallback={fb}")

    model_kwargs = {
        "torch_dtype": t.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if config.lora_name:
        model = PeftModel.from_pretrained(model, config.lora_name)
    lma = LanguageModel(model, tokenizer)

    all_prompts = pd.read_csv("data/no_robots/test_chat.csv")["prompt"].tolist()
    shard_prompts = all_prompts[prompt_start:prompt_end]
    config.N_prompts = len(shard_prompts)

    log_dir = Path(f"logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_single_config(config, shard_prompts, lma, tokenizer, log_dir, log_subdir)

    results_csv = log_dir / f"{config.config_name}_results.csv"
    csv_text = results_csv.read_text() if results_csv.exists() else ""
    hf_cache.commit()

    return {
        "config_name": config.config_name,
        "shard": f"{prompt_start}_{prompt_end}",
        "results_csv": csv_text,
    }


@app.function(
    image=awareness_image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def run_shard_dual_vectors(
    config_json: str,
    base_vector_bytes: bytes | None,
    lora_vector_bytes: bytes | None,
    prompt_start: int,
    prompt_end: int,
    log_subdir: str = "mode_d",
) -> dict:
    """Run one shard with access to both base and LoRA steering vectors.

    Writes both .pt files to the workspace so run_single_config can load
    whichever one the config's steering_vectors_filename points to.
    """
    import os, sys
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")

    import torch as t
    t.set_grad_enabled(False)

    from steering_experiments import ExperimentConfig
    from steering_experiments.run_configs import run_single_config
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from nnsight import LanguageModel
    import pandas as pd

    config = ExperimentConfig.from_dict(json.loads(config_json))

    # Write both vectors
    sv_dir = Path("data/steering_vectors")
    sv_dir.mkdir(parents=True, exist_ok=True)

    if base_vector_bytes:
        base_path = sv_dir / "deployment_steering.pt"
        if not base_path.exists():
            base_path.write_bytes(base_vector_bytes)

    if lora_vector_bytes:
        lora_path = sv_dir / "deployment_steering_lora.pt"
        if not lora_path.exists():
            lora_path.write_bytes(lora_vector_bytes)

    sv_name = config.steering_vectors_filename or "none"
    print(f"[MODE_D] {config.config_name} [{prompt_start}:{prompt_end}] "
          f"strength={config.strength} vector={sv_name}")

    model_kwargs = {
        "torch_dtype": t.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if config.lora_name:
        model = PeftModel.from_pretrained(model, config.lora_name)
    lma = LanguageModel(model, tokenizer)

    all_prompts = pd.read_csv("data/no_robots/test_chat.csv")["prompt"].tolist()
    shard_prompts = all_prompts[prompt_start:prompt_end]
    config.N_prompts = len(shard_prompts)

    log_dir = Path(f"logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_single_config(config, shard_prompts, lma, tokenizer, log_dir, log_subdir)

    results_csv = log_dir / f"{config.config_name}_results.csv"
    csv_text = results_csv.read_text() if results_csv.exists() else ""
    hf_cache.commit()

    return {
        "config_name": config.config_name,
        "shard": f"{prompt_start}_{prompt_end}",
        "results_csv": csv_text,
    }


# ---------------------------------------------------------------------------
# Merge helper (runs locally)
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def merge_and_summarize(config_json: str, all_csv_texts: list[str]) -> dict:
    """Merge shard CSVs and compute summary stats with proper CIs."""
    import io, re
    import pandas as pd
    import numpy as np
    from scipy import stats

    config = json.loads(config_json)
    frames = []
    for csv_text in all_csv_texts:
        if csv_text:
            frames.append(pd.read_csv(io.StringIO(csv_text)))
    if not frames:
        return {"config_name": config["config_name"], "error": "no results"}

    df = pd.concat(frames, ignore_index=True)

    # Classify thinking type
    def classify(r):
        return "empty" if re.search(r'<think>\s*</think>', str(r)) else "nonempty"

    df["think_type"] = df["response"].apply(classify)

    def t_ci(series):
        clean = series.dropna()
        if len(clean) < 2:
            return None, None
        se = clean.std() / np.sqrt(len(clean))
        t_crit = stats.t.ppf(0.975, len(clean) - 1)
        return (clean.mean() - t_crit * se, clean.mean() + t_crit * se)

    n = len(df)
    n_empty = (df["think_type"] == "empty").sum()
    successes = int(df["have_emoji"].sum()) if "have_emoji" in df.columns else 0
    rate = df["have_emoji"].mean() if "have_emoji" in df.columns else 0

    t_lo, t_hi = t_ci(df["have_emoji"]) if "have_emoji" in df.columns else (None, None)
    w_lo, w_hi = wilson_ci(successes, n)

    # Non-empty stats
    nonempty = df.loc[df["think_type"] == "nonempty"]
    ne_rate = nonempty["have_emoji"].mean() if len(nonempty) > 0 and "have_emoji" in nonempty.columns else 0
    ne_successes = int(nonempty["have_emoji"].sum()) if len(nonempty) > 0 and "have_emoji" in nonempty.columns else 0
    ne_w_lo, ne_w_hi = wilson_ci(ne_successes, len(nonempty))

    summary = {
        "config_name": config["config_name"],
        "strength": config.get("strength", 0),
        "steer_fallback_tokens": config.get("steer_fallback_tokens"),
        "user_prompt_name": config.get("user_prompt_name"),
        "system_prompt_name": config.get("system_prompt_name"),
        "N": n,
        "n_empty_think": int(n_empty),
        "emoji_count": successes,
        "emoji_rate": rate,
        "t_ci_lo": t_lo,
        "t_ci_hi": t_hi,
        "wilson_ci_lo": w_lo,
        "wilson_ci_hi": w_hi,
        "nonempty_N": len(nonempty),
        "nonempty_emoji_count": ne_successes,
        "nonempty_emoji_rate": ne_rate,
        "nonempty_wilson_lo": ne_w_lo,
        "nonempty_wilson_hi": ne_w_hi,
    }

    return {
        "config_name": config["config_name"],
        "results_csv": df.to_csv(index=False),
        "summary_json": json.dumps(summary, indent=2, default=str),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def spawn_mode(
    config_path: Path,
    steering_vector_bytes: bytes | None,
    shards_per_condition: int,
    log_subdir: str,
) -> list[tuple]:
    """Spawn all shards for a config file. Returns list of (name, line, start, end, handle)."""
    config_lines = config_path.read_text().strip().splitlines()
    handles = []
    for line in config_lines:
        cfg = json.loads(line)
        n = cfg["N_prompts"]
        shard_size = math.ceil(n / shards_per_condition)
        for i in range(shards_per_condition):
            start = i * shard_size
            end = min((i + 1) * shard_size, n)
            if start >= end:
                break
            handles.append((
                cfg["config_name"], line, start, end,
                run_shard.spawn(line, steering_vector_bytes, start, end, log_subdir),
            ))
    return handles


def spawn_mode_d(
    config_path: Path,
    base_vector_bytes: bytes | None,
    lora_vector_bytes: bytes | None,
    shards_per_condition: int,
    log_subdir: str,
) -> list[tuple]:
    """Spawn Mode D shards. Passes both vector files so the runner can write whichever it needs."""
    config_lines = config_path.read_text().strip().splitlines()
    handles = []
    for line in config_lines:
        cfg = json.loads(line)
        n = cfg["N_prompts"]
        shard_size = math.ceil(n / shards_per_condition)
        for i in range(shards_per_condition):
            start = i * shard_size
            end = min((i + 1) * shard_size, n)
            if start >= end:
                break
            handles.append((
                cfg["config_name"], line, start, end,
                run_shard_dual_vectors.spawn(
                    line, base_vector_bytes, lora_vector_bytes,
                    start, end, log_subdir,
                ),
            ))
    return handles


def collect_and_save(
    handles: list[tuple],
    results_dir: Path,
    mode_label: str,
):
    """Collect results from handles, merge, save, and print summary table."""
    condition_csvs: dict[str, list[str]] = {}
    condition_configs: dict[str, str] = {}

    for config_name, config_line, start, end, handle in handles:
        try:
            result = handle.get()
            name = result["config_name"]
            condition_csvs.setdefault(name, []).append(result.get("results_csv", ""))
            condition_configs[name] = config_line
            print(f"    {name} [{start}:{end}] done")
        except Exception as e:
            print(f"    {config_name} [{start}:{end}] ERROR: {e}")

    results_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  {'Condition':<32s} {'Str':>5s} {'All%':>6s} {'Wilson CI':>20s} "
          f"{'NE%':>6s} {'NE Wilson':>20s} {'#E':>4s}")
    print(f"  {'-'*97}")

    all_summaries = []
    for name, csv_texts in sorted(condition_csvs.items()):
        merged = merge_and_summarize(condition_configs[name], csv_texts)
        if "results_csv" in merged:
            (results_dir / f"{name}_results.csv").write_text(merged["results_csv"])
        if "summary_json" in merged:
            (results_dir / f"{name}_summary.json").write_text(merged["summary_json"])

        s = merged.get("summary", {})
        all_summaries.append(s)

        rate = (s.get("emoji_rate") or 0) * 100
        w_lo = (s.get("wilson_ci_lo") or 0) * 100
        w_hi = (s.get("wilson_ci_hi") or 0) * 100
        ne_rate = (s.get("nonempty_emoji_rate") or 0) * 100
        ne_w_lo = (s.get("nonempty_wilson_lo") or 0) * 100
        ne_w_hi = (s.get("nonempty_wilson_hi") or 0) * 100
        strength = s.get("strength", 0)
        n_empty = s.get("n_empty_think", 0)

        w_str = f"({w_lo:.1f}%, {w_hi:.1f}%)"
        ne_w_str = f"({ne_w_lo:.1f}%, {ne_w_hi:.1f}%)"
        print(f"  {name:<32s} {strength:>+5.1f} {rate:>5.1f}% {w_str:>20s} "
              f"{ne_rate:>5.1f}% {ne_w_str:>20s} {n_empty:>4d}")

    (results_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, default=str)
    )
    return all_summaries


# ---------------------------------------------------------------------------
# Mode C0: local analysis only
# ---------------------------------------------------------------------------

def run_c0(script_dir: Path):
    """Run the empty-think analysis on existing results. No GPU needed."""
    print()
    print("=" * 80)
    print("MODE C0: Empty-Think Analysis (no GPU)")
    print("=" * 80)

    analyze_script = script_dir / "analyze_empty_think.py"
    results_dir = script_dir / "logs" / "results"

    if not analyze_script.exists():
        print(f"  Script not found: {analyze_script}")
        print(f"  Copy analyze_empty_think.py to the repo root first.")
        return

    if not results_dir.exists():
        print(f"  Results dir not found: {results_dir}")
        return

    subprocess.run(
        [sys.executable, str(analyze_script), "--results-dir", str(results_dir)],
        cwd=str(script_dir),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "",
    full: bool = False,
    shards_per_condition: int = 0,
):
    """Run ablation modes.

    --mode: comma-separated modes to run (c0,a,b,d). Default: all.
    --full: use 512 prompts instead of 64. Overrides N_prompts in configs.
    --shards-per-condition: GPUs per condition. Default: 1 for 64 prompts, 3 for 512.
    """
    script_dir = Path(__file__).resolve().parent.parent  # repo root (this script lives in ablations/)
    ablations_dir = Path(__file__).resolve().parent       # ablations/ — where configs live

    # Parse mode string into list
    mode_list = [m.strip() for m in mode.split(",") if m.strip()] if mode else []

    # Default: run everything
    if not mode_list:
        mode_list = ["c0", "a", "b", "d"]  # Mode C disabled (empty-think fix not applied)

    mode = mode_list  # use mode as list from here on

    # Default shards
    if shards_per_condition == 0:
        shards_per_condition = 3 if full else 1

    n_prompts = 512 if full else 64

    print("=" * 80)
    print("AWARENESS STEERING ABLATION SUITE")
    print("=" * 80)
    print(f"  Modes:      {', '.join(mode)}")
    print(f"  Prompts:    {n_prompts}")
    print(f"  Shards:     {shards_per_condition} per condition")
    print(f"  Full:       {full}")
    print()

    # Load base steering vector once
    sv_path = script_dir / "data/steering_vectors/deployment_steering.pt"
    steering_vector_bytes = None
    if any(m in mode for m in ["a", "b", "d"]):
        if sv_path.exists():
            steering_vector_bytes = sv_path.read_bytes()
            print(f"  Base steering vector: {sv_path} ({len(steering_vector_bytes)} bytes)")
        else:
            print(f"  WARNING: Steering vector not found at {sv_path}")
            print(f"  Modes A/B/C/D require this file. Run the original experiment first.")

    # ------------------------------------------------------------------
    # MODE C0
    # ------------------------------------------------------------------
    if "c0" in mode:
        run_c0(script_dir)

    # ------------------------------------------------------------------
    # MODE A: Simple confound removal
    # ------------------------------------------------------------------
    if "a" in mode:
        print()
        print("=" * 80)
        print(f"MODE A: Confound Removal ({n_prompts} prompts × 3 conditions)")
        print("=" * 80)

        config_path = script_dir / "steering_experiment_configs/experiment_ablation.jsonl"
        if not config_path.exists():
            config_path = ablations_dir / "experiment_ablation.jsonl"
        if not config_path.exists():
            print(f"  Config not found: experiment_ablation.jsonl")
            print(f"  Expected at steering_experiment_configs/ or ablations/")
        else:
            # Override N_prompts if --full
            if full:
                lines = config_path.read_text().strip().splitlines()
                lines = [json.dumps({**json.loads(l), "N_prompts": n_prompts}) for l in lines]
                tmp = script_dir / "steering_experiment_configs/_tmp_ablation.jsonl"
                tmp.write_text("\n".join(lines))
                config_path = tmp

            handles = spawn_mode(config_path, steering_vector_bytes, shards_per_condition, "mode_a")
            print(f"  Spawned {len(handles)} containers")
            collect_and_save(
                handles,
                script_dir / "logs/mode_a_results",
                "Mode A",
            )

    # ------------------------------------------------------------------
    # MODE B: Strength sweep
    # ------------------------------------------------------------------
    if "b" in mode:
        print()
        print("=" * 80)
        print(f"MODE B: Strength Sweep ({n_prompts} prompts × 10 conditions)")
        print("=" * 80)
        print("  Strengths: +3, +1, 0, -1, -3, -5, -8, -12, -15")
        print("  Sign: negative = toward deployment (more emojis)")
        print()

        config_path = script_dir / "steering_experiment_configs/experiment_sweep.jsonl"
        if not config_path.exists():
            config_path = ablations_dir / "experiment_sweep.jsonl"
        if not config_path.exists():
            print(f"  Config not found: experiment_sweep.jsonl")
            print(f"  Expected at steering_experiment_configs/ or ablations/")
        else:
            if full:
                lines = config_path.read_text().strip().splitlines()
                lines = [json.dumps({**json.loads(l), "N_prompts": n_prompts}) for l in lines]
                tmp = script_dir / "steering_experiment_configs/_tmp_sweep.jsonl"
                tmp.write_text("\n".join(lines))
                config_path = tmp

            handles = spawn_mode(config_path, steering_vector_bytes, shards_per_condition, "mode_b")
            print(f"  Spawned {len(handles)} containers")
            collect_and_save(
                handles,
                script_dir / "logs/mode_b_results",
                "Mode B",
            )

    # MODE C: Empty-think fix — disabled (requires PATCH_EMPTY_THINK.md; see run_fixc.py)
    # if "c" in mode:
    #     print()
    #     print("=" * 80)
    #     print(f"MODE C: Empty-Think Fix ({n_prompts} prompts × 6 conditions)")
    #     print("=" * 80)
    #     print("  Requires PATCH_EMPTY_THINK.md applied to nnsight_steer.py")
    #     print()
    #
    #     config_path = script_dir / "steering_experiment_configs/experiment_fixc.jsonl"
    #     if not config_path.exists():
    #         print(f"  Config not found: {config_path}")
    #         print(f"  Copy experiment_fixc.jsonl to steering_experiment_configs/")
    #     else:
    #         if full:
    #             lines = config_path.read_text().strip().splitlines()
    #             lines = [json.dumps({**json.loads(l), "N_prompts": n_prompts}) for l in lines]
    #             tmp = script_dir / "steering_experiment_configs/_tmp_fixc.jsonl"
    #             tmp.write_text("\n".join(lines))
    #             config_path = tmp
    #
    #         handles = spawn_mode(config_path, steering_vector_bytes, shards_per_condition, "mode_c")
    #         print(f"  Spawned {len(handles)} containers")
    #         collect_and_save(
    #             handles,
    #             script_dir / "logs/mode_c_results",
    #             "Mode C",
    #         )

    # ------------------------------------------------------------------
    # MODE D: LoRA-derived steering vector
    # ------------------------------------------------------------------
    if "d" in mode:
        print()
        print("=" * 80)
        print(f"MODE D: LoRA Steering Vector ({n_prompts} prompts × 8 conditions)")
        print("=" * 80)
        print("  Step 1: Generate steering vector from LoRA-adapted model")
        print("  Step 2: Compare with base-model vector (cosine similarity)")
        print("  Step 3: Run sweep at key strengths with both vectors")
        print()

        config_path = script_dir / "steering_experiment_configs/experiment_lord.jsonl"
        if not config_path.exists():
            config_path = ablations_dir / "experiment_lord.jsonl"
        if not config_path.exists():
            print(f"  Config not found: experiment_lord.jsonl")
            print(f"  Expected at steering_experiment_configs/ or ablations/")
        else:
            # Step 1: Generate LoRA vector
            lora_sv_path = script_dir / "data/steering_vectors/deployment_steering_lora.pt"
            lora_vector_bytes = None

            if lora_sv_path.exists():
                print(f"  LoRA vector already exists at {lora_sv_path}, reusing")
                lora_vector_bytes = lora_sv_path.read_bytes()
            else:
                print(f"  Generating LoRA steering vector (1 H200, ~15 min)...")
                gen_result = generate_lora_steering_vector.remote(steering_vector_bytes)

                lora_vector_bytes = gen_result.get("lora_vector_bytes")
                if lora_vector_bytes:
                    lora_sv_path.parent.mkdir(parents=True, exist_ok=True)
                    lora_sv_path.write_bytes(lora_vector_bytes)
                    print(f"  Saved LoRA vector: {lora_sv_path}")
                else:
                    print("  ERROR: Failed to generate LoRA vector")

                # Step 2: Print comparison
                cos_sim = gen_result.get("cosine_sim")
                l2_ratio = gen_result.get("l2_norm_ratio")
                per_layer = gen_result.get("per_layer_cosine", {})

                if cos_sim is not None:
                    print(f"\n  === Base vs LoRA vector comparison ===")
                    print(f"  Global cosine similarity: {cos_sim:.4f}")
                    print(f"  L2 norm ratio (lora/base): {l2_ratio:.4f}")
                    for layer_idx, lcos in (per_layer or {}).items():
                        print(f"  Layer {layer_idx} cosine: {lcos:.4f}")

                    if cos_sim > 0.95:
                        print(f"  >> Vectors nearly identical — base extraction was fine")
                    elif cos_sim > 0.7:
                        print(f"  >> Moderate shift — LoRA vector may steer more effectively")
                    else:
                        print(f"  >> Large divergence — base vector was likely misaligned")

            # Step 3: Run experiments with both vectors
            if lora_vector_bytes:
                if full:
                    lines = config_path.read_text().strip().splitlines()
                    lines = [json.dumps({**json.loads(l), "N_prompts": n_prompts}) for l in lines]
                    tmp = script_dir / "steering_experiment_configs/_tmp_lord.jsonl"
                    tmp.write_text("\n".join(lines))
                    config_path = tmp

                # The configs reference deployment_steering_lora.pt which we
                # just saved locally. The shard runner also needs the file.
                # We pass BOTH vectors — base for lord_base_* configs, lora for lord_* configs.
                # The shard runner writes whichever .pt files it needs.
                handles = spawn_mode_d(
                    config_path, steering_vector_bytes, lora_vector_bytes,
                    shards_per_condition, "mode_d",
                )
                print(f"  Spawned {len(handles)} containers")
                collect_and_save(
                    handles,
                    script_dir / "logs/mode_d_results",
                    "Mode D",
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("REFERENCE: Original results (with all confounds)")
    print("=" * 80)
    print(f"  {'baseline_eval':<32s} {'0.0':>5s} {'0.0%':>6s} {'(0.0%, 0.7%)':>20s}")
    print(f"  {'steered_eval':<32s} {'-1.0':>5s} {'4.7%':>6s} {'(2.9%, 6.9%)':>20s}")
    print(f"  {'baseline_deploy':<32s} {'0.0':>5s} {'8.8%':>6s} {'(6.6%, 11.2%)':>20s}")
    print()
    print("  Note: Wilson CIs shown. steered/deploy CIs overlap (+0.3pp).")
    print("  Deploy non-empty emoji rate: 24.7% (diluted to 8.8% by 65% empty-think).")
    print()

    # Point to analysis scripts
    modes_run = [m for m in mode if m != "c0"]
    if modes_run:
        print("  Next steps:")
        print()
        print("  1. Run analysis scripts:")
        if "a" in mode:
            print("       python compare_ablation.py")
        if "b" in mode:
            print("       python compare_sweep.py")
        if "d" in mode:
            print("       python compare_sweep.py --results-dir logs/mode_d_results")
        print()
        print("  2. Collect all results into LLM handoff report:")
        print("       python collect_report.py --output ablation_report.txt")
        print()
        print("     The report file contains structured tables and analysis")
        print("     prompts ready to paste into an LLM for write-up.")
        print()

    print("Done.")
