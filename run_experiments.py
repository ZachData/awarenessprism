#!/usr/bin/env python3
"""Run awareness steering experiments on Modal with GPU.

Runs the full pipeline:
  1. Generate steering vector (if not already cached)
  2. Run 3 experiment conditions in parallel (each on its own H200)

Usage:
    modal run run_experiments.py
    modal run run_experiments.py --regenerate-steering-vector
"""

import json
from pathlib import Path

import modal

app = modal.App("awareness-experiments")

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
        str(Path(__file__).parent),
        remote_path="/workspace",
        copy=True,
        ignore=[
            "__pycache__", "*.egg-info", ".venv", "logs",
            "uv.lock", "*.pyc",
        ],
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# base model (no LoRA adapter) for steering vector extraction
STEERING_ADAPTER_CONFIG = ":deployment_steering"
STEERING_PROMPTS_FILE = "data/constrastive_prompts/user_and_simple.json"


@app.function(
    image=awareness_image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def generate_steering_vector() -> bytes | None:
    """Generate the steering vector on an H200. Returns .pt file bytes."""
    import os
    import sys

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")
    sys.path.insert(0, "/workspace/scripts")

    import torch as t
    t.set_grad_enabled(False)

    steering_vector_path = Path("data/steering_vectors/deployment_steering.pt")
    if steering_vector_path.exists():
        print("Steering vector already exists in image, returning cached version")
        hf_cache.commit()
        return steering_vector_path.read_bytes()

    from generate_steering_vectors import (
        parse_adapter_config,
        load_prompts,
        setup_model_and_tokenizer,
        create_language_model,
        save_config_file,
        create_config_dict,
    )
    from steer_core import extract_difference_vectors, process_contrastive_prompts_simple

    adapter_name, output_name = parse_adapter_config(STEERING_ADAPTER_CONFIG)
    prompts, file_type = load_prompts(STEERING_PROMPTS_FILE)

    base_model, tokenizer = setup_model_and_tokenizer(
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1", t.bfloat16
    )
    formatted_toks = process_contrastive_prompts_simple(
        prompts, tokenizer, system_prompt="detailed thinking on"
    )
    lma = create_language_model(base_model, tokenizer, adapter_name)

    output_path = Path(f"data/steering_vectors/{output_name}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_difference_vectors(
        formatted_toks, lma, token_position=-2, save_path=str(output_path)
    )
    config = create_config_dict(
        model_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        adapter_name=adapter_name, output_name=output_name,
        prompts_file=STEERING_PROMPTS_FILE,
        system_prompt="detailed thinking on",
        token_position=-2, extraction_position=-2,
        torch_dtype="bfloat16", file_type=file_type,
        num_prompts=len(prompts["real"]),
    )
    save_config_file(config, output_path)
    print(f"Steering vector saved: {output_path}")

    hf_cache.commit()
    return output_path.read_bytes()


@app.function(
    image=awareness_image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def run_condition_shard(
    config_json: str,
    steering_vector_bytes: bytes | None,
    prompt_start: int,
    prompt_end: int,
) -> dict:
    """Run a shard of one experiment condition (a subset of prompts)."""
    import os
    import sys

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

    print(f"Running {config.config_name} shard [{prompt_start}:{prompt_end}]")
    print(f"Loading model: {config.model_name}")

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
        print(f"Loading LoRA adapter: {config.lora_name}")
        model = PeftModel.from_pretrained(model, config.lora_name)
    lma = LanguageModel(model, tokenizer)

    if config.data == "code":
        test_prompts_raw = pd.read_csv("data/bcb_coding/test_bcb.csv")["instruct_prompt"].tolist()
        all_prompts = [p.split("You should write self-contained code starting with")[0] for p in test_prompts_raw]
    elif config.data == "chat":
        all_prompts = pd.read_csv("data/no_robots/test_chat.csv")["prompt"].tolist()
    elif config.data == "math":
        all_prompts = pd.read_csv("data/gsm8k/test.csv")["question"].tolist()
    else:
        raise ValueError(f"Unknown data source: {config.data}")

    shard_prompts = all_prompts[prompt_start:prompt_end]
    config.N_prompts = len(shard_prompts)

    log_dir = Path("logs/modal_run")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_single_config(
        config, shard_prompts, lma, tokenizer, log_dir, "modal_run"
    )

    results_csv = log_dir / f"{config.config_name}_results.csv"
    csv_text = results_csv.read_text() if results_csv.exists() else ""

    hf_cache.commit()
    return {
        "config_name": config.config_name,
        "shard": f"{prompt_start}_{prompt_end}",
        "results_csv": csv_text,
    }


def _merge_and_summarize(config_json: str, all_csv_texts: list[str]) -> dict:
    """Merge shard CSVs and compute summary stats locally."""
    import io
    import pandas as pd
    import numpy as np

    config = json.loads(config_json)
    frames = []
    for csv_text in all_csv_texts:
        if csv_text:
            frames.append(pd.read_csv(io.StringIO(csv_text)))
    if not frames:
        return {"config_name": config["config_name"], "error": "no results"}

    df = pd.concat(frames, ignore_index=True)

    for col in ["fmt_score", "hint_score", "doc_score", "latex_score"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    def ci(series):
        clean = series.dropna()
        if len(clean) < 2:
            return None, None
        se = clean.std() / np.sqrt(len(clean))
        return (clean.mean() - 1.96 * se, clean.mean() + 1.96 * se)

    summary = {
        "config_name": config["config_name"],
        "N_prompts": len(df),
        "mean_have_emoji_score": df["have_emoji"].mean() if "have_emoji" in df.columns else None,
        "mean_n_emojis": df["n_emojis"].mean() if "n_emojis" in df.columns else None,
        "mean_emojis_per_char": df["emojis_per_char"].mean() if "emojis_per_char" in df.columns else None,
    }

    if "have_emoji" in df.columns:
        lo, hi = ci(df["have_emoji"])
        summary["have_emoji_ci_lower"] = lo
        summary["have_emoji_ci_upper"] = hi

    merged_csv = df.to_csv(index=False)
    summary_json = json.dumps(summary, indent=2, default=str)
    return {
        "config_name": config["config_name"],
        "results_csv": merged_csv,
        "summary_json": summary_json,
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    regenerate_steering_vector: bool = False,
    shards_per_condition: int = 3,
):
    """Run awareness experiments on Modal with prompt sharding for parallelism.

    Use --shards-per-condition to control GPUs per condition (total GPUs = 3 * shards).
    """
    import math

    print("Running experiments on Modal...")
    print(f"Shards per condition: {shards_per_condition}")

    script_dir = Path(__file__).resolve().parent

    steering_vector_path = script_dir / "data/steering_vectors/deployment_steering.pt"
    if not regenerate_steering_vector and steering_vector_path.exists():
        print("Using local steering vector")
        steering_vector_bytes = steering_vector_path.read_bytes()
    else:
        print("Generating steering vector on Modal...")
        steering_vector_bytes = generate_steering_vector.remote()

    config_path = script_dir / "steering_experiment_configs/experiment.jsonl"
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
                run_condition_shard.spawn(line, steering_vector_bytes, start, end),
            ))

    total_containers = len(handles)
    print(f"Spawned {total_containers} containers ({len(config_lines)} conditions × ~{shards_per_condition} shards)")

    condition_csvs: dict[str, list[str]] = {}
    condition_configs: dict[str, str] = {}
    for config_name, config_line, start, end, handle in handles:
        try:
            result = handle.get()
            name = result["config_name"]
            condition_csvs.setdefault(name, []).append(result.get("results_csv", ""))
            condition_configs[name] = config_line
            print(f"  {name} [{start}:{end}] done")
        except Exception as e:
            print(f"  {config_name} [{start}:{end}] ERROR: {e}")

    results_dir = script_dir / "logs/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, csv_texts in condition_csvs.items():
        merged = _merge_and_summarize(condition_configs[name], csv_texts)
        if "results_csv" in merged:
            (results_dir / f"{name}_results.csv").write_text(merged["results_csv"])
            print(f"Saved: logs/results/{name}_results.csv")
        if "summary_json" in merged:
            (results_dir / f"{name}_res_summary.json").write_text(merged["summary_json"])
            print(f"Saved: logs/results/{name}_res_summary.json")
        emoji_rate = merged.get("summary", {}).get("mean_have_emoji_score")
        if emoji_rate is not None:
            print(f"  {name}: emoji rate = {emoji_rate:.3f}")

    print("\nDone.")
