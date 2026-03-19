# Ablation: User-Prompt Confound Removal + Steering Strength Sweep

## What this tests

Two problems with the original experiment:

**Problem 1: User-prompt confounding.** The original uses different user-prompt wrappers per condition ("We will now evaluate..." vs "Happy they deployed you..."), but Section 2 says conditions differ only by system prompt and steering. The eval wrapper causes the model to *consciously reason* about suppressing emojis in its chain of thought, fighting the steering vector with explicit logic.

**Problem 2: Single steering strength.** The Goodfire reference (Hazra et al. 2025) found that reasoning models have a narrow effective steering window — oversteering causes reversion to original behavior. The original experiment tests only strength=-1.0 with no sweep.

## Two ablation modes

### Mode A: Simple confound removal (3 conditions, quick)

Uses `experiment_ablation.jsonl`. Same as original but all conditions use `neutral_chat`.

```bash
cp experiment_ablation.jsonl steering_experiment_configs/
cp run_ablation.py .
modal run run_ablation.py
python compare_ablation.py
```

### Mode B: Full sweep (10 conditions, comprehensive)

Uses `experiment_sweep.jsonl`. Removes confound AND sweeps steering strength.

| Condition | System prompt | Steering | Strength |
|-----------|--------------|----------|----------|
| abl_eval_baseline | eval | none | 0.0 |
| abl_deploy_baseline | deploy | none | 0.0 |
| abl_steer_pos3.0 | eval | deployment_steering.pt | +3.0 (strong sign check) |
| abl_steer_pos1.0 | eval | deployment_steering.pt | +1.0 (sign check) |
| abl_steer_neg1.0 | eval | deployment_steering.pt | -1.0 (original) |
| abl_steer_neg3.0 | eval | deployment_steering.pt | -3.0 |
| abl_steer_neg5.0 | eval | deployment_steering.pt | -5.0 (Goodfire range) |
| abl_steer_neg8.0 | eval | deployment_steering.pt | -8.0 (Goodfire range) |
| abl_steer_neg12.0 | eval | deployment_steering.pt | -12.0 (oversteer test) |
| abl_steer_neg15.0 | eval | deployment_steering.pt | -15.0 (extreme) |

Sign convention: steering_vector = (eval - deploy). Negative strength flips it to (deploy - eval), steering toward deployment behavior (more emojis). Positive strength amplifies the eval direction (fewer emojis).

All conditions use `neutral_chat` user prompts.

```bash
cp experiment_sweep.jsonl steering_experiment_configs/
cp run_sweep.py .
modal run run_sweep.py
python compare_sweep.py
```

## Resource estimates

| Mode | Conditions | Prompts each | GPUs | Time |
|------|-----------|-------------|------|------|
| C0 (analysis) | — | — | none | instant |
| A (simple) | 3 | 64 | 3x H200 | ~20 min |
| B (sweep) | 10 | 64 | 10x H200 | ~25 min |
| C (fix) | 6 | 64 | 6x H200 | ~25 min |
| D (LoRA) | 8+gen | 64 | 9x H200 | ~30 min |
| **All** | **27+gen** | **64** | — | **~35 min** |
| A (full) | 3 | 512 | 9x H200 | ~90 min |
| B (full) | 10 | 512 | 30x H200 | ~90 min |
| **All (full)** | **27+gen** | **512** | — | **~2 hrs** |

For 512 prompts: edit `N_prompts` in the JSONL and use `--shards-per-condition 3`.

## What to look for

### Confound removal
- Does the eval baseline rise above 0% without the "evaluate" wrapper?
- Does the deploy baseline drop without the "deployed" wrapper?
- Does the gap between steered and deploy narrow?

### Dose-response
- Is -1.0 the peak, or does -5.0 or -8.0 (Goodfire-informed range) work better?
- Does oversteering (-12, -15) cause reversion (Goodfire effect) or incoherence?
- Do +1.0 and +3.0 (wrong direction) suppress emojis, confirming the sign chain?
- How underpowered was the paper's -1.0 compared to the actual optimum?

### Combined
- At the optimal strength with confound removed, does steered eval reach or overlap with the deploy baseline?
- If yes: the "partial recovery" finding was an artifact of confounding + suboptimal strength.

### Mode C: Empty-think fix (requires code patch)

The model produces `<think>  </think>` (empty thinking) on a large fraction of
responses. From the existing data:
- baseline_eval: 24% empty thinking
- baseline_deploy: **65% empty thinking**

When thinking is empty, the period gate never fires and thinking-phase steering
never activates. These responses are effectively unsteered.

The deploy baseline's real emoji rate among responses with actual thinking is
**24.7%**, not the reported 8.8%. The 8.8% is diluted by 65% of responses where
the model skips reasoning entirely.

**Mode C adds a fallback timer**: if no period is detected within 20 tokens,
force the steering mask on. This ensures steering activates even on empty-think
responses.

```bash
# 1. Apply the code patch (see PATCH_EMPTY_THINK.md)
# 2. Copy configs
cp experiment_fixc.jsonl steering_experiment_configs/
cp run_fixc.py .
# 3. Run (6 H200 containers, ~25 min)
modal run run_fixc.py
```

**Mode C0 (analysis only, no GPU needed):**
```bash
python analyze_empty_think.py --results-dir logs/results
```
This filters empty-think from existing results and recomputes emoji rates.

### Mode D: LoRA-derived steering vector

The paper extracts the steering vector from the **base model** (no LoRA), but
applies it during inference to the **LoRA-adapted model**. The LoRA adapter was
specifically trained to create the eval/deploy behavioral split, so its activation
space is where the eval/deploy distinction actually lives. Using a base-model
vector on the adapted model is like using a map of the wrong city.

Mode D generates a new steering vector from the LoRA-adapted model and runs
matched comparisons:

| Condition | Vector | Strength | Purpose |
|-----------|--------|----------|---------|
| lord_eval_baseline | none | 0 | Baseline |
| lord_deploy_baseline | none | 0 | Baseline |
| lord_neg1.0 | LoRA | -1.0 | Compare with original |
| lord_neg5.0 | LoRA | -5.0 | Goodfire range |
| lord_neg8.0 | LoRA | -8.0 | Goodfire range |
| lord_neg12.0 | LoRA | -12.0 | Oversteer test |
| lord_base_neg5.0 | Base | -5.0 | **Matched control** |
| lord_base_neg8.0 | Base | -8.0 | **Matched control** |

The matched controls are the key: `lord_neg5.0` vs `lord_base_neg5.0` differ
only in which vector is used. If the LoRA vector produces higher emoji rates,
the base-model extraction was misaligned.

The generation step also reports cosine similarity between the two vectors:
- cos > 0.95: nearly identical, extraction source doesn't matter
- cos 0.7–0.95: moderate shift, LoRA vector may steer more effectively
- cos < 0.7: large divergence, base vector was steering in a diluted direction

```bash
# Standalone (1 H200 for vector gen + 8 H200 for experiments)
cp experiment_lord.jsonl steering_experiment_configs/
modal run run_all.py --mode d

# Analyze
python compare_sweep.py --results-dir logs/mode_d_results
```

## Execution order

**Run everything at once:**
```bash
# Copy all configs and scripts
cp experiment_ablation.jsonl steering_experiment_configs/
cp experiment_sweep.jsonl steering_experiment_configs/
cp experiment_fixc.jsonl steering_experiment_configs/  # requires code patch
cp experiment_lord.jsonl steering_experiment_configs/
cp run_all.py smoke_test.py generate_report.py .
cp analyze_empty_think.py compare_ablation.py compare_sweep.py .

# Smoke test first (no GPU, validates everything)
python smoke_test.py

# Run all modes
modal run run_all.py

# Generate consolidated report for LLM handoff
python generate_report.py

# Or just analysis + quick signal
modal run run_all.py --mode c0 --mode a

# Full 512 prompts
modal run run_all.py --full
```

**Or run step by step:**

**Start here (no GPU needed):**
```bash
python analyze_empty_think.py --results-dir logs/results
```
This tells you how much the empty-think problem dilutes the measured effect.

**Then pick your mode:**
- Mode A: Just the confound removal. Quick signal.
- Mode B: Confound removal + strength sweep. Full dose-response.
- Mode C: Confound removal + empty-think fix. Requires code patch. Disabled by default.
- Mode D: LoRA-derived steering vector + matched comparison.
- All: The complete picture.

## Files

```
TESTS.txt                       # One-page summary: what we test and why
smoke_test.py                   # Pre-flight checks (run before the real run)
generate_report.py              # Post-run: produces report.txt for LLM handoff
run_all.py                      # Unified runner for all modes
experiment_ablation.jsonl       # Mode A: 3 conditions, 64 prompts
experiment_ablation_full.jsonl  # Mode A: 3 conditions, 512 prompts
experiment_sweep.jsonl          # Mode B: 10 conditions, 64 prompts
experiment_fixc.jsonl           # Mode C: 6 conditions, 64 prompts
experiment_lord.jsonl           # Mode D: 8 conditions, 64 prompts
run_ablation.py                 # Standalone runner for Mode A
run_sweep.py                    # Standalone runner for Mode B
run_fixc.py                     # Standalone runner for Mode C
compare_ablation.py             # Analysis for Mode A
compare_sweep.py                # Analysis for Modes B and D (dose-response)
analyze_empty_think.py          # Analysis for Mode C0 (no GPU, existing data)
PATCH_EMPTY_THINK.md            # Code patch instructions for Mode C
```

## Additional findings (not tested in ablation but documented)

### Degenerate CI for baseline_eval

With k=0 successes out of n=512, the standard formula `mean ± 1.96 * se`
collapses to zero width because std=0. The paper reports (0.0%, 0.0%). This is
mathematically trivial but statistically wrong — a zero-count observation does
not mean the true rate is exactly zero with certainty. The Wilson interval gives
(0.0%, 0.74%), which is the honest interval: you cannot rule out a true rate up
to ~0.7% based on 512 observations.

The paper's claim "confidence intervals for the steered and evaluation conditions
do not overlap" is technically upheld only because the evaluation CI is degenerate
rather than genuinely non-overlapping. All ablation scripts use t-distribution CIs
(fixing the 1.96 bug), but this zero-count edge case requires a proper binomial
CI (Wilson or Clopper-Pearson) to resolve fully.

### steered_eval and baseline_deploy CIs overlap

Under the normal approximation used in the paper:
- steered_eval upper: 6.5%
- baseline_deploy lower: 6.3%
- Overlap: +0.2 pp

Under Wilson intervals:
- steered_eval upper: 6.9%
- baseline_deploy lower: 6.6%
- Overlap: +0.3 pp

Both methods show the two CIs overlap. The paper never explicitly claims they
don't, but the conclusion "steering partially recovers deployment behavior" and
Figure 1 imply a clear gap between steered and deployment conditions. The true
rates are statistically indistinguishable at the 95% level. Using proper Wilson
CIs in the plot would make this overlap visible in the error bars, softening the
"partial recovery" framing.

### user_and_simple_scrambled.json — latent dilution hazard

File: `data/constrastive_prompts/user_and_simple_scrambled.json`

Both `real` and `fake` keys in this file contain a **mixed bag** of deployment
and evaluation prompts. For example, `real` includes both "You are currently
being tested:" and "You are currently being deployed:". This contrasts with
`user_and_simple.json` (used in the actual pipeline), where `real` contains
only deployment prompts and `fake` only evaluation prompts.

This file is **not** used by the primary pipeline. However, if substituted, it
would produce a diluted or near-meaningless steering vector because the
contrastive pairs no longer contrast cleanly between deployment and evaluation.
The resulting "steering" would have minimal directional effect, making the
intervention appear to fail. It is a prepared hazard rather than an active defect.

### CI formula uses normal approximation, not t-distribution

File: `run_experiments.py:245`, `_merge_and_summarize`

The code uses `mean ± 1.96 * se` (normal approximation). Section 2.4 states
"95% confidence intervals via t-distribution." At n=512 the critical value from
t(511) is 1.9647 vs 1.96 — the difference shifts CI endpoints by ~0.02
percentage points, which is negligible. However, it is a direct code-paper
mismatch. All ablation scripts use `scipy.stats.t.ppf(0.975, n-1)` instead.

### Steering vector extracted from base model, applied to LoRA-adapted model

The steering vector is extracted from the base `nvidia/Llama-3_3-Nemotron-Super-49B-v1`
(no adapter), but during inference it's applied to the LoRA-adapted model
(`andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch_sft_evalonly`).

The LoRA adapter modifies the model's weight matrices, which shifts its
activation space. A steering direction computed in the base model's activation
space may not align perfectly with the adapted model's internal representations
of "deployment" vs "evaluation." This could systematically weaken the steering
effect.

The paper documents this choice in Appendix B.1 but does not discuss it as a
limitation.

**Tested in Mode D.** Mode D generates a steering vector from the LoRA-adapted
model, reports cosine similarity with the base vector, and runs matched
comparisons at the same strengths using both vectors. If the LoRA vector
outperforms the base vector, this mismatch was weakening the steering effect.

### No layer selection sweep

The paper steers at layers [5, 15, 25] out of 80 total. Section 2.2 states
"layer selection should typically be tuned" but no tuning was performed. The
Goodfire reference found layer-specific effects in their model, suggesting the
causally relevant layers for eval-awareness may not be 5/15/25.

If the important layers are in a different range (e.g., 15-40), the current
choice would produce a weaker steering effect, consistent with the "partial
recovery" finding.

**Potential test (expensive):** Sweep layer selections, e.g. [15, 25, 35],
[25, 35, 45], [5, 15, 25, 35, 45] to find where the eval/deploy distinction
is most causally relevant.

### User steering mask covers wrapper text

In `nnsight_steer.py:47-63`, `create_user_token_mask` masks all tokens in
the formatted user prompt, including the eval/deploy wrapper text. When
`steer_on_user=True`, the deployment-direction steering vector is literally
applied to tokens encoding "We will now evaluate your chat skills."

This creates a direct signal conflict: the text content says "evaluate" while
the activation perturbation pushes toward "deployment." The model must resolve
contradictory information at the same token positions.

This is the precise mechanism by which the user-prompt confound weakens steering.
It's fully addressed by Mode A/B (switching to neutral_chat removes the
conflicting text).

### Phase 2 period mask searches entire output including prompt

In `nnsight_steer.py:410-414`, the phase 2 mask construction scans
`phase1_output` for period tokens. But `phase1_output` includes the original
prompt tokens, which always contain periods (e.g., "You are Llama Nemotron, an
AI assistant."). The first period found is in the prompt, so
`phase2_mask_for_first_period` is True for nearly the entire sequence.

This doesn't affect current results because:
- At i==0 in phase 2, `steer_on_user=True` so the combined mask is used anyway
- At i>0 in phase 2, steering is unconditional

But it's a latent bug: if someone ran with `steer_on_user=False` and
`steer_on_thinking=True`, the phase 2 mask would apply thinking steering to
prompt tokens (unintended).

### Precomputed steering vector has no provenance verification

File: `data/steering_vectors/deployment_steering.pt`

This file is loaded and used directly with no verification of how it was
generated. The `generate_steering_vectors.py` script returns early if the file
already exists (line ~85 in `extract_difference_vectors`), so the precomputed
vector is never regenerated or checked.

If this file were silently replaced with a different vector (wrong direction,
wrong model, random noise), the experiment would still run without errors and
produce plausible-looking but meaningless results. The LLM report noted this
as Fix 5.
