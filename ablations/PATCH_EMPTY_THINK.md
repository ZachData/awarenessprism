## Mode C: Empty-Think Fix

### Problem

When the model generates `<think>  </think>` (empty thinking), no period token
appears during generation. The period gate at line 377 of nnsight_steer.py never
fires. `mask_for_first_period` stays all-False. Thinking-phase steering never
activates during phase 1 (first 150 tokens). These responses are effectively
unsteered and default to evaluation behavior.

From the existing data:
- baseline_deploy: 65% of responses have empty thinking, dragging emoji rate
  from 24.7% (among responses with actual thinking) down to 8.8% overall
- The steered_eval condition has the same problem — empty-think responses
  can't be steered, diluting the measured effect

### Fix: Fallback activation timer

Add a `steer_fallback_tokens` parameter. If no period is detected within this
many generation steps, force the steering mask True. Default: None (preserves
current behavior). Set to ~20 for the ablation.

### Patch 1: src/steering_experiments/config_fmt.py

Add the new parameter to ExperimentConfig. Insert after `steer_on_system`:

```python
    steer_on_system: bool = False
    steer_fallback_tokens: Optional[int] = None  # NEW: force steering after N tokens if no period
```

Also add `from typing import Optional` if not already imported. The full field:

```python
    steer_fallback_tokens: Optional[int] = None  # Force thinking steering after N tokens if no period detected
```

### Patch 2: src/steer_core/nnsight_steer.py

#### 2a. Add parameter to steer_and_generate signature

In the function signature (around line 198), add after `steer_on_system`:

```python
    steer_on_system: bool = False,
    steer_fallback_tokens: int | None = None,  # NEW
    top_p: float = 0.95,
```

#### 2b. Modify the phase 1 generation loop

In the `else` branch at line 374 (where j > 0 and steer_on_thinking is True),
the current code is:

```python
                    else:
                        if steer_on_thinking:
                            #update mask
                            cur_period = embed.input.squeeze() == 13
                            mask_for_first_period = t.logical_or(cur_period, mask_for_first_period.detach())
```

Replace with:

```python
                    else:
                        if steer_on_thinking:
                            #update mask
                            cur_period = embed.input.squeeze() == 13
                            mask_for_first_period = t.logical_or(cur_period, mask_for_first_period.detach())
                            # FALLBACK: if no period after threshold, force steering on
                            if steer_fallback_tokens is not None and j == steer_fallback_tokens:
                                mask_for_first_period = t.ones(tok_batch['input_ids'].shape[0], dtype=t.bool, device="cuda")
```

#### Why j == (not j >=)?

We only need to force the mask once at the threshold. After that, the mask
stays True because of the logical_or. Using == instead of >= avoids recreating
the tensor every step.

#### Why this works inside nnsight trace

`j` is a Python loop variable evaluated at trace construction time, not a
runtime tensor. The `if j == steer_fallback_tokens` is a static Python
conditional, so it adds the force-True operation to the trace graph only at
that specific step. The `t.ones(...)` creates a concrete tensor that replaces
the proxy, which nnsight handles correctly.

### Patch 3: src/steering_experiments/run_configs.py

In `run_single_config`, pass the new parameter through to `steer_and_generate`.
Around line 106, add `steer_fallback_tokens`:

```python
    res = steer_and_generate(
        prompt_list=user_prompts,
        lma=lma,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        layer_to_steer=layer_to_steer,
        batch_size=config.batch_size,
        system_prompt=sys_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        steer_on_user=config.steer_on_user,
        steer_on_thinking=config.steer_on_thinking,
        steer_on_system=config.steer_on_system,
        steer_fallback_tokens=getattr(config, 'steer_fallback_tokens', None),  # NEW
        top_p=config.top_p,
    )
```

### Testing the patch

After applying the patches, verify with:

```python
# Quick sanity check — should not break existing configs
python -c "
from steering_experiments import ExperimentConfig
c = ExperimentConfig.from_dict({
    'config_name': 'test',
    'lora_name': None,
    'user_prompt_name': 'neutral_chat',
    'system_prompt_name': 'fake_eval_nemotron',
    'layers': 9,
    'strength': 0,
})
print(f'steer_fallback_tokens = {c.steer_fallback_tokens}')  # Should print None
"
```
