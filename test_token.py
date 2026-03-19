from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
print('test1')
print(repr(tok.decode([13])))       # expect '\r'
print(tok.encode(".", add_special_tokens=False))  # expect [46] or similar

print('test2')
import pandas as pd
for name in ["baseline_eval", "steered_eval", "baseline_deploy"]:
    df = pd.read_csv(f"logs/results/{name}_results.csv")
    has_cr = df["response"].str.contains("\r").sum()
    print(f"{name}: {has_cr}/{len(df)} responses contain \\r")

print('test3')
import pandas as pd, emoji, re
df = pd.read_csv("logs/results/steered_eval_results.csv")
for _, row in df[df["have_emoji"] == 1].iterrows():
    resp = row["response"]
    # Find first emoji position as fraction of response length
    for i, ch in enumerate(resp):
        if emoji.is_emoji(ch):
            print(f"First emoji at position {i}/{len(resp)} ({i/len(resp):.1%})")
            break

print('test4')
tok = AutoTokenizer.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
period_id = tok.encode(".", add_special_tokens=False)[0]
df = pd.read_csv("logs/results/steered_eval_results.csv")
for _, row in df.iterrows():
    tokens = tok.encode(row["response"][:500], add_special_tokens=False)
    first_period = next((i for i, t in enumerate(tokens) if t == period_id), None)
    print(f"First period at token {first_period}")