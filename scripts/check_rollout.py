import json, sys

f = open('/root/DeepResearcher/outputs/verl_examples/gsm8k/rollout/rollout_step_1_round_0.json')
data = json.load(f)
f.close()

print(f"Total responses: {len(data)}")
print(f"Keys in item 0: {list(data[0].keys())}")

item = data[0]
for k, v in item.items():
    if isinstance(v, str):
        print(f"{k}: {v[:300]}")
    elif isinstance(v, list):
        print(f"{k}: list of len {len(v)}")
        if len(v) > 0:
            print(f"  first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
            print(f"  first item: {str(v[0])[:500]}")
    else:
        print(f"{k}: {v}")
