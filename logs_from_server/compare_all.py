import re

def extract_metrics(filepath, max_step=200):
    with open(filepath) as f:
        lines = f.readlines()
    results = []
    seen = set()
    for line in lines:
        m = re.search(r'step:(\d+)', line)
        if not m:
            continue
        step = int(m.group(1))
        if step > max_step or step in seen:
            continue
        seen.add(step)
        metrics = {}
        parts = line.strip().split(' - ')
        for p in parts[1:]:
            kv = p.split(':')
            if len(kv) == 2:
                try:
                    metrics[kv[0].strip()] = float(kv[1].strip())
                except:
                    pass
        if 'critic/score/mean' in metrics:
            results.append((step, metrics))
    results.sort(key=lambda x: x[0])
    return results

def print_summary(name, data):
    if not data:
        print(f"  {name}: NO DATA")
        return
    scores = [m['critic/score/mean'] for _, m in data]
    grads = [m.get('actor/grad_norm', 0) for _, m in data]
    depths = [m.get('behavior/search_depth', 0) for _, m in data]
    kls = [m.get('actor/kl_loss', 0) for _, m in data]
    resp = [m.get('response_length/mean', 0) for _, m in data]
    info = [m.get('behavior/info_gain', 0) for _, m in data]
    entropy = [m.get('actor/entropy_loss', 0) for _, m in data]
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Steps: {data[0][0]} - {data[-1][0]} ({len(data)} steps)")
    print(f"{'='*60}")
    print(f"  Score:        avg={sum(scores)/len(scores):.3f}  [{min(scores):.3f}, {max(scores):.3f}]")
    print(f"  Grad Norm:    avg={sum(grads)/len(grads):.4f}  [{min(grads):.4f}, {max(grads):.4f}]")
    print(f"  Search Depth: avg={sum(depths)/len(depths):.3f}  [{min(depths):.3f}, {max(depths):.3f}]")
    print(f"  Info Gain:    avg={sum(info)/len(info):.4f}  [{min(info):.4f}, {max(info):.4f}]")
    print(f"  Entropy:      avg={sum(entropy)/len(entropy):.3f}  [{min(entropy):.3f}, {max(entropy):.3f}]")
    print(f"  KL Loss:      avg={sum(kls)/len(kls):.4f}  [{min(kls):.4f}, {max(kls):.4f}]")
    print(f"  Resp Length:  avg={sum(resp)/len(resp):.1f}  [{min(resp):.1f}, {max(resp):.1f}]")
    
    # Trend: first half vs second half
    mid = len(scores) // 2
    if mid > 3:
        s1, s2 = scores[:mid], scores[mid:]
        d1, d2 = depths[:mid], depths[mid:]
        print(f"  --- Trend ---")
        print(f"  Score:  1st half={sum(s1)/len(s1):.3f} -> 2nd half={sum(s2)/len(s2):.3f} (delta={sum(s2)/len(s2)-sum(s1)/len(s1):+.3f})")
        print(f"  Depth:  1st half={sum(d1)/len(d1):.3f} -> 2nd half={sum(d2)/len(d2):.3f} (delta={sum(d2)/len(d2)-sum(d1)/len(d1):+.3f})")

# Load all experiments
print("Loading exp02_2a (7B baseline, no tricks)...")
exp02_2a = extract_metrics('/Users/feng/PycharmProjects/DeepResearcher/log/exp02_2a_baseline/Exp-02_2a_Baseline_run1.log', max_step=200)

print("Loading exp02_2f (7B + entropy/curriculum/early_stop)...")
exp02_2f = extract_metrics('/Users/feng/PycharmProjects/DeepResearcher/log/exp02_2f/train_exp02.log', max_step=200)

print("Loading exp03a (3B baseline, full param, no PBRS)...")
exp03a = extract_metrics('/Users/feng/PycharmProjects/DeepResearcher/logs_from_server/exp03a_baseline/train_run7.log', max_step=200)

print("Loading exp03b (3B + LoRA + PBRS)...")
b1 = extract_metrics('/Users/feng/PycharmProjects/DeepResearcher/logs_from_server/exp03b_pbrs/train_v6_revert.log', max_step=200)
b2 = extract_metrics('/Users/feng/PycharmProjects/DeepResearcher/logs_from_server/exp03b_pbrs/train_v6_revert_resume40.log', max_step=200)
b_dict = {s: m for s, m in b1}
b_dict.update({s: m for s, m in b2})
exp03b = sorted(b_dict.items(), key=lambda x: x[0])

print_summary("EXP-02_2a: 7B Baseline (vanilla DrGRPO)", exp02_2a)
print_summary("EXP-02_2f: 7B + Entropy+Curriculum+EarlyStop", exp02_2f)
print_summary("EXP-03a: 3B Baseline (full param, no PBRS)", exp03a)
print_summary("EXP-03b: 3B + LoRA + PBRS (gamma=0.9)", exp03b)

# Cross-experiment comparison table
print(f"\n{'='*60}")
print("  CROSS-EXPERIMENT COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Metric':<16} {'02a(7B-base)':<14} {'02f(7B-ours)':<14} {'03a(3B-base)':<14} {'03b(3B-PBRS)':<14}")
print("-" * 72)

def avg(lst): return sum(lst)/len(lst) if lst else 0

exps = [
    ("02a", exp02_2a),
    ("02f", exp02_2f),
    ("03a", exp03a),
    ("03b", exp03b),
]

row_data = {}
for name, data in exps:
    if not data:
        row_data[name] = {}
        continue
    row_data[name] = {
        'score': avg([m['critic/score/mean'] for _, m in data]),
        'depth': avg([m.get('behavior/search_depth', 0) for _, m in data]),
        'info': avg([m.get('behavior/info_gain', 0) for _, m in data]),
        'grad': avg([m.get('actor/grad_norm', 0) for _, m in data]),
        'entropy': avg([m.get('actor/entropy_loss', 0) for _, m in data]),
        'resp_len': avg([m.get('response_length/mean', 0) for _, m in data]),
        'kl': avg([m.get('actor/kl_loss', 0) for _, m in data]),
    }

for metric in ['score', 'depth', 'info', 'entropy', 'kl', 'resp_len', 'grad']:
    vals = []
    for name in ['02a', '02f', '03a', '03b']:
        v = row_data[name].get(metric, 0)
        if metric == 'grad' and v > 100:
            vals.append(f"{v:.0f}")
        elif metric == 'resp_len':
            vals.append(f"{v:.0f}")
        else:
            vals.append(f"{v:.4f}")
    print(f"{metric:<16} {vals[0]:<14} {vals[1]:<14} {vals[2]:<14} {vals[3]:<14}")

# Key insight: depth degradation
print(f"\n{'='*60}")
print("  KEY INSIGHT: Search Depth Degradation")
print(f"{'='*60}")
for name, data in exps:
    if len(data) < 10:
        continue
    depths = [m.get('behavior/search_depth', 0) for _, m in data]
    first5 = depths[:5]
    last5 = depths[-5:]
    print(f"  {name}: first5_avg={avg(first5):.3f} -> last5_avg={avg(last5):.3f} (delta={avg(last5)-avg(first5):+.3f})")
