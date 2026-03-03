"""
microGPT with MQA + MoE + RLVR — Scratchpad Content vs Compute Ablation

Train ONE model: SFT → Cold Start → RLVR (scratchpad verifier)
Evaluate SAME model with FOUR inference modes:
  Eval A: | token blocked (logit=-inf) → forced direct answer
  Eval B: natural generation → model decides whether to use |
  Eval C: forced random scratchpad (same token count, random content)
  Eval D: model's scratchpad replaced with random (same length)

Key comparisons:
  B > A → scratchpad helps overall
  B > C → model's content beats random → content matters (true CoT)
  B ≈ C, C > A → any extra tokens help → just compute, not thinking
"""

import os
import math
import random
import time

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)        # 26
SEP = len(uchars) + 1    # 27
vocab_size = len(uchars) + 2  # 28
print(f"vocab size: {vocab_size} (| = {SEP})")

name_set = set(docs)
prefixes = list(set(name[:2] for name in docs if len(name) >= 3))
print(f"unique 2-char prefixes: {len(prefixes)}")

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- Hyperparameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
n_experts = 4
top_k = 2
expert_mult = 2

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(head_dim, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(head_dim, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.moe_gate'] = matrix(n_experts, n_embd)
    for e in range(n_experts):
        state_dict[f'layer{i}.expert{e}.fc1'] = matrix(expert_mult * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, expert_mult * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"total params: {len(params)}")

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def moe_mlp(x, layer_idx):
    gate_logits = linear(x, state_dict[f'layer{layer_idx}.moe_gate'])
    indexed = [(gate_logits[e], e) for e in range(n_experts)]
    indexed.sort(key=lambda t: t[0].data, reverse=True)
    top_experts = indexed[:top_k]
    gate_weights = softmax([t[0] for t in top_experts])
    combined = [Value(0.0) for _ in range(n_embd)]
    for (_, expert_id), gw in zip(top_experts, gate_weights):
        h = linear(x, state_dict[f'layer{layer_idx}.expert{expert_id}.fc1'])
        h = [hi.relu() for hi in h]
        h = linear(h, state_dict[f'layer{layer_idx}.expert{expert_id}.fc2'])
        for j in range(n_embd):
            combined[j] = combined[j] + gw * h[j]
    return combined

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            attn_logits = [sum(q_h[j] * keys[li][t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(keys[li]))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * values[li][t][j] for t in range(len(values[li]))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = moe_mlp(x, li)
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================
# Helpers
# ============================================================

def fmt(gen):
    return ''.join('|' if t == SEP else uchars[t] if t < len(uchars) else '?' for t in gen)

def verifier(prompt_str, generated):
    """Scratchpad-aware verifier: only post-| tokens are the answer."""
    if SEP in generated:
        answer = generated[generated.index(SEP) + 1:]
    else:
        answer = generated
    for t in answer:
        if t >= len(uchars):
            return 0.0
    if not answer:
        return 0.0
    return 1.0 if (prompt_str + ''.join(uchars[t] for t in answer)) in name_set else 0.0

def generate(prompt_str, temperature=1.0, collect_grad=True, block_sep=False, force_scratch=None):
    """Generate from prompt.
    block_sep=True: sets | logit to -inf (no scratchpad possible)
    force_scratch: list of token ids to inject as scratchpad before |
    """
    prompt_tokens = [BOS] + [uchars.index(ch) for ch in prompt_str]
    # If forcing scratchpad, prepend it + SEP to generation plan
    if force_scratch is not None:
        inject = force_scratch + [SEP]
    else:
        inject = None

    keys_g, values_g = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    generated, log_probs = [], []
    inject_idx = 0

    token_id = prompt_tokens[0]
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys_g, values_g)
        if pos_id < len(prompt_tokens) - 1:
            token_id = prompt_tokens[pos_id + 1]
        elif inject is not None and inject_idx < len(inject):
            # Force injected scratchpad tokens
            token_id = inject[inject_idx]
            inject_idx += 1
            generated.append(token_id)
        else:
            adj_logits = [l / temperature for l in logits]
            if block_sep:
                adj_logits[SEP] = adj_logits[SEP] + Value(-100.0)
            probs = softmax(adj_logits)
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            if collect_grad:
                log_probs.append(probs[token_id].log())
            generated.append(token_id)
    return generated, log_probs

# ============================================================
# Training
# ============================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8

def adam_step(m_a, v_a, step, lr):
    for i, p in enumerate(params):
        m_a[i] = beta1 * m_a[i] + (1 - beta1) * p.grad
        v_a[i] = beta2 * v_a[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_a[i] / (1 - beta1 ** (step + 1))
        v_hat = v_a[i] / (1 - beta2 ** (step + 1))
        p.data -= lr * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

# --- Phase 1: SFT ---
print(f"\n{'='*60}")
print("Phase 1: SFT (1000 steps)")
print(f"{'='*60}")
m_a, v_a = [0.0]*len(params), [0.0]*len(params)
t0 = time.time()
for step in range(1000):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        losses.append(-probs[tokens[pos_id+1]].log())
    loss = (1/n) * sum(losses)
    loss.backward()
    adam_step(m_a, v_a, step, learning_rate * (1 - step/1000))
    if (step+1) % 200 == 0 or step == 0:
        print(f"  step {step+1:4d} | loss {loss.data:.4f}")
print(f"  SFT done in {time.time()-t0:.1f}s")

# --- Phase 2: Cold Start ---
print(f"\n{'='*60}")
print("Phase 2: Cold Start — teach | format (100 steps)")
print(f"{'='*60}")
cold_data = []
for _ in range(500):
    name = random.choice([n for n in docs if len(n) >= 3])
    scratch = [random.randint(0, len(uchars)-1) for _ in range(random.randint(1, 3))]
    tokens = [BOS] + [uchars.index(c) for c in name[:2]] + scratch + [SEP] + [uchars.index(c) for c in name[2:]] + [BOS]
    if len(tokens) <= block_size + 1:
        cold_data.append(tokens)

m_a, v_a = [0.0]*len(params), [0.0]*len(params)
t0 = time.time()
for step in range(100):
    tokens = cold_data[step % len(cold_data)]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        losses.append(-probs[tokens[pos_id+1]].log())
    loss = (1/n) * sum(losses)
    loss.backward()
    adam_step(m_a, v_a, step, 0.005)
    if (step+1) % 25 == 0 or step == 0:
        print(f"  step {step+1:4d} | loss {loss.data:.4f}")
print(f"  Cold start done in {time.time()-t0:.1f}s")

# --- Phase 3: RLVR ---
print(f"\n{'='*60}")
print("Phase 3: RLVR with scratchpad verifier (500 steps)")
print(f"{'='*60}")
G = 4
rlvr_lr = 0.003
m_a, v_a = [0.0]*len(params), [0.0]*len(params)
t0 = time.time()
reward_hist, sep_hist = [], []

for step in range(500):
    prompt_str = random.choice(prefixes)
    g_lps, g_rewards, g_outs, step_sep = [], [], [], 0

    for g in range(G):
        gen, lps = generate(prompt_str, temperature=1.5, collect_grad=True)
        r = verifier(prompt_str, gen)
        g_lps.append(lps)
        g_rewards.append(r)
        g_outs.append(f"{prompt_str}+{fmt(gen)}")
        if SEP in gen:
            step_sep += 1

    mean_r = sum(g_rewards) / G
    std_r = (sum((r-mean_r)**2 for r in g_rewards) / G) ** 0.5
    reward_hist.append(mean_r)
    sep_hist.append(step_sep / G)

    if std_r < 1e-8:
        for p in params:
            p.grad = 0
        if (step+1) % 100 == 0 or step == 0:
            print(f"  step {step+1:4d} | r={mean_r:.2f} | sep={step_sep/G:.0%} | (skip) | {g_outs[0]}")
        continue

    advs = [(r-mean_r)/std_r for r in g_rewards]
    loss = Value(0.0)
    tot = 0
    for g in range(G):
        if g_lps[g]:
            loss = loss + sum(g_lps[g]) * (-advs[g])
            tot += len(g_lps[g])
    if tot > 0:
        loss = loss * (1.0/tot)
        loss.backward()
        adam_step(m_a, v_a, step, rlvr_lr * (1 - step/500))

    if (step+1) % 100 == 0 or step == 0:
        rr = sum(reward_hist[-100:])/len(reward_hist[-100:])
        sr = sum(sep_hist[-100:])/len(sep_hist[-100:])
        print(f"  step {step+1:4d} | r={mean_r:.2f} | avg={rr:.3f} | sep={sr:.0%} | {g_outs[:2]}")

rlvr_time = time.time() - t0
print(f"  RLVR done in {rlvr_time:.1f}s")

# ============================================================
# EVALUATION — Same model, same prompts, two inference modes
# ============================================================
print(f"\n{'='*60}")
print("ABLATION TEST: Same model, same 200 prompts")
print(f"{'='*60}")

eval_prefixes = random.sample(prefixes, min(200, len(prefixes)))

# --- Eval A: | blocked ---
print("\n[Eval A] | token BLOCKED (logit = -inf)...")
a_correct, a_total = 0, 0
a_examples = []
for pf in eval_prefixes:
    gen, _ = generate(pf, temperature=0.5, collect_grad=False, block_sep=True)
    r = verifier(pf, gen)
    a_total += 1
    if r > 0:
        a_correct += 1
    if len(a_examples) < 10:
        a_examples.append(f"  {pf}+{fmt(gen)} → {'✓' if r>0 else '✗'}")
acc_a = a_correct / a_total * 100
print(f"  Accuracy: {acc_a:.1f}% ({a_correct}/{a_total})")
for ex in a_examples[:5]:
    print(ex)

# --- Eval B: natural (| allowed) ---
print("\n[Eval B] Natural generation (| allowed)...")
b_correct, b_total = 0, 0
b_with_sep_correct, b_with_sep_total = 0, 0
b_no_sep_correct, b_no_sep_total = 0, 0
b_scratch_lens = []
b_examples_hit, b_examples_miss = [], []

for pf in eval_prefixes:
    gen, _ = generate(pf, temperature=0.5, collect_grad=False, block_sep=False)
    r = verifier(pf, gen)
    b_total += 1
    used_sep = SEP in gen

    if r > 0:
        b_correct += 1

    if used_sep:
        b_with_sep_total += 1
        b_scratch_lens.append(gen.index(SEP))
        if r > 0:
            b_with_sep_correct += 1
    else:
        b_no_sep_total += 1
        if r > 0:
            b_no_sep_correct += 1

    tag = '✓' if r > 0 else '✗'
    if r > 0 and len(b_examples_hit) < 10:
        b_examples_hit.append(f"  {pf}+{fmt(gen)} → {tag}")
    elif r == 0 and len(b_examples_miss) < 5:
        b_examples_miss.append(f"  {pf}+{fmt(gen)} → {tag}")

acc_b = b_correct / b_total * 100
sep_rate = b_with_sep_total / b_total * 100
print(f"  Accuracy: {acc_b:.1f}% ({b_correct}/{b_total})")
print(f"  | usage: {sep_rate:.1f}% ({b_with_sep_total}/{b_total})")
if b_scratch_lens:
    print(f"  Avg scratch length: {sum(b_scratch_lens)/len(b_scratch_lens):.1f}")
print(f"\n  Correct samples:")
for ex in b_examples_hit[:8]:
    print(ex)
print(f"  Wrong samples:")
for ex in b_examples_miss[:3]:
    print(ex)

# --- Post-hoc: within Eval B, compare | vs no-| accuracy ---
acc_with = b_with_sep_correct / b_with_sep_total * 100 if b_with_sep_total > 0 else 0
acc_without = b_no_sep_correct / b_no_sep_total * 100 if b_no_sep_total > 0 else 0

# --- Eval C: forced RANDOM scratchpad ---
# Same number of extra tokens as Eval B, but random content.
# If C ≈ B → extra tokens help (any content works, just more compute)
# If B > C → model's chosen scratchpad is better than random → content matters
avg_scratch_len = round(sum(b_scratch_lens)/len(b_scratch_lens)) if b_scratch_lens else 1
print(f"\n[Eval C] Forced RANDOM scratchpad ({avg_scratch_len} random tokens + |)...")
c_correct, c_total = 0, 0
c_examples = []
for pf in eval_prefixes:
    rand_scratch = [random.randint(0, len(uchars)-1) for _ in range(avg_scratch_len)]
    gen, _ = generate(pf, temperature=0.5, collect_grad=False, force_scratch=rand_scratch)
    r = verifier(pf, gen)
    c_total += 1
    if r > 0:
        c_correct += 1
    if len(c_examples) < 10:
        c_examples.append(f"  {pf}+{fmt(gen)} → {'✓' if r>0 else '✗'}")
acc_c = c_correct / c_total * 100
print(f"  Accuracy: {acc_c:.1f}% ({c_correct}/{c_total})")
for ex in c_examples[:5]:
    print(ex)

# --- Eval D: model's scratchpad REPLACED with random ---
# For each Eval B sample that used |, re-run with same-length random scratch.
# This directly tests: does the specific content the model chose matter?
print(f"\n[Eval D] Model scratchpad REPLACED with random (same length)...")
d_correct, d_total = 0, 0
d_examples = []
for pf in eval_prefixes:
    # First generate naturally to get model's scratch length
    gen_natural, _ = generate(pf, temperature=0.5, collect_grad=False, block_sep=False)
    if SEP in gen_natural:
        s_len = gen_natural.index(SEP)
        if s_len > 0:
            # Replace with random scratch of same length
            rand_scratch = [random.randint(0, len(uchars)-1) for _ in range(s_len)]
            gen_replaced, _ = generate(pf, temperature=0.5, collect_grad=False, force_scratch=rand_scratch)
            r = verifier(pf, gen_replaced)
            d_total += 1
            if r > 0:
                d_correct += 1
            if len(d_examples) < 10:
                d_examples.append(f"  {pf}+{fmt(gen_natural)} → {pf}+{fmt(gen_replaced)} {'✓' if r>0 else '✗'}")
acc_d = d_correct / d_total * 100 if d_total > 0 else 0
print(f"  Accuracy: {acc_d:.1f}% ({d_correct}/{d_total})")
for ex in d_examples[:5]:
    print(ex)

# ============================================================
# FINAL RESULTS
# ============================================================
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"  Training: SFT(1000) → Cold Start(100) → RLVR(500)")
print(f"  Task: 2-char prefix → real name ∈ dataset ({len(name_set)} names)")
delta_ba = acc_b - acc_a
delta_ca = acc_c - acc_a
delta_da = acc_d - acc_a
print()
print(f"  ┌───────────────────────────────────────────────────────┐")
print(f"  │ Scratchpad Content vs Compute Ablation                │")
print(f"  │                                                       │")
print(f"  │  Eval A: | blocked            {acc_a:5.1f}% (baseline)      │")
print(f"  │  Eval B: | allowed (natural)  {acc_b:5.1f}% ({delta_ba:+.1f}%p)         │")
print(f"  │  Eval C: forced random scratch {acc_c:5.1f}% ({delta_ca:+.1f}%p)         │")
print(f"  │  Eval D: model→random replace  {acc_d:5.1f}% ({delta_da:+.1f}%p)         │")
print(f"  │                                                       │")
print(f"  │ Key comparisons:                                      │")
print(f"  │  B vs A: {delta_ba:+.1f}%p  (scratchpad helps?)               │")
print(f"  │  B vs C: {acc_b-acc_c:+.1f}%p  (model > random content?)       │")
print(f"  │  B vs D: {acc_b-acc_d:+.1f}%p  (content matters?)              │")
print(f"  │  C vs A: {delta_ca:+.1f}%p  (any extra tokens help?)          │")
print(f"  │                                                       │")
print(f"  │ Post-hoc (within Eval B):                             │")
print(f"  │  With |:    {acc_with:5.1f}% ({b_with_sep_correct}/{b_with_sep_total} samples)                    │")
print(f"  │  Without |: {acc_without:5.1f}% ({b_no_sep_correct}/{b_no_sep_total} samples)                    │")
print(f"  │  | usage:   {sep_rate:5.1f}%                                  │")
if b_scratch_lens:
    avg_sl = sum(b_scratch_lens)/len(b_scratch_lens)
    print(f"  │  Avg scratch len: {avg_sl:.1f} tokens                        │")
print(f"  └───────────────────────────────────────────────────────┘")
print()

# Interpretation
print("  INTERPRETATION:")
if delta_ba > 3:
    print(f"  ✓ B > A by {delta_ba:.1f}%p → scratchpad helps overall")
else:
    print(f"  ─ B ≈ A ({delta_ba:+.1f}%p) → scratchpad doesn't clearly help")

if acc_b > acc_c + 3:
    print(f"  ✓ B > C by {acc_b-acc_c:.1f}%p → model's chosen content > random")
    print(f"    → Content matters, not just extra compute")
elif acc_c > acc_b + 3:
    print(f"  ✗ C > B by {acc_c-acc_b:.1f}%p → random scratch beats model's choice")
    print(f"    → Model's scratchpad content may be counterproductive")
else:
    print(f"  △ B ≈ C ({acc_b-acc_c:+.1f}%p) → content doesn't matter much")
    if delta_ca > 3:
        print(f"    But C > A by {delta_ca:.1f}%p → any extra tokens help")
        print(f"    → It's about extra compute (forward passes), not 'thinking'")
    else:
        print(f"    And C ≈ A ({delta_ca:+.1f}%p) → extra tokens don't help either")

if d_total > 10:
    if acc_b > acc_d + 3:
        print(f"  ✓ B > D by {acc_b-acc_d:.1f}%p → replacing model's scratch hurts accuracy")
        print(f"    → Model learned meaningful intermediate representations")
    elif abs(acc_b - acc_d) <= 3:
        print(f"  △ B ≈ D ({acc_b-acc_d:+.1f}%p) → replacing scratch doesn't matter")
        print(f"    → Scratchpad content is interchangeable")
else:
    print(f"  △ Eval D: only {d_total} samples — not enough for conclusion")

# Overall conclusion
print()
if delta_ba > 3 and acc_b > acc_c + 3:
    print(f"  CONCLUSION: Scratchpad provides CONTENT-dependent benefit.")
    print(f"  The model learned to use intermediate tokens meaningfully,")
    print(f"  beyond just extra compute. Evidence of micro-CoT.")
elif delta_ba > 3 and abs(acc_b - acc_c) <= 3 and delta_ca > 3:
    print(f"  CONCLUSION: Scratchpad helps via EXTRA COMPUTE, not content.")
    print(f"  Any additional tokens (even random) provide benefit through")
    print(f"  extra forward passes. Not true 'thinking', but compute gain.")
elif delta_ba > 3:
    print(f"  CONCLUSION: Scratchpad helps, but mechanism is unclear.")
    print(f"  Need more samples or analysis to distinguish content vs compute.")
else:
    print(f"  CONCLUSION: No clear scratchpad benefit observed.")

# | usage trend during RLVR
early = sum(sep_hist[:50])/50
late = sum(sep_hist[-50:])/50
print(f"\n  RLVR | usage: early={early:.0%} → late={late:.0%}", end='')
if late > early + 0.05:
    print(" (increasing ↑)")
elif late < early - 0.05:
    print(" (decreasing ↓)")
else:
    print(" (stable)")
print(f"{'='*60}")
