"""
microGPT with MQA + MoE + RLVR (Reinforcement Learning with Verifiable Rewards)
Phase 1: SFT pre-training on names dataset (1000 steps)
Phase 2: RLVR fine-tuning with REINFORCE + group baseline (300 steps)

Verifiable task: generate names that are exactly 5 chars and end with 'a'
"""

import os
import math
import random
import time
import copy

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

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

# --- Model weights (MQA + MoE) ---
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
print(f"[RLVR] total params: {len(params)}")

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
# Phase 1: SFT Pre-training
# ============================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_adam = [0.0] * len(params)
v_adam = [0.0] * len(params)

sft_steps = 1000
print(f"\n[Phase 1] SFT pre-training for {sft_steps} steps...")
sft_start = time.time()

for step in range(sft_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)
    loss.backward()

    lr_t = learning_rate * (1 - step / sft_steps)
    for i, p in enumerate(params):
        m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
        v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
        v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step+1:4d} / {sft_steps:4d} | loss {loss.data:.4f}")

sft_elapsed = time.time() - sft_start
print(f"[Phase 1] SFT done in {sft_elapsed:.1f}s")

# --- Evaluate pre-RLVR ---
def verifier(name):
    """Verifiable reward: name is exactly 5 chars and ends with 'a'"""
    if len(name) == 5 and len(name) > 0 and name[-1] == 'a':
        return 1.0
    return 0.0

def sample_names(n_samples, temperature=0.5):
    """Generate names and return (names, rewards)"""
    names = []
    for _ in range(n_samples):
        keys_g = [[] for _ in range(n_layer)]
        values_g = [[] for _ in range(n_layer)]
        token_id = BOS
        name_chars = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys_g, values_g)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            name_chars.append(uchars[token_id])
        names.append(''.join(name_chars))
    rewards = [verifier(n) for n in names]
    return names, rewards

print("\n[Pre-RLVR] Sampling 50 names...")
pre_names, pre_rewards = sample_names(50)
pre_rate = sum(pre_rewards) / len(pre_rewards) * 100
print(f"  Verifier pass rate: {pre_rate:.0f}% ({int(sum(pre_rewards))}/50)")
print(f"  Samples: {pre_names[:10]}")
matching = [n for n in pre_names if verifier(n) == 1.0]
print(f"  Matching names: {matching[:10]}")

# ============================================================
# Phase 2: RLVR with REINFORCE + Group Baseline
# ============================================================
rlvr_lr = 0.003
rlvr_steps = 300
G = 4  # group size (completions per step)

# Reset Adam state for RLVR phase
m_adam = [0.0] * len(params)
v_adam = [0.0] * len(params)

print(f"\n[Phase 2] RLVR fine-tuning for {rlvr_steps} steps (G={G})...")
print(f"  Task: generate names that are exactly 5 chars and end with 'a'")
rlvr_start = time.time()

reward_history = []

for step in range(rlvr_steps):
    # --- Sample G completions with log_probs (autograd on) ---
    group_log_probs = []
    group_rewards = []
    group_names = []

    for g in range(G):
        keys_g = [[] for _ in range(n_layer)]
        values_g = [[] for _ in range(n_layer)]
        token_id = BOS
        name_chars = []
        log_probs_seq = []

        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys_g, values_g)
            probs = softmax(logits)

            # Sample token
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

            if token_id == BOS:
                break

            # Collect log_prob of chosen token (connected to params via autograd)
            log_probs_seq.append(probs[token_id].log())
            name_chars.append(uchars[token_id])

        name = ''.join(name_chars)
        reward = verifier(name)
        group_log_probs.append(log_probs_seq)
        group_rewards.append(reward)
        group_names.append(name)

    # --- Group-normalized advantage (GRPO-style baseline) ---
    mean_r = sum(group_rewards) / G
    var_r = sum((r - mean_r) ** 2 for r in group_rewards) / G
    std_r = var_r ** 0.5

    reward_history.append(mean_r)

    # Skip update if no reward variance (all same reward → no signal)
    if std_r < 1e-8:
        for p in params:
            p.grad = 0
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step+1:4d} / {rlvr_steps:4d} | mean_r {mean_r:.2f} | (skipped, no variance)")
        continue

    advantages = [(r - mean_r) / std_r for r in group_rewards]

    # --- REINFORCE loss: -sum(log_prob * advantage) ---
    loss = Value(0.0)
    total_tokens = 0
    for g in range(G):
        if len(group_log_probs[g]) > 0:
            seq_lp = sum(group_log_probs[g])
            loss = loss + (seq_lp * (-advantages[g]))
            total_tokens += len(group_log_probs[g])

    if total_tokens > 0:
        loss = loss * (1.0 / total_tokens)
        loss.backward()

        lr_t = rlvr_lr * (1 - step / rlvr_steps)
        for i, p in enumerate(params):
            m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
            v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
            v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

    if (step + 1) % 50 == 0 or step == 0:
        recent_r = sum(reward_history[-50:]) / len(reward_history[-50:])
        print(f"  step {step+1:4d} / {rlvr_steps:4d} | mean_r {mean_r:.2f} | recent_avg {recent_r:.3f} | names: {group_names}")

rlvr_elapsed = time.time() - rlvr_start
print(f"[Phase 2] RLVR done in {rlvr_elapsed:.1f}s")

# ============================================================
# Post-RLVR Evaluation
# ============================================================
print("\n[Post-RLVR] Sampling 50 names...")
post_names, post_rewards = sample_names(50)
post_rate = sum(post_rewards) / len(post_rewards) * 100
print(f"  Verifier pass rate: {post_rate:.0f}% ({int(sum(post_rewards))}/50)")
print(f"  Samples: {post_names[:10]}")
matching = [n for n in post_names if verifier(n) == 1.0]
print(f"  Matching names: {matching[:10]}")

# --- Summary ---
print("\n" + "=" * 50)
print("[RLVR Summary]")
print(f"  Task: 5-char names ending with 'a'")
print(f"  Pre-RLVR pass rate:  {pre_rate:.0f}%")
print(f"  Post-RLVR pass rate: {post_rate:.0f}%")
print(f"  SFT time:  {sft_elapsed:.1f}s")
print(f"  RLVR time: {rlvr_elapsed:.1f}s")
print(f"  Total time: {sft_elapsed + rlvr_elapsed:.1f}s")
print("=" * 50)
