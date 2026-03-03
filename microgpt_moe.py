"""
microGPT with MQA + Mixture of Experts (MoE)
MLP layer replaced with sparse MoE: router picks top-2 of 4 experts per token.
Each expert is a smaller MLP (2x instead of 4x), so active params per token stay similar.
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

# MoE config
n_experts = 4
top_k = 2
expert_mult = 2  # each expert: 2x width (vs original 4x for single MLP)

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# --- Model weights (MQA + MoE) ---
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

for i in range(n_layer):
    # MQA attention (same as before)
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(head_dim, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(head_dim, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # Router: n_embd -> n_experts
    state_dict[f'layer{i}.moe_gate'] = matrix(n_experts, n_embd)
    # Expert MLPs: each expert has fc1 (expert_mult*n_embd x n_embd) and fc2 (n_embd x expert_mult*n_embd)
    for e in range(n_experts):
        state_dict[f'layer{i}.expert{e}.fc1'] = matrix(expert_mult * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, expert_mult * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"[MoE] total params: {len(params)}")

# Count active params per forward pass (top-k experts only)
attn_params = n_embd*n_embd + head_dim*n_embd*2 + n_embd*n_embd  # Q + K + V + O
gate_params = n_experts * n_embd
single_expert_params = expert_mult * n_embd * n_embd + n_embd * expert_mult * n_embd
active_mlp_params = gate_params + top_k * single_expert_params
embed_params = vocab_size * n_embd * 2 + block_size * n_embd
print(f"[MoE] active params per token: {embed_params + attn_params + active_mlp_params} (top-{top_k} of {n_experts} experts)")

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
    """Sparse MoE: route to top-k experts, combine outputs by gate weights."""
    # Router logits
    gate_logits = linear(x, state_dict[f'layer{layer_idx}.moe_gate'])
    # Top-k selection (greedy)
    indexed = [(gate_logits[e], e) for e in range(n_experts)]
    indexed.sort(key=lambda t: t[0].data, reverse=True)
    top_experts = indexed[:top_k]
    # Softmax over top-k gate scores only
    gate_weights = softmax([t[0] for t in top_experts])

    # Run selected experts and combine
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
        # --- Attention (MQA) ---
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

        # --- MoE MLP (replaces dense MLP) ---
        x_residual = x
        x = rmsnorm(x)
        x = moe_mlp(x, li)
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# --- Training ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000

print(f"\n[MoE] Training for {num_steps} steps...")
start_time = time.time()

for step in range(num_steps):
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

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

elapsed = time.time() - start_time
print(f"[MoE] Training time: {elapsed:.1f}s")

# --- Inference ---
temperature = 0.5
print("\n[MoE] --- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        if token_id == BOS:
            break

        sample.append(uchars[token_id])

    print(f"  sample {sample_idx+1:2d}: {''.join(sample)}")
