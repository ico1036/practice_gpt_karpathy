# microGPT Experiments

Experimental variants of [Andrej Karpathy's microGPT](https://karpathy.github.io/2026/02/12/microgpt/) — a ~200-line, dependency-free GPT implementation in pure Python with scalar autograd.

This repo explores architectural modifications (MQA, MoE) and training methods (RLVR) to understand their impact at the smallest possible scale.

## Files

| File | Description |
|------|-------------|
| `microgpt_original.py` | Original microGPT with **Multi-Head Attention (MHA)** |
| `microgpt_mqa.py` | Modified to use **Multi-Query Attention (MQA)** |
| `microgpt_moe.py` | MQA + **Mixture of Experts (MoE)** replacing the dense MLP |
| `microgpt_rlvr.py` | MQA + MoE + **RLVR** (REINFORCE with verifiable rewards) |

## What Changed

### MHA → MQA (`microgpt_mqa.py`)

Multi-Query Attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) shares a single Key and Value head across all query heads.

```
MHA:  Q [n_embd × n_embd], K [n_embd × n_embd], V [n_embd × n_embd]
MQA:  Q [n_embd × n_embd], K [head_dim × n_embd], V [head_dim × n_embd]
```

- K/V projections shrink from `16×16` to `4×16`
- All 4 query heads share the same K/V — no per-head slicing needed
- At scale, this dramatically reduces KV cache memory during inference

### MQA + MoE (`microgpt_moe.py`)

Mixture of Experts replaces the single dense MLP with a sparse, routed ensemble.

```
Dense MLP:   x → fc1 (4× width) → ReLU → fc2 → out
MoE:         x → Router → top-2 of 4 experts
               Expert_i: x → fc1 (2× width) → ReLU → fc2 → out
             → weighted sum of selected expert outputs
```

- 4 expert MLPs, each with 2× hidden width (half of original 4×)
- Router (linear layer) produces gate logits, selects top-2 experts per token
- Softmax over top-2 gate scores determines mixing weights
- Active compute per token stays similar, but total model capacity increases

### MoE + RLVR with Scratchpad (`microgpt_rlvr.py`)

RLVR (Reinforcement Learning with Verifiable Rewards) adds a post-SFT fine-tuning phase using REINFORCE with a verifiable reward function. This implementation includes a **scratchpad mechanism** to test emergent intermediate computation (micro-scale Chain-of-Thought).

```
Phase 1 (SFT):        next-token prediction on names (1000 steps)
Phase 2 (Cold Start):  teach | separator format with random scratchpad (100 steps)
Phase 3 (RLVR):        REINFORCE + group baseline (500 steps, G=4)
```

**Scratchpad design:** A `|` separator token is added to the vocabulary. The model can optionally generate tokens before `|` (scratchpad) — only tokens after `|` are evaluated by the verifier. The model decides whether and when to use `|`.

```
Without scratchpad:  "an" → "na"           → verify "anna" ∈ dataset
With scratchpad:     "an" → "l|na"         → ignore "l", verify "anna" ∈ dataset
                            ^^^              ^^
                         scratchpad        answer only
```

- Verifiable task: given 2-char prefix, complete to a real name in the dataset
- Cold start teaches the FORMAT only (random scratchpad content) — not HOW to think
- RLVR then determines whether using scratchpad actually improves accuracy

## Results

### Architecture Comparison (SFT only)

All models trained on the [names dataset](https://github.com/karpathy/makemore) for 1000 steps with identical hyperparameters (`n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01`).

| Metric | MHA (Original) | MQA | MQA + MoE |
|--------|----------------|-----|-----------|
| Total params | 4,192 | 3,808 | 5,920 |
| Active params/token | 4,192 | 3,808 | 3,872 (top-2/4) |
| Training time | 84.7s | 75.7s | 74.7s |
| Final loss | 2.6497 | 2.6369 | **2.5166** |
| Param reduction | — | -9.2% | +41.2% total, -7.6% active |

### Key Takeaways

1. **MQA** reduces parameters by 9.2% and training time by 10.6% with no loss degradation — the shared K/V is sufficient at this scale.
2. **MoE** achieves the lowest loss (2.52) by increasing total capacity while keeping per-token compute comparable to MQA. The router learns to specialize experts for different input patterns.
3. The real-world benefit of MoE scales with model size: more experts = more total knowledge, but constant inference cost. This is the strategy behind Mixtral, DeepSeek, and other production MoE models.

### Sample Outputs (SFT)

**MHA**: kamon, ann, karai, jaire, vialan, karia, anna, areli, keylen, anton

**MQA**: kerila, anarin, jozeri, janni, chiah, orely, elelin, jalen, dalena, kadan

**MoE**: karan, meeran, kana, seane, maran, laelin, alenan, dane, arel, ladid

### RLVR Scratchpad: Content vs Compute Ablation

The central question: does a scratchpad help because the model **thinks** (content matters), or because it gets **extra forward passes** (any tokens help)?

**Experiment:** Train ONE model (SFT → Cold Start → RLVR), then evaluate with FOUR inference modes on the same 200 prompts:

| Eval | Mode | Accuracy | vs Baseline |
|------|------|----------|-------------|
| A | `\|` blocked (baseline) | 0.5% | — |
| B | `\|` allowed (natural) | **11.0%** | **+10.5%p** |
| C | Forced random scratchpad | 10.5% | +10.0%p |
| D | Model's scratchpad → random | 12.8% | +12.3%p |

**Key comparisons:**

| Comparison | Delta | Interpretation |
|------------|-------|----------------|
| B vs A | +10.5%p | Scratchpad helps overall |
| B vs C | +0.5%p | Model's content ≈ random content |
| B vs D | -1.8%p | Replacing model's scratch doesn't hurt |
| C vs A | +10.0%p | Random tokens also help just as much |

**Observations:**
- `|` usage rate: **99.5%** — the model almost always chose to use scratchpad
- `|` usage during RLVR training: 54% → 74% (model learned to use it **more** over time)
- Average scratchpad length: **1.0 tokens**

**Example outputs:**
```
tu + r|la → "turla" ✓    (scratchpad: "r", answer: "la")
ab + y|y  → "aby"   ✓    (scratchpad: "y", answer: "y")
lo + r|la → "lorla" ✓    (scratchpad: "r", answer: "la")
```

### Interpretation

The 4-way ablation reveals that the scratchpad benefit is **compute-driven, not content-driven**:

1. **Scratchpad helps** (B > A by +10.5%p): Blocking `|` at inference drops accuracy from 11% to 0.5%. The model clearly depends on the scratchpad mechanism.

2. **Content doesn't matter** (B ≈ C ≈ D): Random scratchpad tokens perform equally well (10.5%) as the model's own chosen tokens (11.0%). Replacing the model's scratchpad with random content (Eval D: 12.8%) doesn't hurt — it even slightly helps.

3. **It's extra compute**: Each additional token provides an extra forward pass through the attention mechanism. Subsequent tokens can attend to the computed hidden state of the scratchpad token, giving the model more sequential computation before committing to an answer. The content of that computation doesn't matter — just having *any* token there creates a richer attention context.

This is the same principle behind Chain-of-Thought at scale (more tokens = more sequential computation = better answers), but at this 5,952-parameter scale, we can prove it's a **compute effect** rather than **emergent reasoning**. The model learned to use `|` as a "compute buffer" — not to think, but to buy itself an extra forward pass. True content-dependent reasoning likely requires more model capacity.

## Usage

```bash
uv run python microgpt_original.py   # MHA baseline
uv run python microgpt_mqa.py        # MQA variant
uv run python microgpt_moe.py        # MQA + MoE
uv run python microgpt_rlvr.py       # MQA + MoE + RLVR
```

## Credits

- Original microGPT by [Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/) ([gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95))
- MQA: [Fast Transformer Decoding (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)
- MoE: [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- RLVR/GRPO: [DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- CoT emergence: [DeepSeek-R1 (Guo et al., 2025)](https://arxiv.org/abs/2501.12948)
