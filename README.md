# Multi-Perspective Synthesis Language Model

A small language model built **from scratch** around a novel idea: instead of a single
forward pass, one input is processed **simultaneously by six independent "perspective"
engines**, whose outputs are then woven back together into a single response.

Everything in the core track — embeddings, attention, feed-forward, layer norm, the loss,
and **all of backpropagation** — is hand-written in NumPy. No autograd, no ML framework.
A separate PyTorch track is included as a "known-correct" reference and an interactive chat.

---

## Table of Contents
- [The Core Idea](#the-core-idea-fan-out--converge)
- [Architecture & Data Flow](#architecture--data-flow-numpy-track)
- [The Six Perspectives](#the-six-perspectives)
- [PyTorch Reference Track](#pytorch-reference-track)
- [Getting Started](#getting-started)
- [Training](#training)
- [Project Status](#project-status)
- [Repository Layout](#repository-layout)

---

## The Core Idea: Fan-Out → Converge

```
                                 ┌──────────────┐
                             ┌──▶│  Optimist    │──┐
                             │   ├──────────────┤  │
                             ├──▶│  Pessimist   │──┤
                             │   ├──────────────┤  │
  input ─▶ tokenize ─▶ embed ┼──▶│  Analyst     │──┼─▶ synthesize ─▶ output ─▶ text
           (+ pos.enc.)      │   ├──────────────┤  │   (converge)     project
                             ├──▶│  Philosopher │──┤
                             │   ├──────────────┤  │
                             ├──▶│  Innovator   │──┤
                             │   ├──────────────┤  │
                             └──▶│  Pragmatist  │──┘
                                 └──────────────┘
     shared embedding + attention      each node = embed → attention → FFN
```

1. **Fan-out** — the same query is prefixed with six different *system prompts*
   (perspectives) and run through six `PerspectiveNode` instances **concurrently**
   (`run_nodes.py` uses a `ThreadPoolExecutor`).
2. **Per-node processing** — each node runs `embed → self-attention → feed-forward`,
   producing its own sequence of vectors. Nodes share the embedding and attention weights
   but see different prompts, so each produces a different "reading" of the input.
3. **Converge** — `synthesizer.py` merges the six sequences (padding shorter ones) into a
   single sequence with an equal-weight mean.
4. **Project** — `output_layer.py` maps the synthesized vectors back to vocabulary logits
   and samples tokens with temperature and a repetition penalty.

---

## Architecture & Data Flow (NumPy track)

| # | Stage | Module | Class / fn | Responsibility |
|---|-------|--------|-----------|----------------|
| 1 | Tokenize | `tokenizer.py` | `build_vocab`, `encode`, `decode` | Words → integer IDs. Special `<PAD>`/`<UNK>` tokens. |
| 2 | Embed | `embeddings.py` | `EmbeddingMap`, `PositionalEncoding` | IDs → vectors, scaled by √d, plus sinusoidal positional encoding. |
| 3 | Attention | `attention.py` | `SelfAttention` | Scaled dot-product self-attention + residual + LayerNorm. Full `backward()`. |
| 4 | Feed-forward | `ffn.py` | `FeedForward` | Two-layer ReLU MLP (4× hidden) + residual. Full `backward()`. |
| 5 | Perspective node | `node.py` | `PerspectiveNode` | Bundles prompt + embed + attention + FFN into one perspective. |
| 6 | Fan-out | `run_nodes.py` | `run_all_nodes` | Runs all nodes concurrently, returns results ordered by name. |
| 7 | Converge | `synthesizer.py` | `synthesize`, `ConvergenceLayer` | Merge node sequences into one (mean; learned layer is opt-in). |
| 8 | Project | `output_layer.py` | `OutputProjection` | Vectors → logits → sampled tokens. Full `backward()`. |
| — | Loss | `loss.py` | `cross_entropy_loss` | Sequence cross-entropy. |
| — | Backprop | `backprop.py` | `backpropagate_step_sequence` | Chains gradients back: projection → FFN → attention → embeddings. |
| — | Inference | `main.py` | `main` | Full pipeline; auto-loads trained weights if present. |
| — | Training | `train.py` | `run_training_loop` | Training loop, cosine LR decay, gradient clipping, weight persistence. |

**Key techniques implemented by hand:**
- Scaled dot-product self-attention with residual connection and layer normalization,
  plus the full analytical backward pass through all of it.
- Xavier/Glorot weight initialization and sinusoidal positional encodings.
- Cross-entropy loss with a numerically stable softmax.
- L2 gradient clipping (`max_norm = 1.0`) and a cosine-decay learning-rate schedule.
- Weight persistence to `.npy` / `.json` and automatic reloading at inference time.

### Dynamic embedding dimension
`train.py::calculate_dim_for_fanout` sizes the embedding dimension from the vocabulary
size and node count ("The 6-Node Multiplier Rule"), rounding up to a power of two. For the
bundled 131-word vocabulary this yields **embedding_dim = 128**.

---

## The Six Perspectives

The same weights are steered into six different readings by prepending a system prompt.
`main.py` and `train.py` use the **same** six prompts so trained weights match inference:

```python
prompts = {
    "Optimist":    "Focus on the positive potential and opportunities.",
    "Pessimist":   "Identify the risks, flaws, and constraints.",
    "Analyst":     "Look purely at facts, data, and logical sequences.",
    "Philosopher": "Consider the ethical and deeper meaning implications.",
    "Innovator":   "Think outside the box and suggest novel paradigms.",
    "Pragmatist":  "Focus on actionable, realistic execution steps.",
}
```

---

## PyTorch Reference Track

A parallel, framework-based implementation used to validate the concept:

- **`model_pt.py`** — `CustomTransformer`: scaled + positional `EmbeddingMap` →
  `MultiHeadAttention` (**causal-masked**, 4 heads) → `FeedForward` → `OutputLayer`.
  This is the *properly autoregressive* version (predicts the next token).
- **`train_pt.py`** — Adam-optimized next-token training loop with a generation demo.
- **`chat.py`** — Interactive REPL with a `DynamicTokenizer` (spell-checking, on-the-fly
  vocabulary growth up to 5000 words, optional gensim word vectors) that persists learned
  words to `memory.csv`.

> Note: the two tracks reuse some class names (`EmbeddingMap`, `PerspectiveNode`,
> `FeedForward`) for different implementations — use the tables above to tell them apart.

---

## Getting Started

**Requirements:** Python 3.11+

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt      # everything
# or, core track only:
pip install numpy
```

### Run inference (NumPy track)
```bash
python main.py
```
If the trained weight files (`vocab.json`, `embedding_weights.npy`, `attention_W_*.npy`,
`output_matrix.npy`) are present, they load automatically; otherwise the model initializes
randomly and builds a vocabulary from the built-in corpus.

Example output:
```
3. The Fan-Out: Running 6 perspectives simultaneously on input query: 'what is data'
   -> Node [Analyst] completed. Generated 11 position vectors.
   ...
Final Model Output: 'gases devices plants mixture data toward being other events light statistics'
```

### PyTorch reference / chat
```bash
python train_pt.py   # autoregressive training demo
python chat.py       # interactive chat (needs torch, gensim, pyspellchecker)
```

---

## Training

```bash
python train.py
```
Runs the full loop over a built-in Q→A dataset (definitions of *data*, *code*, *gravity*,
*DNA*, …), then saves all weights to disk:

- `vocab.json` — token → id map
- `embedding_weights.npy` — embedding matrix
- `attention_W_q.npy`, `attention_W_k.npy`, `attention_W_v.npy` — attention weights
- `output_matrix.npy` — output projection

Highlights of the loop: cosine-decayed learning rate, per-epoch shuffling, automatic query
padding so input length matches the target sequence, gradient-clipping, and a periodic
generation sample so you can watch the model learn.

---

## Project Status

**Working & verified**
- End-to-end inference and text generation (NumPy track) — runs with 6 perspectives and
  correctly merges their differing sequence lengths.
- Hand-written backprop through LayerNorm, residuals, attention, FFN, and embeddings, with
  L2 gradient clipping and cosine LR decay.
- Training converges (overfits the small dataset toward very low loss); weights save and
  reload correctly (verified shapes: 131 × 128 embeddings, 128 × 128 attention,
  128 × 131 output).
- PyTorch reference trains autoregressively and generates.

**Recently fixed**
- `main.py` was missing `import math`, which crashed the random-initialization path — added.
- `train.py` trained only 4 perspectives while the concept and `main.py` used 6 — both now
  use the same 6 prompts, so trained weights align with inference.
- The synthesizer previously built a fresh, **randomly initialized** `ConvergenceLayer` on
  every call, so convergence weighting was nondeterministic and inconsistent with the
  backward pass. `synthesize()` now uses a deterministic equal-weight mean that matches the
  gradient distribution in `backprop.py`. The learned `ConvergenceLayer` is kept as a
  documented, opt-in path for future work.
- Added `.gitignore` and `requirements.txt`; stopped tracking `__pycache__`.

**Known limitations / next steps**
- The NumPy model emits one token per synthesized position rather than predicting the
  *next* token — it is not truly autoregressive (the PyTorch model is). Adding
  autoregressive generation to the NumPy track is the biggest opportunity.
- Nodes share a single attention/FFN instance whose forward cache holds only the last
  node processed, so the backward pass is an approximation. Giving each node its own
  module instances (or processing sequentially) would make gradients exact.
- Train the `ConvergenceLayer` and route its gradients through `backprop.py` to bring the
  learned synthesis into the live pipeline.
- Finish `RelationalPerspectiveNode` (in `node.py`) — encodes *relationships between*
  concepts rather than the concepts themselves (see `test_relations.py`).

---

## Repository Layout

```
Core (NumPy)      tokenizer.py  embeddings.py  attention.py  ffn.py  node.py
                  run_nodes.py  synthesizer.py  output_layer.py  loss.py
                  backprop.py   main.py  train.py
PyTorch           model_pt.py  train_pt.py  chat.py
Experiments       test_fake_data.py  test_relations.py  augment_data.py
                  photon.py  (standalone physics check, unrelated to the model)
Trained weights   vocab.json  embedding_weights.npy  attention_W_{q,k,v}.npy
                  output_matrix.npy
Config            requirements.txt  .gitignore
Docs              docs/*.md   (per-module notes; docs/modifications.md = changelog)
```

---

*This project is a from-scratch educational exploration of transformer internals and a
novel multi-perspective synthesis architecture. It is a research prototype, not a
production model.*
