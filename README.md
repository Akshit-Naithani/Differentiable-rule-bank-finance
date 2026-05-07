# FinRuleTrader TITAN
### Neuro-Symbolic Trading Signal System — AMD MI300X Edition

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.0-red)](https://rocm.docs.amd.com/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--32B-blue)](https://huggingface.co/Qwen/Qwen2.5-32B)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.6%25-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

A neuro-symbolic AI system that generates interpretable BUY / HOLD / SELL trading
signals from financial news headlines. Built for the AMD Developer Cloud Hackathon
on a single AMD Instinct MI300X GPU.

---

## What It Does

Every signal comes with a plain-English symbolic explanation of exactly why it fired:

```
Headline : "Nvidia beats earnings by 40%, announces $10B buyback."
Signal   : 🟢 BUY  (strength=0.401)
Rule     : Base Rule 7 → Meta Rule 4
Why      :
  ✓ f01_pos_prob > 0.387  (actual=1.000, gate=0.76)
```

No black box. Every decision is traceable to a human-readable rule.

---

## Architecture

```
Financial Headline
       │
       ▼
┌─────────────────────────────────┐
│   Qwen2.5-32B + LoRA            │  Neural layer
│   Fine-tuned on 13,000+         │  (32B parameters, 0.42% trainable)
│   financial sentences           │
└─────────────────────────────────┘
       │
       ▼  14-feature vector
       │  [pos_prob, neg_prob, neu_prob, confidence, entropy,
       │   bullish_density, bearish_density, sent_length,
       │   number_density, forward_looking, uncertainty,
       │   magnitude, comparison, negation]
       │
       ▼
┌─────────────────────────────────┐
│   Hierarchical DifferentiableRuleBank   │  Symbolic layer
│                                 │  (ported from ASTC thesis)
│   Level 1: 32 base rules        │
│   Level 2:  8 meta rules        │
└─────────────────────────────────┘
       │
       ▼
  BUY / HOLD / SELL + Symbolic Explanation
```

The **DifferentiableRuleBank** is the core neuro-symbolic contribution. Rules are not
hand-coded — they are learned via gradient descent with L1 sparsity regularisation,
producing human-readable conditions like:

```
Base Rule 7  →  BUY  (strength=0.448)
  IF f01_pos_prob > 0.387  [gate=0.76, τ=0.035]

Base Rule 2  →  SELL  (strength=0.450)
  IF f02_neg_prob > 0.344  [gate=0.74, τ=0.038]
```

---

## Results

| Metric | Value |
|---|---|
| Model | Qwen2.5-32B + LoRA (134M trainable / 32B total) |
| Training data | 13,007 sentences (FiQA-2018 + Twitter Financial News) |
| Fine-tune accuracy | **91.1%** |
| Rule bank accuracy | **91.6%** (hierarchical K=32 base + K=8 meta) |
| Calibration error | **0.034** (near-perfect, 0.0 = ideal) |
| MC Dropout (low-uncertainty 80%) | **96.5% accuracy** |
| Cross-lingual | **86.7%** (German 100%, Chinese 100%, French 60%) |
| Unseen hand-crafted headlines | **77.5%** (40 novel headlines) |
| Ablation stability | 91.5%–91.6% across all 12 K×λ experiments |

### Key Finding — Uncertainty Filtering

Monte Carlo Dropout (50 passes) identifies uncertain predictions:

| Subset | Coverage | Accuracy |
|---|---|---|
| Low-uncertainty signals | 80% | **96.5%** |
| High-uncertainty signals | 20% | 71.8% |

In a live trading system: act only on low-uncertainty signals, flag the rest for human review.

### Cross-Lingual Generalisation (zero additional training)

| Language | Accuracy |
|---|---|
| German | 5/5 = 100% |
| Chinese | 5/5 = 100% |
| French | 3/5 = 60% |

Qwen2.5-32B generalises to multilingual financial text without any multilingual fine-tuning.

---

## AMD Hardware Utilisation

| Spec | Value |
|---|---|
| GPU | AMD Instinct MI300X |
| VRAM | 192 GB HBM3 |
| Framework | PyTorch 2.6 + ROCm 7.0 |
| Model size | ~64 GB in bfloat16 |
| Training time | ~5 hours (10 epochs) |
| Feature extraction | ~2 hours |
| MC Dropout (50 passes) | ~4 hours |

The MI300X's 192 GB HBM3 is what makes this possible — it is one of the only single
GPUs that can run a 32B+ parameter model for fine-tuning without model parallelism
or quantisation.

---

## Live API

A FastAPI server runs on port 8787:

```bash
# Single prediction
curl -X POST http://YOUR_IP:8787/predict \
     -H "Content-Type: application/json" \
     -d '{"headline": "Apple beats earnings expectations with record revenue."}'

# Response
{
  "signal": "BUY",
  "strength": 0.401,
  "base_rule_fired": 7,
  "probabilities": {"BUY": 0.934, "HOLD": 0.043, "SELL": 0.023},
  "explanation": [
    {"feature": "f01_pos_prob", "direction": ">", "threshold": 0.387,
     "value": 1.0, "satisfied": true}
  ]
}

# Other endpoints
GET  /health          — system status and model info
GET  /rules           — all 32 base + 8 meta induced rules
GET  /stats           — accuracy metrics and signal distribution
POST /batch_predict   — up to 20 headlines at once
```

---

## Datasets

| Dataset | Source | Sentences | License |
|---|---|---|---|
| FiQA-2018 | HuggingFace (pauri32/fiqa-2018) | 1,211 | Free |
| Twitter Financial News | HuggingFace (zeroshot/twitter-financial-news-sentiment) | 11,931 | Free |

No accounts, no contracts, no MIMIC-style gating. Everything downloads automatically.

---

## Technical Stack

- **PyTorch 2.6 + ROCm 7.0** — GPU training and inference
- **HuggingFace Transformers** — model loading and tokenization
- **PEFT (LoRA)** — parameter-efficient fine-tuning
- **FastAPI + Uvicorn** — REST inference server
- **scikit-learn** — evaluation metrics
- **Matplotlib + Seaborn** — visualisation

---

## Project Structure

```
FinRuleTrader_TITAN.ipynb   ← main notebook (11 steps, run top to bottom)
data/
  train.csv                 ← 11,055 training sentences
  test.csv                  ← 1,952 test sentences
checkpoints/
  qwen_finetuned/           ← LoRA-adapted Qwen2.5-32B checkpoint
features/
  train_features.pt         ← 14-dimensional feature vectors (train)
  test_features.pt          ← 14-dimensional feature vectors (test)
outputs/
  rule_bank.pt              ← trained hierarchical rule bank weights
  induced_rules.txt         ← human-readable symbolic rules
  signal_log.json           ← full signal log with explanations
  ablation.csv              ← 12-experiment K×λ ablation results
plots/
  step2_training.png        ← fine-tuning curve
  ablation.png              ← K×λ heatmap
  performance.png           ← portfolio + calibration + rule frequency
  error_analysis.png        ← confusion matrix
  mc_dropout.png            ← uncertainty analysis
  unseen_eval.png           ← unseen data evaluation
unseen_test/
  all_results.json          ← Test Set A/B/C results
  crosslingual_results.json ← DE/FR/ZH results
  live_headlines.json       ← live RSS classification
```

---

## Known Limitations

**Negation handling (25% accuracy on negated positives):** The model reads "no losses"
as containing "losses" and under-weights the negation. Future work: negation-aware
preprocessing that flips sentiment polarity when negation tokens precede sentiment words.

**French financial terminology (60%):** Terms like "dividende exceptionnel" and "croissance
record" were under-represented in Qwen's financial pre-training data. Future work:
include French financial corpora (e.g., AMF filings) in fine-tuning.

**Very short headlines (ambiguous):** "CEO out." and "Guidance cut." lack enough context
for confident classification — the model correctly returns HOLD (abstain) rather than
guessing, which is actually the right behaviour for a risk-averse trading system.

---

## Relation to ASTC Research

The `DifferentiableRuleBank` architecture is adapted from the author's master's thesis
work on **Adaptive Symbolic Training Control (ASTC)** — a neuro-symbolic system that
uses differentiable rules to control neural network training dynamics. In FinRuleTrader,
the same rule bank architecture is applied to a completely different domain (financial
sentiment → trading signals) to demonstrate cross-domain generalisability of the
technique.

---

## License

MIT — see LICENSE file.

## Acknowledgements

Built on AMD Instinct MI300X via AMD Developer Cloud. Fine-tuning and inference
powered by ROCm 7.0 and PyTorch 2.6.
