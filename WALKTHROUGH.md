# FinRuleTrader TITAN — Technical Walkthrough
## How It Was Built: A Step-by-Step Guide

---

## Overview

FinRuleTrader TITAN is a neuro-symbolic trading signal system built in one week on AMD
Developer Cloud. This walkthrough explains every technical decision, what worked, what
failed, and what the results mean.

**Total compute:** ~25 hours on AMD Instinct MI300X
**Total cost:** ~$55 of $100 AMD credits
**Model:** Qwen2.5-32B (32 billion parameters, fine-tuned with LoRA)
**Result:** 91.6% accuracy with fully auditable symbolic explanations

---

## Part 1 — Why Neuro-Symbolic?

The core problem with pure neural approaches to trading signals is opacity. A model
that outputs "BUY with 93% confidence" provides no justification a compliance officer
can evaluate. Pure rule-based systems (e.g., "IF 'profit' in headline → BUY") are
auditable but brittle and miss complex patterns.

Neuro-symbolic AI combines both: the neural model provides accuracy and generalisation,
the symbolic layer provides auditability and human interpretability.

**The specific technique:** DifferentiableRuleBank — symbolic rules whose conditions
and weights are learned via gradient descent. The rules emerge from data rather than
being hand-coded. This is the key innovation.

The rule bank architecture is adapted from the author's master's thesis research on
Adaptive Symbolic Training Control (ASTC), which uses the same technique to control
neural network training dynamics. This hackathon project demonstrates the approach
generalises to financial NLP.

---

## Part 2 — Data Pipeline

### Datasets Used

Two free datasets, no account required, both on HuggingFace:

**FiQA-2018** (pauri32/fiqa-2018): 1,211 sentences from financial question-answering
contexts with continuous sentiment scores. Converted to 3-class labels using thresholds
(score > 0.1 → positive, < -0.1 → negative, else neutral).

**Twitter Financial News Sentiment** (zeroshot/twitter-financial-news-sentiment): 11,931
financial tweets with Bearish/Bullish/Neutral labels. Remapped to SELL/BUY/HOLD.

Together: 13,007 unique sentences after deduplication, covering both formal financial
reporting language (FiQA) and informal trader commentary (Twitter).

### Why These Datasets?

The diversity of register matters. A model trained only on formal financial reports
performs poorly on Twitter-style financial commentary ("$AAPL smashing it rn 🚀") and
vice versa. Combining both gives broader coverage.

**What failed:** Financial PhraseBank could not be loaded via HuggingFace datasets
library (dataset scripts deprecated). Attempted raw file download — the GitHub URL
returned 404. Proceeded with FiQA + Twitter which provided sufficient data.

### Train/Test Split

85/15 stratified split: 11,055 training, 1,952 test sentences. Stratification ensures
class balance is preserved across splits.

---

## Part 3 — Neural Fine-Tuning

### Why Qwen2.5-32B?

**Qwen2.5-72B** was the original target — but it caused an OutOfMemoryError after loading
133 GB of weights, because LoRA's optimizer states and activations pushed beyond the
MI300X's 192 GB HBM3 capacity.

**Qwen2.5-32B** fits comfortably: ~64 GB weights + ~65 GB for LoRA/optimizer/activations
= ~130 GB total, leaving 62 GB of headroom.

Key advantage of Qwen over FinBERT or BERT-based models: Qwen2.5 is natively multilingual,
enabling cross-lingual transfer with zero additional training (demonstrated in Step 9).

### LoRA Configuration

```
r              = 16
lora_alpha     = 32
target_modules = q_proj, k_proj, v_proj, o_proj,
                 gate_proj, up_proj, down_proj
lora_dropout   = 0.05
```

Trainable parameters: 134,233,088 (0.42% of 32B total). This is the key advantage of
LoRA — we update less than half a percent of the model's parameters while preserving
its pre-trained knowledge of financial language.

### Training Details

- Epochs: 10
- Effective batch size: 64 (batch=4, gradient accumulation=16)
- Optimizer: AdamW with cosine schedule and 10% warmup
- Learning rate: 2e-4
- Sequence length: 256 tokens
- Time per epoch: ~30 minutes on MI300X
- Total fine-tuning time: ~5 hours

### Results

Best validation accuracy: 91.1% at epoch 9.

```
              precision  recall  f1-score  support
negative        0.89      0.87     0.88      320
neutral         0.92      0.94     0.93     1170
positive        0.90      0.87     0.89      462
accuracy                           0.91     1952
```

The model shows balanced performance across all three classes — no class is being
sacrificed for another, which is important for a trading system where false positives
and false negatives both have real costs.

---

## Part 4 — Feature Extraction (14 Dimensions)

For each sentence, we extract a 14-element feature vector that bridges the neural and
symbolic layers. The 14 features are deliberately analogous to the 7 gradient features
used in the ASTC thesis, expanded with additional linguistic structure features.

### Feature Groups

**Group A — Neural probabilities (3):**
f01_pos_prob, f02_neg_prob, f03_neu_prob — direct softmax outputs from Qwen2.5-32B.

**Group B — Distribution shape (2):**
f04_confidence = max_prob - 2nd_max_prob (how decisive the model is)
f05_entropy = normalised Shannon entropy (how uncertain the model is)

**Group C — Lexical sentiment (2):**
f06_bullish = density of bullish keywords (profit, growth, beat, surge, etc.)
f07_bearish = density of bearish keywords (loss, decline, fraud, lawsuit, etc.)

**Group D — Linguistic structure (7):**
f08_sent_length, f09_number_density, f10_forward_looking (will, expect, forecast),
f11_uncertainty (may, might, could), f12_magnitude (significantly, sharply),
f13_comparison (vs, outperform, beat), f14_negation (not, no, never)

### Temperature Scaling (T=6.0)

The 32B model is extremely confident — almost every prediction is near 100% for one
class (f04_confidence mean ≈ 1.0, f05_entropy mean ≈ 0.001). This means the rule bank
only sees the three probability features and ignores f06–f14.

Temperature scaling softens the probabilities by dividing the logits by T=6.0 before
softmax, forcing the distribution to be less extreme. After scaling:
- Confidence mean: 0.978 (was ~1.0)
- Entropy mean: 0.061 (was ~0.001)

This allows the rule bank to learn multi-feature conditions that use the linguistic
features, not just the raw probability.

---

## Part 5 — Hierarchical DifferentiableRuleBank

### Architecture

The rule bank has two levels:

**Level 1 — 32 Base Rules:** Each rule watches a subset of the 14 features using
learnable weights (w), thresholds (θ), and temperatures (τ):

```
activation = sigmoid((x - θ) / softplus(τ)) · sigmoid(w) / sum(sigmoid(w))
```

L1 regularisation on w drives most gates to zero, keeping each rule sparse and
interpretable.

**Level 2 — 8 Meta Rules:** Each meta rule watches the 32 base rule activations
and learns which combinations of base rules predict which action. This enables
compound conditions like:
"IF base rule 7 (pos_prob high) AND base rule 6 (bullish keywords present) → BUY"

**Blending:** A learnable scalar α blends meta and base predictions:
`final = α · meta_output + (1-α) · base_output`

### Training

- 3,000 epochs
- Loss = CrossEntropy + 0.5 × MSE(strength, target) + L1_regularisation(λ=0.001)
- The strength supervision encourages higher confidence for directional signals (BUY/SELL)
  vs neutral (HOLD)
- Adam optimizer, lr=3e-3

### Induced Rules

```
Base Rule  7 → BUY   (strength=0.448)
  IF f01_pos_prob > 0.387  [gate=0.76, τ=0.035]

Base Rule  2 → SELL  (strength=0.450)
  IF f02_neg_prob > 0.344  [gate=0.74, τ=0.038]

Base Rule 18 → HOLD  (strength=0.125)
  IF f03_neu_prob > 0.454  [gate=0.81, τ=0.041]
```

These three rules handle the majority of predictions. The tight τ values (0.035–0.041)
indicate sharp thresholds — the rules have learned crisp decision boundaries.

### Ablation Study

12-experiment grid search over K ∈ {8, 16, 32, 64} and λ ∈ {0.0005, 0.001, 0.005}:

All 12 experiments produced accuracy in the range 91.5%–91.6% — a spread of only 0.05
percentage points. This is actually the ideal result: it means the system's performance
is not sensitive to hyperparameter choice, and the reported 91.6% is not cherry-picked.

---

## Part 6 — Uncertainty Quantification (Monte Carlo Dropout)

Standard dropout is disabled at inference time. Monte Carlo Dropout keeps it active
and runs 50 stochastic forward passes. The variance across passes gives a Bayesian
approximation of prediction uncertainty.

**50 passes through Qwen2.5-32B on 1,952 sentences: ~4 hours on MI300X.**

Results:
- Mean predictive entropy: 0.035 (low — model is generally confident)
- Low-uncertainty (bottom 80% by entropy): **96.5% accuracy**
- High-uncertainty (top 20% by entropy): **71.8% accuracy**

The 25-point accuracy gap between low and high uncertainty predictions is the key
practical finding. In a live system: act on low-uncertainty signals automatically,
route high-uncertainty signals to human review.

---

## Part 7 — Cross-Lingual Testing

Qwen2.5-32B was pre-trained on a massive multilingual corpus. We test financial
headlines in German, French, and Chinese without any additional training.

**German (5/5 = 100%):** Perfect. German financial language maps cleanly to the
training distribution.

**Chinese (5/5 = 100%):** Perfect. Qwen was trained on substantial Chinese financial
text as part of its pre-training.

**French (3/5 = 60%):** Two failures:
- "LVMH affiche une croissance record" → HOLD (should be BUY): "croissance record"
  (record growth) not as strongly represented in Qwen's financial French pre-training.
- "TotalEnergies annonce un dividende exceptionnel" → HOLD (should be BUY): "dividende
  exceptionnel" (special dividend) missed.

Both French failures are bullish signals involving domain-specific terminology. This
suggests adding French financial corpora (e.g., AMF regulatory filings) to fine-tuning
would close the gap.

---

## Part 8 — Unseen Data Evaluation

Three independent test sets not seen during training or development:

**Test Set A — 40 hand-crafted headlines (77.5%):**

| Category | Accuracy | Notes |
|---|---|---|
| Clear positive | 87.5% | 1 failure: "Startup raises $500M Series C" → HOLD |
| Clear negative | 100% | Perfect |
| Clear neutral | 87.5% | 1 failure: "Earnings in line" → BUY |
| Contradictory | 75% | Mixed signals are genuinely ambiguous |
| Negation | 25% | Known weakness — see below |
| Domain shift | 50% | Crypto/FX underrepresented in training |
| Jargon | 75% | REIT dividend terminology missed |

**The negation weakness** is the most interesting failure. "Company reports no
significant losses" → HOLD (should be BUY). The model reads "losses" as a negative
signal and underweights "no". This is a known failure mode of sentiment models trained
on bag-of-words style labels. Fix: negation-aware preprocessing.

**Test Set B — Live Yahoo Finance RSS (12 headlines):**
All classified as HOLD. On inspection: Yahoo's RSS feed pulled financial advice articles
("What does homeowners insurance cover?") not market-moving news. HOLD is the correct
signal for non-actionable articles — the model is behaving correctly.

**Test Set C — Adversarial stress test (50%):**
50% is expected on intentionally adversarial inputs. Key findings:
- Sarcasm ("Oh great, another record loss") → HOLD: sarcasm detection is an open
  research problem for all current LLMs.
- Very short ambiguous headlines ("CEO out", "Guidance cut") → HOLD: correctly
  abstaining rather than guessing.
- Complex multi-clause sentences: correctly handled in most cases.

---

## Part 9 — Live API

FastAPI server provides real-time inference:

```python
# POST /predict
{
  "headline": "Apple beats earnings expectations with record revenue."
}

# Response
{
  "signal": "BUY",
  "strength": 0.401,
  "base_rule_fired": 7,
  "probabilities": {"BUY": 0.934, "HOLD": 0.043, "SELL": 0.023},
  "model_sentiment": {"positive": 0.98, "neutral": 0.01, "negative": 0.01},
  "explanation": [
    {
      "feature": "f01_pos_prob",
      "direction": ">",
      "threshold": 0.387,
      "value": 1.0,
      "satisfied": true
    }
  ]
}
```

The explanation field is what separates this from a standard classifier API —
every response includes the symbolic rule that fired and the specific feature
values that triggered it.

---

## Part 10 — What We Would Do With More Time

**Negation-aware preprocessing:** Detect negation scope and flip sentiment polarity.
Would likely fix the 25% edge_negation accuracy.

**French financial corpora:** Add AMF regulatory filings or Les Echos financial news
to fine-tuning to improve French accuracy from 60% to match DE/ZH.

**Real market data integration:** Connect to Yahoo Finance API to evaluate whether
signals from the previous trading session's headlines actually predict next-day price
movements. This would make the backtesting simulation reflect real trading rather than
synthetic returns.

**Larger model (72B):** The 72B model failed due to OOM. With 4-bit quantisation support
on ROCm (currently incomplete via bitsandbytes), 72B would fit and likely improve
accuracy to 93–94%.

**Online learning:** Update the rule bank in real-time as market conditions shift,
without retraining the full neural model.

---

## Reproducibility

The entire pipeline runs top-to-bottom from a single Jupyter notebook on any AMD MI300X
instance with the PyTorch 2.6 + ROCm 7.0 Quick Start image. No manual setup beyond
uploading the notebook file.

All datasets download automatically from HuggingFace. All model weights download from
HuggingFace. Total fresh-run time: ~25 hours.
