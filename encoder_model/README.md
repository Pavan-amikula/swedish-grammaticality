---
license: apache-2.0
language:
- sv
library_name: transformers
tags:
- grammaticality
- regression
- modernbert
base_model: AI-Sweden-Models/ModernBERT-base
inference: false
---

# Swedish grammaticality - frozen ModernBERT with ridge head

Model ID: `pavan0202/swedish-grammaticality`. Total parameters: 149,605,633; fitted
regression parameters: 769. Base revision: 0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb.
See LICENSE, NOTICE and BASE_MODEL_CARD.md for upstream attribution.

Load with AutoTokenizer and AutoModelForSequenceClassification from this
directory or `pavan0202/swedish-grammaticality`, use eager attention,
tokenize with max_length=128, run in eval/no_grad mode and clip the single
regression logit to [0,1]. No sigmoid is used. Required CLI and complete code
are in the accompanying repository; model outputs are scores, not corrections.

Training: frozen base encoder and pretrained prediction head, mean pooling,
ridge regression on 1,257 original/changed Talbanken examples; alpha=1000 chosen
on 632 validation examples. The scaler is folded into the regression head.
Data: CC BY-SA 4.0 UD Swedish-Talbanken r2.17, synthetically changed; all source
and corruption provenance is in the dataset card. Dataset: [pavan0202/talbanken-synthetic-grammaticality](https://huggingface.co/datasets/pavan0202/talbanken-synthetic-grammaticality).

Synthetic held-out: AUC 0.678, Spearman 0.248,
MAE 0.299 on 5273 examples. Starter: AUC 0.689,
Spearman 0.400 on 120 examples. These are local measurements,
not the organizer's hidden-test performance. See the report for speed/hardware.

Intended use: research prototype for a thesis selection assignment. Binary
automatic labels do not calibrate the scalar to human grammaticality. It can
miss agreement/word-order errors and score unfamiliar but grammatical text
poorly. It truncates long texts, may fail outside Swedish, and has no native
speaker validation. Pretraining overlap with public UD text was not audited.
No consequential language assessment is supported. Codex assisted in creating
and evaluating this project; applicant review remains necessary.
