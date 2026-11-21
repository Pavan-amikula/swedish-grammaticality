---
language: sv
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: AI-Sweden-Models/ModernBERT-base
datasets:
- pavan0202/swedish-grammaticality-training
metrics:
- roc_auc
- spearmanr
tags:
- grammaticality
- swedish
---
# Swedish grammaticality scoring

Author: **Pavan Kumar Goud Amikula**, MSc Computer Science student,
Blekinge Institute of Technology.

Native `ModernBertForSequenceClassification`, 149,606,402 parameters.
The score is `softmax(logits)[correct]` in [0,1], with `correct=1` and
`incorrect=0`. Mean pooling and a 128-token input limit are used. Temperature
3.0 is already folded into the head weights: do not apply it again.
This estimates binary correctness confidence, not a validated human severity score.

## Training and reproducibility

The fixed base revision is `0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb`.
The head was fitted over frozen features, then the last six contextual blocks
were adapted for two epochs with equal-weight classification and pair-ranking
losses. Earlier blocks/embeddings stayed fixed. Seed 42; CPU four threads;
AdamW, encoder LR 3e-5/head LR 3e-4, batch 16, weight decay .01.
Selected stage: `fine_tune_last_6`, epoch 2.
Selection uses 0.75 learner validation AUC + 0.25 synthetic bridge AUC.
Calibration uses a separate 500-family set. No held-out test scores selected
the model. The starter test had been inspected in the initial project and
was used only as a final diagnostic for this replacement.

Source-confirmed DaLAJ pairs: 8,000 training rows, 1,000 calibration rows,
1,258 validation rows, 1,180 test rows. Synthetic bridge: 800 training,
400 validation and 400 test rows. Corrected sentences and error variants are
grouped across splits. DaLAJ is CC BY 4.0; bridge data is CC BY-SA 4.0.
The dataset card gives sources, filtering, exact revisions and license texts.
Required model and data attributions are retained in NOTICE and BASE_MODEL_CARD.md.

## Actual measured evaluation

| Test | n | ROC AUC | Spearman |
|---|---:|---:|---:|
| Source-confirmed learner pairs | 1180 | 0.730689 | 0.399566 |
| Official starter | 120 | 0.830278 | 0.634890 |

The paired learner subset is not the full DaLAJ benchmark. These measurements
must not be compared directly with published full-DaLAJ results. On the same
learner test the previous project model had AUC 0.605629.
The official evaluator measured 21.1 seconds / 341
samples per minute on an Intel i7-1165G7, 16 GB RAM, CPU-only Windows run;
time includes imports and model loading and is rounded by that evaluator.
Hidden organizer performance and speed on other hardware are unknown.

## Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = 'pavan0202/swedish-grammaticality'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, attn_implementation='eager').eval()
x = tok(['Hon läser en bok.'], truncation=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    scores = model(**x).logits.softmax(-1)[:,model.config.label2id['correct']]
```

For the assignment interface and an immutable model revision use the code:
https://github.com/Pavan-amikula/swedish-grammaticality

## Limitations

Partially normalized single-error learner sentences differ from whole essays.
Filtering to source-confirmed corrections excludes complex cases. Scores may
reflect vocabulary, punctuation and domain as well as grammar. Binary labels
do not establish severity calibration. Inputs after 128 tokens are truncated.
No independent native-speaker review was performed for this project. Writer
IDs and base pretraining overlap cannot be audited; grouping only addresses
observable sentence/correction families. Not suitable for high-stakes grading.
