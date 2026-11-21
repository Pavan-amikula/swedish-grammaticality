# Swedish grammaticality scoring

Author: **Pavan Kumar Goud Amikula**, MSc Computer Science student,
Blekinge Institute of Technology.

A CPU-trained Swedish correctness scorer with **149,606,402
parameters**, below the 1.3B limit. It returns the probability of the `correct`
class, between 0 and 1. It uses paired fine-tuning of the last six blocks. The native Transformers checkpoint
has two logits; the temperature calibration is already folded into its weights.
The score is not a human-validated severity rating.

- Code: https://github.com/Pavan-amikula/swedish-grammaticality
- Model: https://huggingface.co/pavan0202/swedish-grammaticality
- Current training data: https://huggingface.co/datasets/pavan0202/swedish-grammaticality-training
- Original synthetic baseline data: https://huggingface.co/datasets/pavan0202/talbanken-synthetic-grammaticality
- [Paper](report.pdf), [editable paper](report.md), [requirement check](REQUIREMENTS_CHECK.md)
- [Submission email](SUBMISSION_EMAIL.md), [publication record](PUBLISHING.md)
- [Explanation guide](EXPLAINING_THE_PROJECT.md)

## Reproduce the selected model

Tested with Python 3.13.4, CPU-only PyTorch, Windows, Intel i7-1165G7 and 16 GB
RAM. Use this project directory as the working directory. No paid compute,
API inference service or GPU is needed. The initial base-model download is
about 600 MB. Large weight files are on Hugging Face rather than in GitHub.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
python prepare_grammar_data.py
python train_grammar.py
python -m unittest discover -s tests -v
python predict.py --input test.jsonl --output scores.jsonl
```

This reproduces data preparation and downstream training from the fixed base
checkpoint, not the base model's large-scale pretraining. Direct dependencies,
base-model/data revisions, source hashes, seed 42, CPU thread count and training
hyperparameters are recorded. Floating-point differences across platforms may
remain. See results/grammar_protocol.json and results/grammar_training.json.
Training and export elapsed 78070.4 seconds in this run,
including substantial machine pauses; this is not uninterrupted CPU time.

The preprocessing retains only released error sentences whose source-provided
correction matches a sentence explicitly labeled correct in the same upstream
split. Spaces around omitted tokens are resolved only by matching a released
positive. A connected-component graph keeps corrected sentences and their
variants together; cross-split families are excluded from earlier splits.
Source text is not lowercased for input. Demographics and essay metadata are
not used. There are 4,000 training pairs, 500 calibration pairs, 629 validation
pairs and 590 test pairs, plus a small, separately split synthetic bridge.
Full details and licenses are in data/grammar/README.md.

The fixed validation objective is 0.75 learner AUC + 0.25 bridge AUC. The
previous submission is replaced only when learner validation AUC increases by
at least 0.02, bridge AUC drops by no more than 0.02, and the objective improves.
Tests are evaluated after model selection. Temperature is fitted on separate
calibration families. Calibration does not change ranking.

## Required prediction interface

```bash
python predict.py --input test.jsonl --output scores.jsonl
```

Input: one JSON object per line containing `id` and string `text`. Output:
exactly `id` and a finite float `score` in [0,1], one row per input. IDs and
input order are preserved, including duplicate IDs. Empty files and Unicode
are supported. Empty-text scores have no linguistic interpretation.

The default checkpoint is grammar_model. If its weights are absent in a fresh
checkout, predict.py downloads the immutable Hub revision in selection.json.
`--model LOCAL_DIRECTORY_OR_HUB_ID` overrides it. The selected model uses
128 tokenizer tokens; later text is truncated. Inputs are length-bucketed
within bounded batches to reduce padding, then restored to input order.

Native loading (calibration is already applied; do not apply temperature twice):

```python
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
settings = json.loads(Path("selection.json").read_text())
tok = AutoTokenizer.from_pretrained(settings["hub_model_id"], revision=settings["hub_revision"])
model = AutoModelForSequenceClassification.from_pretrained(
    settings["hub_model_id"], revision=settings["hub_revision"],
    attn_implementation="eager").eval()
batch = tok(["Hon läser en bok."], truncation=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    score = model(**batch).logits.softmax(-1)[:, model.config.label2id["correct"]]
```

## Measured results

| Evaluation | n | Previous AUC | Selected AUC | Selected Spearman |
|---|---:|---:|---:|---:|
| Source-confirmed learner pairs | 1180 | 0.606 | 0.731 | 0.400 |
| Synthetic bridge holdout | 400 | 0.681 | 0.750 | 0.433 |
| Official starter | 120 | 0.689 | 0.830 | 0.635 |

On the learner holdout, strict pair ranking accuracy is
0.847, versus
0.742 previously. The group-bootstrap
95% interval for the AUC difference is
[0.106, 0.144].
This paired subset differs from the full DaLAJ benchmark, so these scores
must not be compared directly with published full-DaLAJ scores.

The official starter evaluator passed. Its measured whole-prediction time was
21.1 seconds (341 samples/min), including model
loading and imports. Timing is from the upstream evaluator's rounded output.
Organizer hardware/hidden-test performance is unknown. The assignment's
published starter reference is Spearman 0.528 and AUC 0.783; it is not a
baseline retrained here and uses a different model/training set.

The starter test had already been inspected during the initial prototype. It
was not used to select this replacement. The independent learner test was
scored only after validation selection. See results/promotion_decision.json,
results/grammar_error_analysis.json and the raw prediction files.

## Experiments and files

- prepare_grammar_data.py: fixed, source-confirmed pair preparation.
- train_grammar.py: frozen-head fit, paired contextual fine-tuning, calibration.
- evaluate_grammar.py and decide_promotion.py: validation comparison and guard.
- analyze_grammar_results.py: fixed holdout comparison and family bootstrap.
- experiments/train_minilm.py: preliminary semantic-encoder experiment. It used
  the earlier pair-curation rule and is not a matched final-data ablation.
  Reproduce it separately with `prepare_grammar_data.py --allow-unconfirmed-corrections`.
  Do not overwrite the selected preparation when running it.
- train.py/model/: original count baseline; train_encoder.py/encoder_model/:
  original synthetic-data regression baseline. Their results are retained.

## Limitations and licenses

The corpus uses partially normalized learner sentences and single-error
variants, not complete untouched essays. Source-positive matching excludes
many complex corrections. Binary correctness probabilities do not calibrate
human error severity. Lexical/punctuation labels can depend on context.
Writer/essay IDs are unavailable; direct families/text are separated, but
writer-level and pretraining overlap cannot be audited. Rare vocabulary,
pseudonyms, long text and domain shift can affect scores. No independent
native-speaker review was performed for this project.

Code: MIT. Base/adapted model: Apache 2.0. DaLAJ files: CC BY 4.0; synthetic
bridge: CC BY-SA 4.0. Required original notices and citations are retained.
The submission email was sent to niclas.hertzberg@ai.se with report.pdf attached; Gmail confirmed sending.
