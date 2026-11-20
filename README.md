# Swedish grammaticality scoring

Prepared for Pavan Kumar Goud Amikula, MSc Computer Science, Blekinge Institute
of Technology. Public repositories (final submission email remains unsent):

- GitHub: [Pavan-amikula/swedish-grammaticality](https://github.com/Pavan-amikula/swedish-grammaticality)
- Model: [pavan0202/swedish-grammaticality](https://huggingface.co/pavan0202/swedish-grammaticality)
- Dataset: [pavan0202/talbanken-synthetic-grammaticality](https://huggingface.co/datasets/pavan0202/talbanken-synthetic-grammaticality)

Immutable Hub revisions and fresh-download verification are recorded in
results/publication.json. The exact published data matches local files; the
Hub uses flat encoder_train.jsonl/encoder_validation.jsonl filenames, while
this repository keeps those files under data/encoder/. Their contents match.

The complete local folder includes the saved encoder weights (about 598 MB).
The source archive and GitHub exclude that large weight file; use the training
commands below, or the automatic verified Hub download, to obtain it.

A CPU prototype that outputs a score in [0,1]. The selected model is a frozen
Scandinavian ModernBERT-base encoder with mean pooling and a ridge-fitted
linear regression head: **149,605,633 total parameters**, 769 fitted
head parameters. It is saved in standard Transformers/Hugging Face format.
The fitted scalar is clipped to [0,1]; it is an acceptability proxy rather
than a calibrated human judgment. The hidden evaluation set is unavailable.

## Read first

- [Official assignment](https://nicher92.github.io/swedish_correctness_classifier/#task).
- [Two-page paper](report.pdf), editable [paper text](report.md).
- [Data card and licensing](data/README.md), [publication plan](PUBLISHING.md),
  and [unsent submission email draft](SUBMISSION_EMAIL.md).
- [What to understand before submission](EXPLAINING_THE_PROJECT.md).

## Reproduce from scratch

This reproduces the project's data preparation and scoring-head fit from the
fixed pretrained checkpoint; it does not repeat large-scale encoder pretraining.

Tested on Windows, Python 3.13.4, Intel i7-1165G7 (4 cores/8 logical CPUs),
16 GB RAM, CPU-only PyTorch. Run in this project directory. A virtual
environment is recommended. No GPU, paid services, API keys or private data
are used. Initial encoder download is approximately 600 MB. Training itself
took 227.6 seconds in this run; runtime varies.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
python train.py
python train_encoder.py
python -m unittest discover -s tests -v
python benchmark.py --name synthetic
python analyze_errors.py
```

Direct dependency versions are pinned. The observed package-version snapshot
is results/environment-full.txt; it describes this machine and is not a
portable installation lock. Source data/model revisions, random seeds and
source SHA-256 hashes are pinned in code and provenance.json. CPU eager
attention is explicit; seeds and deterministic PyTorch algorithms are set.
Floating-point differences across operating systems/BLAS versions may remain.
Data generation uses canonical LF line endings and is byte reproducible.
Test scores never select hyperparameters. A 16-example eager/SDPA parity check
passed at tolerance 1e-4; its single-batch timings did not justify changing
attention implementations. The final model uses eager attention.

The count baseline uses a separate clean-text partition to avoid memorizing
the same sentences it scores during head training. train_encoder.py uses
1,257 examples from 256 sentence groups for its head, and 632 examples from
128 disjoint validation groups. Source document boundaries are kept apart.
It compares ridge strengths 10/100/1000 and selects 1000 by validation AUC.
selection.json selects the encoder over the count model on the same validation
subset (0.676 versus 0.578).

## Predict using the required interface

```bash
python predict.py --input test.jsonl --output scores.jsonl
```

Input: one JSON object per line, e.g. `{"id": 1, "text": "Hon läser en bok."}`.
Output: `{"id": 1, "score": ...}`, with a finite scalar between 0 and 1.
IDs and input order are preserved. Unicode, empty input files and empty texts
are supported. Empty-text scores are numerically valid but have no linguistic
interpretation. Long texts are truncated to 128 tokenizer tokens; this can
miss errors near the end. Inference is CPU-only, 16 examples per model batch.

The supplied selected directory is encoder_model; both it and model (the small
count baseline) can be passed explicitly with `--model LOCAL_DIRECTORY`.
The default is read from selection.json, independent of the working directory.
The verified model ID and immutable revision are recorded in selection.json,
so fresh clones download the checkpoint when local weights are absent. You
can also pass a model ID explicitly with `--model`.

Standard selected-model loading (model outputs raw regression logits; clip):

```python
import torch, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
selection = json.loads(Path("selection.json").read_text())
tok = AutoTokenizer.from_pretrained("pavan0202/swedish-grammaticality", revision=selection["hub_revision"])
model = AutoModelForSequenceClassification.from_pretrained(
    "pavan0202/swedish-grammaticality", revision=selection["hub_revision"],
    attn_implementation="eager").eval()
batch = tok(["Hon läser en bok."], padding=True, truncation=True,
            max_length=128, return_tensors="pt")
with torch.no_grad():
    scores = model(**batch).logits.squeeze(-1).clamp(0, 1)
```

## Actual results

| Evaluation | n | Spearman | ROC-AUC | MAE | Samples/min |
|---|---:|---:|---:|---:|---:|
| Synthetic UD test | 5273 | 0.248 | 0.678 | 0.299 | 442 |
| Official starter test | 120 | 0.400 | 0.689 | 0.345 | 114 |

Whole-process time, including imports and model loading: synthetic
715.51 s; starter 62.93 s. These are single local runs,
not organizer hardware results. Benchmark logs are in results/. The unmodified
official evaluate.py was also run; see results/official_evaluator.txt. The
starter has continuous scores, while our synthetic test has binary targets;
their MAEs and Spearman correlations answer different questions.

For context, the assignment's published ModernBERT-large/SVR starter baseline
reports Spearman 0.528 and AUC 0.783 on this toy test, using its toy training
data. Those are published reference values, not a baseline reproduced here.
This prototype falls short of that reference; its data and model setup differ.
Hardware differs too, so the published speed is not a fair timing comparison.

The count baseline achieved test AUC 0.573 and
Spearman 0.102. Its MAE
0.325 should be compared with the constant predictor's
0.325; it is weak. Initial count-only test diagnostics
were inspected before the encoder experiment. The encoder's test predictions
were made only after validation selection. One correction to count-model
sentence-start normalization was made during code review; final results use
the corrected implementation. Do not describe the entire development process
as a single blind test.

To rerun the starter check without redistributing its unlicensed data/code:

```bash
git clone https://github.com/nicher92/swedish_correctness_classifier.git .cache/starter
git -C .cache/starter checkout a3f4f81b16694473f528975f20939e4ca586c88a
python .cache/starter/evaluate.py --repo . --input .cache/starter/data/test.jsonl --labels .cache/starter/data/test_labels.jsonl --output results/starter_scores.jsonl
python benchmark.py --input .cache/starter/data/test.jsonl --labels .cache/starter/data/test_labels.jsonl --name starter
```

## Limitations and assistance

Training uses licensed UD prose plus automatic edits, not Swedish learner
judgments. Deletion/swaps may preserve grammaticality; spelling errors are
mixed with grammar, and the 1:4 class ratio affects the regression scale.
No human calibration validates what a score such as 0.7 means. Domain shift,
rare grammatical vocabulary, negation, word order, agreement, long documents
and foreign-language inputs remain limitations. The pretrained encoder's
training corpus may overlap public UD text; that overlap was not audited.
Do not use this prototype for consequential language assessment.

Codex assisted with research, data-generation code, implementation, execution,
evaluation and documentation. The applicant should review and understand
the code and results before submission. This repository does not claim
independent implementation, native Swedish
expertise or human annotation. The official webpage states no specific
assistance policy; any separate institutional/email conditions still apply.

New project code: MIT. Dataset and count tables: CC BY-SA 4.0. Pretrained and
adapted encoder: Apache 2.0 with retained upstream card/license/notices. See
LICENSE, data/LICENSE.txt and encoder_model/NOTICE. Public links are above;
the final submission email has not been sent.
