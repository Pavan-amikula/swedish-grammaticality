# Swedish grammaticality scoring with correction pairs

Pavan Kumar Goud Amikula | MSc Computer Science, Blekinge Institute of Technology | 21 November 2026

## Approach

The selected model uses paired fine-tuning of the last six blocks of Scandinavian ModernBERT-base [1], with masked mean pooling and a two-class prediction head. Its score is P(correct), a scalar in [0,1]. The checkpoint contains 149,606,402 parameters, below the assignment's 1.3B limit [2], and is saved in native Transformers format. A scalar temperature, fitted on reserved calibration families and folded into the final weights, adjusts confidence without changing ranking.

Partial fine-tuning keeps CPU training manageable while allowing grammaticality supervision to change contextual representations. The paired objective combines classification loss with a ranking loss favoring the corrected member of each pair. Earlier contextual blocks and embeddings stay fixed. This is a correctness-confidence score, not a human-validated measure of error severity.

## Data and provenance

DaLAJ-GED-SuperLim [3], released under CC BY 4.0, supplies learner-error sentences, expert corrections and correct-labeled sentences. An error/correction pair is retained only when the correction, with source-compatible spacing, matches a released positive in the same upstream split. The actual released positive text is used; no new linguistic judgments are added. Source revisions and hashes are pinned. Learner demographics are omitted.

A graph connects repeated sentences and correction variants using normalized, case-folded family keys. Official test families take precedence over development and training. One error per family is chosen deterministically. There are 4,000 training, 500 calibration, 629 validation and 590 test pairs. Another 400 training, 200 validation and 200 test pairs come from licensed Talbanken text with automatic edits [4]. Text/family overlap checks passed. Writer IDs are unavailable, so writer-level overlap cannot be verified.

## Experiments and selection

The initial count and frozen-regression baselines were weak. A preliminary MiniLM semantic-encoder configuration did not yield useful validation results; it used an earlier curation rule and is not a matched final-data comparison. On the final data, a fitted frozen head is compared with two epochs of fine-tuning the last six blocks. The fixed objective is 0.75 learner-validation AUC + 0.25 synthetic-validation AUC. Replacement also requires learner AUC to improve by at least 0.02 without a bridge-AUC drop exceeding 0.02.

The frozen head reached learner-validation AUC 0.632; paired adaptation reached 0.715. AdamW uses encoder/head learning rates 3e-5/3e-4, batch 16, weight decay .01, 6% warmup and 128 tokens. Seed 42 and full reproduction commands are recorded. Checkpoint selection uses validation only, calibration separate families, and tests follow selection. Training/export elapsed 78070 seconds including substantial machine pauses; this is not uninterrupted CPU time.

## Measured results

All reported results are actual local measurements on an Intel i7-1165G7, 16 GB RAM, Windows and CPU-only PyTorch, using four inference threads. The new and previous models are compared on the same held-out inputs. The source-confirmed paired subset is not the full DaLAJ benchmark.

| Evaluation | n | Previous AUC | Selected AUC | Selected Spearman |
|---|---:|---:|---:|---:|
| Source-confirmed learner pairs | 1180 | 0.606 | 0.731 | 0.400 |
| Synthetic bridge holdout | 400 | 0.681 | 0.750 | 0.433 |
| Official starter | 120 | 0.689 | 0.830 | 0.635 |

The learner-test AUC improvement has a family-bootstrap 95% interval of [0.106, 0.144] (1,000 resamples, seed 42). Strict correct-over-error pair accuracy is 0.847, versus 0.742 previously. The official evaluator passed the required JSONL interface and measured 21.1 seconds, or 341 samples/min, including imports and model loading. Its timing output is rounded. Hidden organizer results and speed on their hardware are unknown.

For context, the assignment's published baseline reports Spearman 0.528 and AUC 0.783 on its starter test [2]. That model was not retrained here and uses different training data. No claim of universal best performance or guaranteed selection is made.

## Limitations and actual error example

The data is partially normalized, single-error learner language. Requiring a confirmed positive excludes some complex corrections and changes the evaluation distribution. Binary labels do not validate human severity scores. Lexical and punctuation decisions may be context-dependent. Near duplicates, writer style and pretraining overlap cannot be fully excluded. Inputs are truncated at 128 tokens. Rare words, pseudonyms and other domains remain risks; no independent native-speaker review is claimed.

An actual held-out pair (source category S): corrected 'Därför började jag min i kurs tyska och jag gick på olika fester.' scores 0.047; error version 'Därför började jag min kurs tyska och jag gick på olika fester.' scores 0.514. These labels come from the released dataset, not a new linguistic assessment. More error slices and raw predictions are included in the repository.

## Availability and references

Code: github.com/Pavan-amikula/swedish-grammaticality. Model: pavan0202/swedish-grammaticality. Training data: pavan0202/swedish-grammaticality-training. Exact revisions and verification are recorded in the repository.

[1] AI-Sweden-Models. ModernBERT-base model card, Apache 2.0. huggingface.co/AI-Sweden-Models/ModernBERT-base. Revision 0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb.

[2] AI Sweden. Swedish Grammaticality Scoring: Master's Thesis Assignment. nicher92.github.io/swedish_correctness_classifier/ (accessed 21 November 2026).

[3] Volodina, Ali Mohammed, Berdicevskis, Bouma and Öhman. 2023. DaLAJ-GED. NLP4CALL, pp. 94-101. doi.org/10.23695/kxvz-tx42; aclanthology.org/2023.nlp4call-1.11/.

[4] Nivre, Smith, Norrman and contributors. UD Swedish-Talbanken r2.17, CC BY-SA 4.0. github.com/UniversalDependencies/UD_Swedish-Talbanken.
