# Swedish grammaticality scoring on a CPU

Pavan Kumar Goud Amikula | MSc Computer Science, Blekinge Institute of Technology | 20 November 2026

## Approach

A frozen Scandinavian ModernBERT-base encoder [2] supplies contextual representations. Real token vectors are mean-pooled and passed through the pretrained prediction head. A ridge regressor predicts synthetic acceptability labels; its scaler is absorbed into a native linear regression layer. The saved Transformers checkpoint has 149,605,633 parameters, below the assignment's 1.3B limit, of which 769 regression coefficients/bias were fitted. predict.py clips raw predictions to [0,1] and implements the required JSONL interface [1].

This avoids expensive full encoder fine-tuning on a CPU and produces a standard Hugging Face artifact. A small count-language-model baseline offers a transparent comparison. Full fine-tuning, prompting and paid compute were not attempted or measured.

## Data and labels

UD Swedish-Talbanken r2.17 [3], commit 463c1eb7176c19e8d7cc54e5f5b0f21d6d6ce44d, provides informative Swedish prose. Its CC BY-SA 4.0 license, acknowledgments and upstream documentation are retained. Sentences are filtered to 6-60 whitespace words. Each unchanged sentence receives target 1; up to four generated variants receive 0: adjacent token swap, function-word deletion, word duplication and final-letter deletion. These are noisy binary proxy labels, not human severity ratings. Some edits preserve grammar; some affect spelling.

Official UD document splits are retained, cross-split source duplicates are removed, and every source sentence stays with its variants. Training documents are divided into clean count-model learning (2,384 sentences) and scoring calibration (6,242 rows). Full validation/test contain 2,253/5,273 rows. The encoder uses deterministic subsets of 256 calibration groups (1,257 rows) and 128 validation groups (632 rows). Document IDs and normalized text are checked for cross-partition overlap. Near duplicates and possible pretrained-corpus overlap remain unaudited.

## Experiments and reproducibility

The count baseline fits ridge on word/character surprisal and unknown-token features. The encoder compares ridge penalties 10, 100 and 1000; 1000 wins validation AUC. On the same 632 validation rows, encoder AUC is 0.676, versus 0.578 for the count model. Selection precedes the encoder test evaluation. Initial count-only test diagnostics were inspected earlier; a count-model sentence-start normalization bug was corrected during review. Thus the entire development history is not a single blind test.

Seeds are 42, versions/revisions and source hashes are pinned, and training uses deterministic CPU eager attention. Encoder fitting and export took 227.6 seconds in this run. Scripts reproduce data, fitting, export, tests and timing. Tests cover split leakage, serialization, score bounds, Unicode/empty inputs and the command-line contract. Cross-platform floating-point variation is possible.

## Measured results

All numbers below are from actual local runs on Windows, Python 3.13.4, Intel Core i7-1165G7, 16 GB RAM, CPU-only PyTorch, with four inference threads. Speed includes imports, model loading and writing output. These single-run timings are not comparable to different organizer hardware. The hidden organizer test was unavailable.

| Evaluation | n | Spearman | ROC-AUC | MAE | Samples/min |
|---|---:|---:|---:|---:|---:|
| Synthetic UD test | 5273 | 0.248 | 0.678 | 0.299 | 442 |
| Official starter test | 120 | 0.400 | 0.689 | 0.345 | 114 |

Elapsed times: 715.51 seconds for the synthetic test and 62.93 seconds for the 120-row starter test. The unmodified official evaluator also passed (43.8 seconds, 164 samples/min on its separate run). Binary synthetic targets and continuous starter targets measure different tasks. The count baseline test AUC is 0.573, Spearman 0.102, MAE 0.325; constant-mean MAE is 0.325. The starter's published encoder/SVR baseline reports Spearman 0.528 and AUC 0.783 [1], which this prototype does not match. That reference uses different training data/model setup and was not reproduced locally; its speed is not directly comparable.

## Limitations and error analysis

The model measures resemblance to a synthetic training distribution rather than verified human grammaticality. A score such as 0.7 has no validated severity interpretation. Class imbalance, dated prose and artificial edits can bias the scale. Frozen representations and small calibration data limit sensitivity to agreement, word order and context. Inputs are truncated to 128 tokenizer tokens, potentially missing later errors. Swedish proficiency of the applicant is beginner; no native-speaker annotation or validation is claimed.

An actual held-out misranking (sv-ud-test-24): original 'Undantagandet kan även påverka pensionsrätten av anställningsinkomst.' receives 0.198; changed 'Undantagandet kan även pensionsrätten påverka av anställningsinkomst.' receives 0.452. The edit type is an adjacent-token swap; no independent human grammaticality claim is made. More actual pairs and per-edit ranking accuracy are recorded in results/error_analysis.json. A small human-rated Swedish set, with genuinely graded targets and separate calibration, is the most useful next improvement.

## Assistance and publication status

Codex assisted with source research, data-generation code, implementation, execution, evaluation and documentation. This is an AI-assisted draft for applicant review; no claim of independent implementation is made. The public task states no special assistance restriction; separate institutional/email conditions must still be respected. Code, model and data were published with Pavan's approval. Verified IDs and immutable Hub revisions are recorded in the repository README and results/publication.json. The final submission email remains unsent.

## References

[1] AI Sweden. Swedish Grammaticality Scoring: Master's Thesis Assignment. https://nicher92.github.io/swedish_correctness_classifier/ (accessed 20 November 2026). Starter commit a3f4f81b16694473f528975f20939e4ca586c88a.

[2] AI-Sweden-Models. ModernBERT-base model card (Apache 2.0), revision 0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb. https://huggingface.co/AI-Sweden-Models/ModernBERT-base .

[3] Nivre, Smith, Norrman and contributors. UD Swedish-Talbanken r2.17 (CC BY-SA 4.0); based on Talbanken (Einarsson, 1976). https://github.com/UniversalDependencies/UD_Swedish-Talbanken .
