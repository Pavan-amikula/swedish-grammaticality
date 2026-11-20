# Submission requirements check

Checked on 20 November 2026 against the complete official task at
https://nicher92.github.io/swedish_correctness_classifier/#task and the email
details supplied by Pavan. The original Gmail thread was not opened during
this check. Technical requirements are taken from the official task.

| Requirement | Status | Evidence |
|---|---|---|
| Swedish text to one scalar in [0,1] | Pass | Required CLI tested; 120 starter outputs match the verified public-checkout run exactly. Scores are finite and bounded. |
| Fewer than 1.3B parameters | Pass | Direct count of saved tensor shapes: 149,605,633. |
| Model on Hugging Face in native format | Pass | pavan0202/swedish-grammaticality; standard Transformers config/tokenizer/safetensors, successfully loaded from a public download. |
| Code on GitHub | Pass | https://github.com/Pavan-amikula/swedish-grammaticality |
| Training data on Hugging Face | Pass | pavan0202/talbanken-synthetic-grammaticality; all 10 uploaded files matched prepared files, and all 6 viewer splits loaded. |
| Seeds, pinned dependencies and reproduction commands | Pass | prepare_data.py, train.py and train_encoder.py use seed 42; requirements.txt pins direct versions; source and base-model revisions/hashes are pinned; README documents training. |
| Leakage-aware separation | Pass | Automated document/text/source-group overlap checks passed; generated copies remain with their source sentences. |
| Exact prediction command | Pass | python predict.py --input test.jsonl --output scores.jsonl; IDs/order retained, exactly id and score written. |
| Official starter evaluator | Pass | results/official_evaluator.txt: Spearman 0.400, ROC-AUC 0.689, 43.8 s, 164 samples/min, including loading. |
| Short paper, about two pages excluding references | Pass | report.pdf is 2 pages including references and lists Pavan Kumar Goud Amikula as its only author. |
| Approach, data, experiments, results/speed/hardware, limitations/examples | Pass | All five required topics appear in report.pdf; an actual misranked example is included. |
| All four email submission items | Prepared | SUBMISSION_EMAIL.md contains the GitHub URL, actual model ID, actual dataset ID and the report.pdf attachment instruction. |
| Final email sent | Pending approval | Nothing has been sent to Niclas. |

## Important limits of this check

- The organizer's hidden test and evaluation hardware are unavailable. Meeting
  submission requirements does not guarantee selection or a particular score.
- The prototype's starter results are below the task's published baseline
  (Spearman 0.528, AUC 0.783). The report states this explicitly.
- Labels are automatic binary proxies rather than human graded grammaticality.
  No native-speaker validation or human score calibration is claimed.
- Reproduction documentation and public-checkout inference were verified.
  Full encoder pretraining is not part of this project; the fixed pretrained
  checkpoint is downloaded before reproducing this project's scoring-head fit.
- Dataset and pretrained-model license notices remain in place. These are
  required third-party attributions, not additional project authors.

## Submission details from the supplied email

Recipient: niclas.hertzberg@ai.se. Deadline: 4 December 2026, 23:59 CET
(5 December 2026, 04:29 IST). The acceptance reply was already sent by Pavan;
no second acceptance reply or final submission has been sent by this project.

## Verified public revisions

Model: `609cbf7e1f21c3de16bad534c068f9277324a896`.
Dataset: `b9ad327da01ac92f19338fa83fb26f32b5d4ae02`.
The model-card change updates documentation only: saved weights, tokenizer,
configuration and measured results are unchanged.
