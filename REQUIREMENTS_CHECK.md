# Assignment requirement check

Author: Pavan Kumar Goud Amikula. Checked 21 November 2026.

Checked against the public assignment at
https://nicher92.github.io/swedish_correctness_classifier/
and the email details supplied in this conversation. The original complete
email thread was not independently retrieved.

| Requirement | Evidence / status |
|---|---|
| Swedish score in [0,1] | Native two-class softmax; finite scores checked |
| Fewer than 1.3B parameters | 149,606,402 parameters, verified from saved weights |
| Public Hugging Face model | pavan0202/swedish-grammaticality, revision recorded in selection.json |
| Public training data | pavan0202/swedish-grammaticality-training; seven splits, sources and per-file licenses |
| Public code | https://github.com/Pavan-amikula/swedish-grammaticality |
| Fresh public checkout | 120 predictions matched; max absolute difference 0; pinned Hub download verified |
| Exact required CLI | python predict.py --input test.jsonl --output scores.jsonl |
| One id/score per input | Official evaluator and CLI edge-case tests passed |
| Reproducible preparation/training | Pinned source/base revisions, dependency versions, seed 42 and commands |
| Data preparation verification | All seven derived files reproduced byte-for-byte |
| Short paper | report.pdf, rendered and visually checked before delivery |
| Approach, data and experiments | Paper and README include paired adaptation, source confirmation, failures and selection |
| Measured results/speed/hardware | Official starter: AUC 0.830278, Spearman 0.634890; 341 samples/min; Intel i7-1165G7 CPU |
| Held-out evaluation | Learner test n=1180, AUC 0.730689; bootstrap and error analysis retained |
| Tests | All eight passed; full output in results/tests_v2.txt |
| Author | Pavan Kumar Goud Amikula; mandatory upstream credits/licenses retained |
| Email package | Correct recipient/links and paper attachment named; draft remains unsent |

The model was selected using validation only. Learner validation improved by
0.121696 AUC and synthetic bridge validation by
0.066337; all predefined promotion guards passed.
Training elapsed time includes substantial machine pauses and must not be
interpreted as uninterrupted CPU compute time. The source-confirmed paired
test is not the full DaLAJ benchmark. No independent native-speaker review,
writer-level separation audit or hidden-test result is claimed. Stronger
measured performance does not establish that this is the best possible model
or guarantee thesis selection.

Deadline from the supplied email details: 4 December 2026, 23:59 CET
(5 December 2026, 04:29 IST). Final submission email has not been sent.
