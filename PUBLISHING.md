# Publication record

Publishing was authorized by Pavan on 20 November 2026. The final submission
email remains unsent and requires separate approval.

- GitHub: https://github.com/Pavan-amikula/swedish-grammaticality
- Model: https://huggingface.co/pavan0202/swedish-grammaticality
- Dataset: https://huggingface.co/datasets/pavan0202/talbanken-synthetic-grammaticality
- Model revision: `609cbf7e1f21c3de16bad534c068f9277324a896`
- Dataset revision: `b9ad327da01ac92f19338fa83fb26f32b5d4ae02`

A fresh public download matched the model weights, configuration and tokenizer
files byte for byte. All ten published dataset files matched the upload staging
files; the Hugging Face viewer successfully loaded all six splits across three
configurations. Verification details are in results/publication.json.

The published dataset uses encoder_train.jsonl and encoder_validation.jsonl at
the root; the local repository keeps equivalent files under data/encoder/.
The JSONL contents are identical. The model's ID and immutable revision are
pinned in selection.json, allowing inference from a fresh GitHub checkout
without committing the large model weight file to GitHub.

Review README.md, report.pdf, EXPLAINING_THE_PROJECT.md, measured results,
limitations and the data provenance before submitting. The model
performs below the starter's published baseline; no performance gain is claimed.
The email draft is SUBMISSION_EMAIL.md; attach report.pdf. Do not send another
acceptance email. Recipient: niclas.hertzberg@ai.se.

Deadline from the supplied email: 4 December 2026, 23:59 CET
(5 December 2026, 04:29 IST). No final submission has been sent.
