---
license: cc-by-sa-4.0
language:
- sv
task_categories:
- text-classification
pretty_name: Talbanken synthetic grammaticality proxy
configs:
- config_name: default
  data_files:
  - split: train
    path: calibration.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
- config_name: language_model
  data_files:
  - split: train
    path: lm_train.jsonl
- config_name: encoder
  data_files:
  - split: train
    path: encoder/train.jsonl
  - split: validation
    path: encoder/validation.jsonl
---

# Talbanken synthetic grammaticality proxy

Dataset ID: `pavan0202/talbanken-synthetic-grammaticality`.

Source: Universal Dependencies Swedish-Talbanken r2.17, commit
`463c1eb7176c19e8d7cc54e5f5b0f21d6d6ce44d`.
https://github.com/UniversalDependencies/UD_Swedish-Talbanken/tree/463c1eb7176c19e8d7cc54e5f5b0f21d6d6ce44d
Attribution: the Talbanken creators at Lund University and the UD converters
Joakim Nivre, Aaron Smith, Victor Norrman and contributors acknowledged in
the upstream README and the included LICENSE.txt. The original prose comes
from Swedish informative writing from the 1970s. It is not learner writing.

License: CC BY-SA 4.0, including the generated text adaptations. Changes:
filtered to 6-60 whitespace-delimited words, lowercased/whitespace-normalized
for deduplication only, and paired with programmatically changed variants.
The distributed text retains original casing. Source annotation is not used
for training. Raw CoNLL-U files are reproducibly downloaded rather than
duplicated in this upload. SHA-256 hashes and counts are in provenance.json.

The `language_model` config's train split (`lm_train.jsonl`) has 2,384 unchanged
sentences for count statistics only. The default config contains:
`train`: 6,242 original/changed examples for count-model score fitting.
`validation`: 2,253 examples for selection; `test`: 5,273 held-out examples.
The encoder config uses deterministic subsets: 256 train sentence groups
and 128 validation sentence groups. Actual row counts are recorded in results.
These are subsets of the default config; do not mix configs as independent data.

Official UD train/dev/test document boundaries are respected. When a document
or normalized source sentence occurs across official splits, training yields
to validation/test. Training documents are further divided 60/40 by seeded
shuffle into language-model learning and score fitting. All variants of a
sentence stay together. Cross-partition synthetic exact collisions cause the
later sentence group to be excluded. Source IDs and document IDs permit audit.

Fields: `id`, `text`, `source_id`, `document`, `score`, `split`; derived examples
also have `group`, `label`, `corruption`. Scores are binary proxies: original
= 1, changed = 0. Changes swap adjacent interior whitespace tokens, remove a
selected function word, duplicate an interior word, or remove the last letter
of an alphabetic word of at least five characters. They are inspired by the
automatic corruption approach used by ScaLA, but this data was independently
generated and is not a copy of ScaLA. Reference:
https://huggingface.co/datasets/alexandrainst/scala and Nielsen (2023),
ScandEval, https://aclanthology.org/2023.nodalida-1.20/ .

Important: changed text can remain grammatical and original text can contain
fragments or dated usage. Scores are NOT human grammaticality judgments or
calibrated severity labels. Word ending changes may affect spelling rather
than grammar. Class imbalance is roughly four changed variants per original.
Near-duplicates and common phrases can remain despite exact deduplication.
No human annotation or native-speaker validation was performed. No private
emails or applicant information was used as training text.

The unmodified upstream README is included as SOURCE_README.md for full
provenance and acknowledgments. The starter test is not redistributed here.
