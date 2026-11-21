---
license:
- cc-by-4.0
- cc-by-sa-4.0
language:
- sv
task_categories:
- text-classification
pretty_name: Swedish grammaticality training - correction pairs and synthetic bridge
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: calibration
    path: calibration.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
- config_name: synthetic_bridge
  data_files:
  - split: train
    path: bridge_train.jsonl
  - split: validation
    path: bridge_validation.jsonl
  - split: test
    path: bridge_test.jsonl
---

# Swedish grammaticality training data

Dataset ID: `pavan0202/swedish-grammaticality-training`.

This prepared collection contains source-correction pairs from DaLAJ-GED and
a small synthetic bridge from Swedish-Talbanken. Publication details and exact
model usage are recorded in the accompanying repository's publication record.

## Sources and licenses

| Files | Source | License |
|---|---|---|
| train, calibration, validation, test | DaLAJ-GED-SuperLim 2.0 | CC BY 4.0 |
| bridge_train, bridge_validation, bridge_test | UD Swedish-Talbanken plus automatic edits | CC BY-SA 4.0 |

The two licenses apply to their respective files; they are not alternative
licenses for the entire collection. Full notices are included separately.

DaLAJ attribution: Elena Volodina, Yousuf Ali Mohammed, Aleksandrs Berdicevskis,
Gerlof Bouma and Joey Öhman. 2023. *DaLAJ-GED - a dataset for Grammatical Error
Detection tasks on Swedish*. NLP4CALL 2023, pp. 94-101.
https://aclanthology.org/2023.nlp4call-1.11/
Resource documentation: https://sprakbanken.se/en/resources/dalaj-ged-superlim
DOI: https://doi.org/10.23695/kxvz-tx42

Downloaded from `sprakbanken-se/superlim-2`, revision
`e70516a7f2397b68e4f2bef3cb984944b1c670eb`. The upstream files contain
partially normalized learner-written sentences with one remaining annotated
error, expert corrections, and correctly written coursebook text. This subset
uses learner-error/correction pairs rather than contrasting learner text with
coursebook text, reducing a possible source/style shortcut.

Talbanken attribution: the original Talbanken creators and UD converters
Joakim Nivre, Aaron Smith, Victor Norrman and contributors. Source revision:
`463c1eb7176c19e8d7cc54e5f5b0f21d6d6ce44d`. Full upstream acknowledgment is
included in the CC BY-SA license notice and the parent dataset card.
https://github.com/UniversalDependencies/UD_Swedish-Talbanken
Parent bridge data: https://huggingface.co/datasets/pavan0202/talbanken-synthetic-grammaticality

## Preparation and grouping

For an incorrect DaLAJ sentence, the annotated error span is replaced with
the released correction to identify a candidate corrected counterpart. The
span must exactly match the released error text. The pair is retained only
when a matching sentence is explicitly labeled correct in the same upstream
split; the actual published positive text is used. No new human linguistic
judgments or graded severity annotations were added. Texts outside 3-80 words,
unchanged normalized pairs, and conflicting labels are excluded.
For omitted words, candidate surrounding spaces are considered only to locate
the already published positive sentence; no grammaticality is guessed.

A connected-component graph links every observed sentence with its corrected
counterpart. NFC/whitespace-normalized, case-folded family keys keep repeated
variants and capitalization variants together. Official test groups take
precedence over development groups, which take precedence over training;
members of a family in an earlier split are excluded. One error variant per
family is selected by a deterministic seed-42 hash. Source text is not
lowercased for model input. Learner demographics and essay metadata are omitted.

After reserving 500 official-training families for calibration, 4,000 other
training families are used for learning. Validation uses 629 development
families; test uses 590 test families. Every family contributes one corrected
and one incorrect example. The bridge similarly pairs each original with one
deterministically selected automatic edit. Direct text/family overlaps across
all partitions are checked. SHA-256 source hashes are in provenance.json.

| Split | Rows | Families |
|---|---:|---:|
| train | 8,000 | 4,000 |
| calibration | 1,000 | 500 |
| validation | 1,258 | 629 |
| test | 1,180 | 590 |
| bridge_train | 800 | 400 |
| bridge_validation | 400 | 200 |
| bridge_test | 400 | 200 |

The test labels are not used to select checkpoints or calibration parameters.
The validation objective is fixed in results/grammar_protocol.json. A separate
calibration split fits a scalar temperature after checkpoint selection.

## Fields and limitations

`id`: example identifier; `group`: correction-family identifier; `text`: Swedish
input; `score`: 1 for corrected/original text and 0 for error/changed text;
`label`: good/poor; `source`: dalaj-ged or talbanken-synthetic; `source_split`:
upstream split; `error_type`: released coarse error category or bridge edit.

DaLAJ categories include morphological, syntactic, lexical, orthographic and
punctuation errors. The data is not an untouched sample of learner essays:
sentences were partially normalized by the source creators, and released
single-error variants are used here. The source describes its test as manually
proofread. No independent native-speaker review was performed for this project.

Labels are binary acceptability targets, not human scores of error severity.
Some automatic bridge edits remain grammatical. Essay/author identifiers are
unavailable, so writer-level overlap cannot be checked. Near duplicates can
remain despite direct-family checks. Pretrained models may have encountered
public corpus text. Retained pseudonyms and corpus style may affect predictions.
The positive-confirmation screen excludes many complex corrections. These
paired subsets differ from the full official DaLAJ benchmark; reported scores
must not be compared directly with results on the full benchmark.
