---
library_name: transformers
license: apache-2.0
language:
- sv
- 'no'
- da
- is
tags:
- masked-lm
- fill-mask
- long-context
- modernbert
pipeline_tag: fill-mask
inference: false
base_model:
- answerdotai/ModernBERT-base
---
## Overview  
This checkpoint continues the pre-training of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on Scandinavian text, extending the model’s knowledge with ~1.2 trillion additional masked-language-model (MLM) tokens drawn from [The Nordic Pile](https://arxiv.org/pdf/2303.17183) and [SWEb](https://arxiv.org/pdf/2410.04456) while preserving the original 8k token context window. 

This is a **research artefact** and is only intended for **research purposes**.

Our tokenizer is trained from scratch on a subset of 11 985 103 472 tokens. 

The training is done in one stage with 8192 tokens per sample for the whole run.
## Data Sources  
| Corpus | Size | Selected Languages | Highlights |
|---|---|---|---|
| **The Nordic Pile** | 1.2 TB raw text | sv, no, da, is | Nine diverse categories (CC, Wikipedia, Books, Code, etc.), filtered and deduplicated for high quality |
| **SWEb** | 1 T+ tokens (~3.6 TB) | sv, no, da, is | 98 Common-Crawl snapshots with model-based HTML extraction; 1.2 B documents |
## Training Setup  
| Setting | Value |
|---|---|
| Parameters | 150 M |
| Context length | 8 192 tokens (RoPE + local-global attention) |
| Tokens processed | 1.20 × 10<sup>12</sup> |
| Tokens per batch | 1 572 864 |
| Global batch | 192 sequences (micro-batch = 3) |
| Optimizer & schedule | Decoupled StableAdamW, lr 2 e-4, cosine decay (1 % warm-up) |
| Precision | AMP-bf16 |
| Hardware | 8 nodes × 8 AMD MI250X GPUs (64 GPUs) on the EuroHPC **LUMI-G** system |

See training details [here](https://github.com/timpal0l/ModernBERT/blob/main/training/trainer_lumi.yaml)
## Training Stats
```python
[token=1198511677292/1198510347252]:
  Train time/batch: 873585
  Train time/sample: 167728320
  Train time/batch_in_epoch: 3558
  Train time/sample_in_epoch: 683136
  Train time/token: 1198510256276
  Train time/token_in_epoch: 4882888303
  Train trainer/device_train_microbatch_size: 3
  Train loss/train/total: 0.9966
  Train throughput/batches_per_sec: 1.3117
  Train throughput/samples_per_sec: 251.8442
  Train throughput/device/batches_per_sec: 0.0205
  Train throughput/device/samples_per_sec: 3.9351
  Train throughput/tokens_per_sec: 1804244.5198
  Train throughput/device/tokens_per_sec: 28191.3206
  Train time/train: 184.5555
  Train time/val: 0.0000
  Train time/total: 184.5555
  Train lr-StableAdamW/group0: 0.0000
  Train lr-StableAdamW/group1: 0.0000
```
## Intended Use  
This is a **research artefact** and is only intended for **research purposes**.
* Fill-mask inference, embedding extraction and fine-tuning for Scandinavian downstream NLP tasks (classification, NER, QA, etc.).  
* Drop-in replacement for BERT-style encoders (omit `token_type_ids`).
## Fill-mask
```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='AI-Sweden-Models/ModernBERT-base')
unmasker("Huvudstaden i Sverige är [MASK].")
```
```python
[{'score': 0.0629318505525589,
  'token': 2961,
  'token_str': ' Stockholm',
  'sequence': 'Huvudstaden i Sverige är  Stockholm.'},
 {'score': 0.03635135293006897,
  'token': 49763,
  'token_str': 'awesome',
  'sequence': 'Huvudstaden i Sverige är awesome.'},
 {'score': 0.03006783314049244,
  'token': 751,
  'token_str': ' stor',
  'sequence': 'Huvudstaden i Sverige är  stor.'},
 {'score': 0.029827557504177094,
  'token': 71,
  'token_str': 'a',
  'sequence': 'Huvudstaden i Sverige är a.'},
 {'score': 0.019739385694265366,
  'token': 79,
  'token_str': 'i',
  'sequence': 'Huvudstaden i Sverige är i.'}]
```
## Limitations & Biases  
* Web corpora can contain noise, stereotypes and sensitive content despite filtering.   
* RoPE extrapolation beyond 8 k tokens is untested and may degrade.
## Code to reproduce
* [Training](https://github.com/timpal0l/ModernBERT/tree/main/training)
* [Data Processing](https://github.com/timpal0l/ModernBERT/tree/main/tokenizer)