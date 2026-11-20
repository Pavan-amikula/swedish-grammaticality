---
license: cc-by-sa-4.0
language:
- sv
tags:
- pytorch_model_hub_mixin
- grammaticality
---
# Count language-model baseline

Baseline included with the source code, not the selected submission model. New project
implementation: MIT; derived count tables and data: CC BY-SA 4.0, attribution
in data/README.md and this directory's LICENSE. Import GrammaticalityModel
from the repository's model.py and call from_pretrained("model"). This is a
custom Hugging Face PyTorchModelHubMixin artifact, not an AutoModel checkpoint.
config.json contains count tables and selected feature indices;
model.safetensors stores the linear head and scaler buffers.

18 features summarize word 1-3-gram and character 1-5-gram surprisal and OOV
rates. Counts come from a disjoint clean-text partition; ridge fits the head
on synthetic examples. 19 head parameters and
170,163 count entries. Test AUC:
0.573; limited performance. Empty-text output
has no linguistic meaning. No native-speaker judgments or calibration.
