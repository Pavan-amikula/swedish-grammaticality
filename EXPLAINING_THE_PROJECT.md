# Explaining the project

Author: Pavan Kumar Goud Amikula

The task is to give Swedish text one score between zero and one. Higher means
the model is more confident the sentence is correct. This is a binary
classification probability; it does not measure how serious a human thinks
an error is.

The starting model, Scandinavian ModernBERT-base, already represents language.
The project uses mean pooling to summarize its token representations and a
two-class head to predict correct/incorrect. The first experiment fitted a
head while the encoder stayed fixed. The stronger experiment also trained
the final six of 22 contextual blocks; embeddings and the first 16 stayed
fixed. This limits CPU cost while allowing representations to adapt.

Each training family contains an error and its corrected sentence. Half the
loss teaches their binary labels. The other half penalizes ranking the error
above the correction. Corrections are accepted only when they match an actual
sentence labeled correct in the released source; no new Swedish judgments
were invented. A small synthetic bridge adds automatically edited corpus text.

Repeated sentences and correction variants are connected before splitting.
Test families take priority over development and training. The model is chosen
using validation only, then a temperature adjusts confidence on a separate
calibration partition. It does not change ranking. Finally the selected model
is measured once on the independent learner holdout and on the starter set.

The exact results, including failures and confidence intervals, are in the
paper and results/grammar_error_analysis.json. The official starter reference
uses different training data and was not retrained here. Never describe the
project as the best possible model or promise hidden-test performance.

Main limitations: single-error normalized sentences, unavailable writer IDs,
unknown pretraining overlap, a 128-token limit, and imperfect transfer to other
domains. There was no independent native-speaker review. You should be able
to explain these choices and the measured numbers in your own words before
submitting.

Reproduction means rebuilding this downstream model from a pinned public
pretrained checkpoint, not reproducing its original large-scale pretraining.
See README.md for exact commands, dependency versions and artifacts.
