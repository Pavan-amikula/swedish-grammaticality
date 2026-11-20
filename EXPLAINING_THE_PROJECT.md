# Preparing to explain the project

This guide summarizes the approach and questions to prepare for discussion.

1. **Goal:** predict a scalar in [0,1] for Swedish text. The task does not
   provide the hidden test set. Our labels are synthetic proxies, so the
   scalar does not have a verified human interpretation.
2. **Data:** licensed Talbanken prose provides unchanged text. Simple edits
   supply negative examples. An unchanged sentence and every edited copy stay
   in the same split. Document grouping prevents neighboring source sentences
   from leaking across training and validation/test.
3. **Model:** the encoder maps tokens to contextual vectors. We average real
   token vectors, apply its pretrained prediction head, and fit a linear ridge
   regressor. All encoder weights stay fixed. Ridge penalizes large weights;
   a larger penalty helps when the fitted examples are few.
4. **Export:** scaling is folded into the regression weight and bias, so the
   saved model is a normal Transformers sequence-regression checkpoint.
   predict.py clips its raw scalar and writes the required JSONL format.
5. **Selection:** compare ridge penalties on validation AUC, then evaluate the
   selected artifact on test examples. The count baseline is weaker; its
   early diagnostics were inspected, which the report discloses.
6. **Metrics:** AUC measures ranking of original versus changed texts;
   Spearman measures rank association with scores; MAE measures numerical
   distance from labels. Speed includes imports and model loading.
7. **Weakness:** a generated negative can still be good Swedish. The model can
   learn edit artifacts and familiarity instead of grammar. Binary labels
   cannot teach human severity reliably. We need a separate human-rated
   Swedish dataset and calibration to address that.
8. **Your next work:** create the virtual environment, run prediction, inspect
   an actual misranked pair in results/error_analysis.json, read train_encoder.py,
   and explain the ridge layer in your own words. Change any report statement
   you cannot support.

Before submitting, be able to show the exact commands, distinguish public
starter data from the organizer's hidden test, state the source license, and
explain why the measured performance is limited. Do not memorize unsupported
claims about Swedish grammar.
