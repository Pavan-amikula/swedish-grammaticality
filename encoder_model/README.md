# Legacy frozen-regression baseline

Author: Pavan Kumar Goud Amikula.

This directory describes the previous one-logit regression model. Its weights
remain available at pavan0202/swedish-grammaticality revision
609cbf7e1f21c3de16bad534c068f9277324a896. It used frozen ModernBERT features
and a fitted ridge head on synthetic Talbanken data. Its score was a clamped
regression output. It is preserved for reproducible comparisons.

The selected model is now grammar_model, with two logits and a softmax
correct-class score. The current Hub main revision corresponds to that model.
Use the legacy revision explicitly when reproducing old results.

Base: AI-Sweden-Models/ModernBERT-base, Apache 2.0. Original license, model
card and modification notice are retained here. Data: CC BY-SA 4.0.
