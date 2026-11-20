"""Optional CPU experiment: frozen ModernBERT features, a fitted linear head.

Uses a bounded document-independent subset chosen without looking at labels.
The exported artifact is a standard Transformers sequence regression model.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from train import metrics, read

BASE = "AI-Sweden-Models/ModernBERT-base"
REVISION = "0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb"
MAX_LENGTH = 128


def subset(rows, count):
    groups = sorted({r["group"] for r in rows},key=lambda s: hashlib.sha256(("42"+s).encode()).hexdigest())[:count]
    return [r for r in rows if r["group"] in set(groups)]


@torch.no_grad()
def embed(rows, model, tokenizer):
    output = []
    for i in range(0,len(rows),16):
        batch = tokenizer([r["text"] for r in rows[i:i+16]],padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
        hidden = model.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden*mask).sum(1)/mask.sum(1)
        output.append(model.head(pooled).numpy())
        if i%160 == 0:
            print(f"Embedded {i}/{len(rows)}",flush=True)
    return np.concatenate(output)


def main():
    start = time.perf_counter()
    set_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE,revision=REVISION,num_labels=1,
              classifier_pooling="mean",problem_type="regression",attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    calibration = subset(read("data/calibration.jsonl"),256)
    validation = subset(read("data/validation.jsonl"),128)
    Path("data/encoder").mkdir(exist_ok=True)
    from prepare_data import dump
    dump("data/encoder/train.jsonl",calibration)
    dump("data/encoder/validation.jsonl",validation)
    x,v = [embed(rows,model,tokenizer) for rows in [calibration,validation]]
    scaler = StandardScaler().fit(x)
    trials, best = [], None
    for alpha in [10.0,100.0,1000.0]:
        reg = Ridge(alpha=alpha).fit(scaler.transform(x),[r["score"] for r in calibration])
        scores = np.clip(reg.predict(scaler.transform(v)),0,1)
        result = {"alpha":alpha,**metrics(validation,scores)}
        trials.append(result)
        if best is None or result["roc_auc"] > best[0]["roc_auc"]:
            best = result,reg
    result,reg = best
    from model import GrammaticalityModel
    count_model = GrammaticalityModel.from_pretrained("model").eval()
    count_validation = metrics(validation,count_model([r["text"] for r in validation]).tolist())
    selected_directory = "encoder_model" if result["roc_auc"] > count_validation["roc_auc"] else "model"
    # Fold StandardScaler into the single native regression layer.
    weight = reg.coef_/scaler.scale_
    intercept = reg.intercept_-np.dot(weight,scaler.mean_)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor(weight[None,:]))
        model.classifier.bias.fill_(float(intercept))
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.training_provenance = {"base":BASE,"revision":REVISION,"seed":42,"max_length":MAX_LENGTH}
    model.save_pretrained("encoder_model")
    tokenizer.save_pretrained("encoder_model")
    # Check exported native forward matches the fitted feature regressor.
    batch = tokenizer([r["text"] for r in validation[:16]],padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
    with torch.no_grad():
        actual = model(**batch).logits.squeeze(-1).numpy()
    expected = reg.predict(scaler.transform(v[:16]))
    np.testing.assert_allclose(actual,expected,atol=1e-4)
    Path("selection.json").write_text(json.dumps({"model_directory":selected_directory,
         "criterion":"ROC-AUC on the same 128-group validation subset", "ngram":count_validation,
         "encoder":result},indent=2),encoding="utf-8")
    payload = {"base":BASE,"revision":REVISION,"validation_trials":trials,"selected":result,
              "n_calibration":len(calibration),"n_validation":len(validation),
              "parameters":sum(p.numel() for p in model.parameters()),"fitted_parameters":model.config.hidden_size+1,
              "training_seconds":time.perf_counter()-start}
    Path("results/encoder_training.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__":
    main()
