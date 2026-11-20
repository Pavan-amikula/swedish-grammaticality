"""Fit count tables on clean text, ridge on disjoint synthetic examples."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from model import GrammaticalityModel, features, fit_statistics


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def metrics(rows, scores):
    y = [r["score"] for r in rows]
    return {"mae":float(mean_absolute_error(y,scores)),
            "spearman":float(spearmanr(y,scores).statistic),
            "roc_auc":float(roc_auc_score([r["label"]=="good" for r in rows],scores))}


def main():
    start = time.perf_counter()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lm, calibration, validation = [read(f"data/{s}.jsonl") for s in ["lm_train","calibration","validation"]]
    statistics = fit_statistics([r["text"] for r in lm])
    x = np.array([features(r["text"],statistics) for r in calibration])
    v = np.array([features(r["text"],statistics) for r in validation])
    y = np.array([r["score"] for r in calibration])
    trials, best = [], None
    for family, indices in [("unigram",[0,1,6,7,8,17]),("ngram",list(range(x.shape[1])))]:
        scaler = StandardScaler().fit(x[:,indices])
        for alpha in [1.0,10.0,100.0]:
            reg = Ridge(alpha=alpha).fit(scaler.transform(x[:,indices]),y)
            scores = np.clip(reg.predict(scaler.transform(v[:,indices])),0,1)
            result = {"family":family,"alpha":alpha,**metrics(validation,scores)}
            trials.append(result)
            if best is None or result["roc_auc"] > best[0]["roc_auc"]:
                best = (result,indices,scaler,reg)
    result, indices, scaler, reg = best
    model = GrammaticalityModel(statistics, indices)
    with torch.no_grad():
        model.center.copy_(torch.from_numpy(scaler.mean_))
        model.scale.copy_(torch.from_numpy(scaler.scale_))
        model.linear.weight.copy_(torch.from_numpy(reg.coef_[None,:]))
        model.linear.bias.fill_(reg.intercept_)
    model.save_pretrained("model")
    # Reload the exact submission artifact before reporting held-out results.
    restored = GrammaticalityModel.from_pretrained("model").eval()
    test = read("data/test.jsonl")
    scores = restored([r["text"] for r in test]).tolist()
    payload = {"validation_trials":trials,"selected":result,"synthetic_test":metrics(test,scores),
               "constant_test_mae":float(mean_absolute_error([r["score"] for r in test], np.repeat(y.mean(),len(test)))),
               "trainable_parameters":sum(p.numel() for p in model.parameters()),
               "count_table_entries":sum(len(t) for ts in statistics for t in ts),
               "training_seconds":time.perf_counter()-start}
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
