"""Compare fixed holdout predictions; bootstrap whole correction families."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from train_grammar import metrics


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    rows = read("data/grammar/test.jsonl")
    new = {r["id"]:r["score"] for r in read("results/selected_holdout_test_scores.jsonl")}
    old = {r["id"]:r["score"] for r in read("results/legacy_holdout_test_scores.jsonl")}
    assert set(new)==set(old)=={r["id"] for r in rows}
    groups = {}
    for i,r in enumerate(rows):
        groups.setdefault(r["group"],{})[int(r["score"])] = i
    pairs = np.array([[g[1],g[0]] for g in groups.values()])
    y = np.array([r["score"] for r in rows])
    a,b = [np.array([prediction[r["id"]] for r in rows]) for prediction in [new,old]]
    rng = np.random.default_rng(42)
    delta = []
    for _ in range(1000):
        ix = pairs[rng.integers(0,len(pairs),len(pairs))].ravel()
        delta.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
    types = {}
    for tag in ["M","S","L","O","P"]:
        selected = [pair for pair in pairs if tag in rows[pair[1]]["error_type"].split(", ")]
        if selected:
            ix = np.array(selected).ravel()
            types[tag] = {"pairs":len(selected),"candidate_auc":float(roc_auc_score(y[ix],a[ix])),
                          "legacy_auc":float(roc_auc_score(y[ix],b[ix]))}
    failures = sorted(pairs,key=lambda pair:a[pair[0]]-a[pair[1]])
    short = [pair for pair in failures if len(rows[pair[0]]["text"])<150]
    example = (short or failures)[0]
    result = {"human_test_candidate":metrics(rows,a),"human_test_legacy":metrics(rows,b),
              "strict_pair_accuracy_candidate":float(np.mean(a[pairs[:,0]]>a[pairs[:,1]])),
              "strict_pair_accuracy_legacy":float(np.mean(b[pairs[:,0]]>b[pairs[:,1]])),
              "human_test_auc_delta":float(roc_auc_score(y,a)-roc_auc_score(y,b)),
              "auc_delta_group_bootstrap_95_ci":[float(x) for x in np.quantile(delta,[.025,.975])],
              "bootstrap_replicates":1000,"bootstrap_seed":42,"error_type_slices":types,
              "example":{"group":rows[example[0]]["group"],"error_type":rows[example[1]]["error_type"],
                         "correct":rows[example[0]]["text"],"incorrect":rows[example[1]]["text"],
                         "correct_score":float(a[example[0]]),"incorrect_score":float(a[example[1]])}}
    # Reuse the already recorded legacy predictions on exactly the same UD IDs.
    bridge = read("data/grammar/bridge_test.jsonl")
    previous = {r["id"]:r["score"] for r in read("results/synthetic_scores.jsonl")}
    scores = [previous[r["id"][3:]] for r in bridge]
    result["legacy_bridge_test"] = {**metrics(bridge,scores),
                                    "origin":"stored previous full-UD-test predictions; identical input IDs"}
    Path("results/grammar_error_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="example"},indent=2),flush=True)


if __name__ == "__main__":
    main()
