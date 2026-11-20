"""Summarize actual held-out predictions without changing the model."""
import json
from pathlib import Path

from train import read


def main():
    rows = read("data/test.jsonl")
    predictions = {r["id"]:r["score"] for r in read("results/synthetic_scores.jsonl")}
    groups = {}
    for row in rows:
        groups.setdefault(row["group"],{})[row["corruption"]] = {**row,"prediction":predictions[row["id"]]}
    accuracy = {}
    failures = []
    for kind in ["swap","delete","duplicate","ending"]:
        pairs = [(g["original"],g[kind]) for g in groups.values() if "original" in g and kind in g]
        accuracy[kind] = {"pairs":len(pairs),"strict_pair_accuracy":sum(a["prediction"]>b["prediction"] for a,b in pairs)/len(pairs)}
        wrong = sorted(pairs,key=lambda p: p[0]["prediction"]-p[1]["prediction"])
        # Store a short actual example per type, rather than inventing linguistic judgments.
        short = next(((a,b) for a,b in wrong if len(a["text"])<150),wrong[0])
        a,b = short
        failures.append({"kind":kind,"source_id":a["source_id"],"original":a["text"],"changed":b["text"],
                         "original_prediction":a["prediction"],"changed_prediction":b["prediction"]})
    result = {"pair_results":accuracy,"examples":failures}
    Path("results/error_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
