"""Evaluate a fixed local/Hub model through the required prediction interface."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from train_grammar import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--prefix",required=True)
    parser.add_argument("--splits",nargs="+",default=["validation","bridge_validation"])
    args = parser.parse_args()
    results = {}
    for split in args.splits:
        source = Path(f"data/grammar/{split}.jsonl")
        output = Path(f"results/{args.prefix}_{split}_scores.jsonl")
        start = time.perf_counter()
        subprocess.run([sys.executable,"predict.py","--model",args.model,
                        "--input",str(source),"--output",str(output)],check=True)
        seconds = time.perf_counter()-start
        rows = [json.loads(l) for l in source.read_text(encoding="utf-8").splitlines()]
        predicted = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in rows]==[r["id"] for r in predicted]
        assert all(set(r)=={"id","score"} and 0<=r["score"]<=1 for r in predicted)
        results[split] = {**metrics(rows,[r["score"] for r in predicted]),
                          "seconds":seconds,"samples_per_minute":len(rows)/seconds*60}
        print(json.dumps({split:results[split]}),flush=True)
    if "validation" in results and "bridge_validation" in results:
        results["objective"] = .75*results["validation"]["roc_auc"]+.25*results["bridge_validation"]["roc_auc"]
    Path(f"results/{args.prefix}_evaluation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
