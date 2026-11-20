"""Time the complete required CLI, including imports/model loading."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from train import metrics, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default="data/test.jsonl")
    parser.add_argument("--labels")
    parser.add_argument("--name",default="synthetic")
    args = parser.parse_args()
    output = f"results/{args.name}_scores.jsonl"
    start = time.perf_counter()
    subprocess.run([sys.executable,"predict.py","--input",args.input,"--output",output],check=True)
    seconds = time.perf_counter()-start
    rows = read(args.input)
    predictions = read(output)
    assert [r["id"] for r in rows] == [r["id"] for r in predictions]
    assert all(set(r)=={"id","score"} and 0<=r["score"]<=1 for r in predictions)
    if args.labels:
        labels = {r["id"]:r for r in read(args.labels)}
        rows = [{**r,**labels[r["id"]]} for r in rows]
    result = {**metrics(rows,[r["score"] for r in predictions]),"n":len(rows),"seconds":seconds,
              "samples_per_minute":len(rows)/seconds*60,"platform":platform.platform(),
              "processor":platform.processor(),"python":platform.python_version()}
    Path(f"results/{args.name}_benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
