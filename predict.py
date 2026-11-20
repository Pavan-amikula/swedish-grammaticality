"""Required evaluator interface. Run from any working directory."""
import argparse
import json
from pathlib import Path

from model import GrammaticalityModel


class EncoderScorer:
    def __init__(self, directory):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory,attn_implementation="eager").eval()

    def __call__(self, texts):
        import torch
        if not texts:
            return torch.empty(0)
        result = []
        with torch.no_grad():
            for i in range(0,len(texts),16):
                batch = self.tokenizer(texts[i:i+16],padding=True,truncation=True,max_length=128,return_tensors="pt")
                result.append(self.model(**batch).logits.squeeze(-1).clamp(0,1))
        return torch.cat(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    root = Path(__file__).resolve().parent
    selection = root/"selection.json"
    settings = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {"model_directory":"model"}
    parser.add_argument("--model",help="Local model directory or a published Hugging Face model ID")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must be different files")
    if Path(args.input).stat().st_size == 0:
        Path(args.output).write_text("",encoding="utf-8")
        return
    local = root/settings["model_directory"]
    source = args.model or (str(local) if (local/"model.safetensors").exists() else settings.get("hub_model_id"))
    if not source:
        raise FileNotFoundError("Model weights missing. Run training or set a verified published Hub ID in selection.json.")
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download
        source = snapshot_download(source,revision=settings.get("hub_revision") if args.model is None else None,
                                   allow_patterns=["*.json","*.safetensors"])
    config = json.loads((Path(source)/"config.json").read_text(encoding="utf-8"))
    model = GrammaticalityModel.from_pretrained(source).eval() if "statistics" in config else EncoderScorer(source)
    # Bounded memory, preserving input order, duplicate IDs, and JSON id types.
    with open(args.input, encoding="utf-8") as inp, open(args.output,"w",encoding="utf-8") as out:
        batch = []
        for number, line in enumerate(inp,1):
            row = json.loads(line)
            if "id" not in row or not isinstance(row.get("text"),str):
                raise ValueError(f"Line {number}: expected id and string text")
            batch.append(row)
            if len(batch) == 64:
                write_batch(model,batch,out)
                batch = []
        write_batch(model,batch,out)


def write_batch(model, rows, out):
    scores = model([r["text"] for r in rows]).tolist()
    for row, score in zip(rows,scores):
        out.write(json.dumps({"id":row["id"],"score":score},ensure_ascii=False,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
