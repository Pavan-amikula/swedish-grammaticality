"""Download an immutable licensed source and generate deterministic proxy labels."""
import hashlib
import json
import random
import urllib.request
from pathlib import Path

REVISION = "463c1eb7176c19e8d7cc54e5f5b0f21d6d6ce44d"
URL = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/{REVISION}/"
SEED = 42
SOURCE_HASHES = {
    "train":"133cf28248d4098c68b49f17999cb85403da1d93c9678cd70a564218cc4540a5",
    "dev":"31917fc04cfe9ef7ce8c11ed8570b739a0cefdcda5e06ff93207e6d84f37117b",
    "test":"b8909c4f5f2c64b3cee251de7004fa293717d309e54ca59d149a61d75e8571d7",
}


def dump(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8",newline="\n")


def parse(text):
    doc = None
    for block in text.split("\n\n"):
        item = {}
        for line in block.splitlines():
            if line.startswith("# newdoc id = "):
                doc = line.split(" = ", 1)[1]
            if line.startswith("# sent_id = "):
                item["source_id"] = line.split(" = ", 1)[1]
            if line.startswith("# text = "):
                item["text"] = line.split(" = ", 1)[1]
        if "text" in item and 6 <= len(item["text"].split()) <= 60:
            item["document"] = doc
            yield item


def normalized(text):
    return " ".join(text.lower().split())


def corrupt(text, kind, rng):
    words = text.split()
    if kind == "swap":
        i = rng.randrange(1, len(words)-2)
        words[i], words[i+1] = words[i+1], words[i]
    elif kind == "delete":
        candidates = [i for i,w in enumerate(words) if w.lower() in {"att", "och", "i", "på", "av", "för", "som", "är", "har"}]
        if not candidates:
            return None
        del words[rng.choice(candidates)]
    elif kind == "duplicate":
        i = rng.randrange(1, len(words)-1)
        words.insert(i, words[i])
    elif kind == "ending":
        candidates = [i for i,w in enumerate(words) if w.isalpha() and len(w) >= 5]
        if not candidates:
            return None
        i = rng.choice(candidates)
        words[i] = words[i][:-1]
    return " ".join(words)


def generate(rows, split):
    output = []
    for row in rows:
        # Sentence-local seed makes generation invariant to iteration order.
        rng = random.Random(int(hashlib.sha256((str(SEED)+row["source_id"]).encode()).hexdigest()[:16],16))
        variants = [(row["text"], "original", 1.0)]
        for kind in ["swap", "delete", "duplicate", "ending"]:
            changed = corrupt(row["text"], kind, rng)
            if changed and changed != row["text"]:
                variants.append((changed, kind, 0.0))
        for text, kind, score in variants:
            output.append({**row, "id": row["source_id"]+"-"+kind, "text": text,
                           "group": row["source_id"], "corruption": kind, "score": score,
                           "label": "good" if score else "poor", "split": split})
    return output


def main():
    root = Path("data")
    raw = root/"source"
    raw.mkdir(parents=True, exist_ok=True)
    provenance = {"url": URL, "revision": REVISION, "seed": SEED, "sha256": {}}
    source = {}
    for split in ["train", "dev", "test"]:
        name = f"sv_talbanken-ud-{split}.conllu"
        path = raw/name
        if not path.exists():
            urllib.request.urlretrieve(URL+name, path)
        provenance["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if provenance["sha256"][name] != SOURCE_HASHES[split]:
            raise ValueError(f"Source hash mismatch: {path}; remove it and download again")
        source[split] = list(parse(path.read_text(encoding="utf-8")))
    # Preserve official document separation and remove cross-split duplicates.
    seen_docs, seen_texts = set(), set()
    for split in ["test", "dev", "train"]:
        rows = [r for r in source[split] if r["document"] not in seen_docs and normalized(r["text"]) not in seen_texts]
        unique = {}
        for row in rows:
            unique.setdefault(normalized(row["text"]), row)
        source[split] = list(unique.values())
        seen_docs.update(r["document"] for r in rows)
        seen_texts.update(unique)
    # Whole documents separate language-model learning from score calibration.
    documents = sorted({r["document"] for r in source["train"]})
    random.Random(SEED).shuffle(documents)
    lm_docs = set(documents[:int(len(documents)*0.6)])
    lm = [r for r in source["train"] if r["document"] in lm_docs]
    calibration = [r for r in source["train"] if r["document"] not in lm_docs]
    dump(root/"lm_train.jsonl", [{**r, "id":r["source_id"], "score":1.0, "split":"lm_train"} for r in lm])
    occupied = {normalized(r["text"]) for r in lm}
    counts = {"lm_train":len(lm)}
    # Exclude exact collisions, including synthetic collisions, across partitions.
    for name, rows in [("calibration",calibration),("validation",source["dev"]),("test",source["test"])]:
        generated = generate(rows,name)
        bad_groups = {r["group"] for r in generated if normalized(r["text"]) in occupied}
        generated = [r for r in generated if r["group"] not in bad_groups]
        occupied.update(normalized(r["text"]) for r in generated)
        dump(root/f"{name}.jsonl",generated)
        counts[name] = len(generated)
    provenance["counts"] = counts
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
