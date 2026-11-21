"""Prepare paired, correction-family-disjoint DaLAJ data and a small UD bridge.

The upstream release is public CC BY 4.0. No learner demographics or essay
metadata are retained. Corrected sentences use the released expert correction.
"""
import hashlib
import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "sprakbanken-se/superlim-2"
REVISION = "e70516a7f2397b68e4f2bef3cb984944b1c670eb"
SEED = 42


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows),
                          encoding="utf-8", newline="\n")


class Families:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def ud_pairs(path, limit, reserved):
    groups = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        groups[row["group"]][row["corruption"]] = row
    result = []
    for group in sorted(groups, key=lambda g: digest(str(SEED)+g)):
        variants = groups[group]
        bad = [r for k, r in variants.items() if k != "original"]
        if "original" not in variants or not bad:
            continue
        wrong = min(bad, key=lambda r: digest(str(SEED)+r["text"]))
        pair = [variants["original"], wrong]
        if any(normalize(r["text"]).casefold() in reserved for r in pair):
            continue
        for r in pair:
            result.append({"id":"ud-"+str(r["id"]), "group":"ud-"+group,
                           "text":r["text"], "score":r["score"], "label":r["label"],
                           "source":"talbanken-synthetic", "source_split":r["split"],
                           "error_type":r["corruption"]})
        if len(result) == 2*limit:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unconfirmed-corrections",action="store_true",
                        help="Reproduce the preliminary experiment; omit for final data")
    args = parser.parse_args()
    raw = Path("data/grammar/source")
    raw.mkdir(parents=True, exist_ok=True)
    families, records, labels, hashes = Families(), [], defaultdict(set), {}
    rejected = Counter()
    for split in ["train", "dev", "test"]:
        source = Path(hf_hub_download(REPOSITORY,
                    f"data/dalaj-ged-superlim/dalaj-ged-superlim_{split}.jsonl",
                    repo_type="dataset", revision=REVISION))
        content = source.read_bytes()
        hashes[split] = hashlib.sha256(content).hexdigest()
        (raw/f"{split}.jsonl").write_bytes(content)
        released_rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        correct_texts = defaultdict(list)
        for row in released_rows:
            if row["label"]=="correct":
                correct_texts[normalize(row["sentence"]).casefold()].append(row["sentence"])
        for number, r in enumerate(released_rows):
            text = r["sentence"]
            if not 3 <= len(text.split()) <= 80 or "\ufffd" in text:
                rejected["length_or_encoding"] += 1
                continue
            key = normalize(text)
            labels[key].add(r["label"])
            families.find(key.casefold())
            if r["label"] == "correct":
                records.append({"split":split,"text":text,"canonical":text,
                                "source_row":number,"incorrect":False})
                continue
            meta = r["meta"]
            start, stop = meta["error_span"]["start"], meta["error_span"]["stop"]
            wrong = meta["confusion_pair"]["incorrect_span"] or ""
            correction = meta["confusion_pair"]["correction"] or ""
            if not (isinstance(start,int) and isinstance(stop,int) and
                    0 <= start <= stop <= len(text) and text[start:stop] == wrong):
                rejected["span_mismatch"] += 1
                continue
            canonical = text[:start]+correction+text[stop:]
            if not args.allow_unconfirmed_corrections:
                # Empty error spans denote omitted tokens. Source corrections
                # need not contain their surrounding spaces. Accept only a
                # spacing choice that matches a released correct sentence.
                for left,right in [("",""),(""," "),(" ",""),(" "," ")]:
                    candidate = text[:start]+left+correction+right+text[stop:]
                    matches = correct_texts[normalize(candidate).casefold()]
                    if matches:
                        canonical = min(matches,key=lambda t:digest(str(SEED)+t))
                        break
            if not 3 <= len(canonical.split()) <= 80 or normalize(canonical)==key:
                rejected["unchanged_or_length"] += 1
                continue
            families.union(key.casefold(), normalize(canonical).casefold())
            records.append({"split":split,"text":text,"canonical":canonical,
                            "error_type":meta["error_label"],"source_row":number,
                            "incorrect":True})
    conflicts = {k for k, values in labels.items() if len(values)>1}
    published_good = defaultdict(list)
    for r in records:
        if not r["incorrect"]:
            published_good[(r["split"],normalize(r["text"]).casefold())].append(r["text"])
    groups = defaultdict(list)
    for r in records:
        groups[families.find(normalize(r["canonical"]).casefold())].append(r)
    candidates, counts = defaultdict(list), Counter()
    priority = {"train":0,"dev":1,"test":2}
    for group, members in groups.items():
        winner = max((r["split"] for r in members),key=priority.get)
        counts["source_members_excluded_from_earlier_splits"] += sum(r["split"]!=winner for r in members)
        bad = [r for r in members if r["incorrect"] and r["split"]==winner and
               normalize(r["text"]) not in conflicts and normalize(r["canonical"]) not in conflicts]
        if not args.allow_unconfirmed_corrections:
            confirmed = []
            for r in bad:
                matches = published_good[(winner,normalize(r["canonical"]).casefold())]
                if matches:
                    canonical = min(matches,key=lambda text:digest(str(SEED)+text))
                    confirmed.append({**r,"canonical":canonical})
                else:
                    rejected["no_published_correct_counterpart"] += 1
            bad = confirmed
        if not bad:
            continue
        # One variant and its expert correction per family avoids frequency bias.
        chosen = min(bad, key=lambda r:digest(str(SEED)+normalize(r["text"])))
        candidates[winner].append((digest(group), chosen))
    for name in candidates:
        candidates[name].sort(key=lambda pair:digest(str(SEED)+pair[0]))
    # All group decisions are fixed before any model is scored.
    assignments = {
        "calibration":candidates["train"][:500],
        "train":candidates["train"][500:4500],
        "validation":candidates["dev"][:750],
        "test":candidates["test"],
    }
    output = {}
    for split, selected in assignments.items():
        rows = []
        for group, r in selected:
            for correct in [True, False]:
                kind = "correct" if correct else "incorrect"
                rows.append({"id":f"dalaj-{group}-{kind}","group":f"dalaj-{group}",
                             "text":r["canonical"] if correct else r["text"],
                             "score":float(correct),"label":"good" if correct else "poor",
                             "source":"dalaj-ged","source_split":r["split"],
                             "error_type":"none" if correct else r["error_type"]})
        output[split] = rows
        write(f"data/grammar/{split}.jsonl", rows)
    reserved = {normalize(r["text"]).casefold() for rows in output.values() for r in rows}
    output["bridge_train"] = ud_pairs("data/calibration.jsonl",400,reserved)
    output["bridge_validation"] = ud_pairs("data/validation.jsonl",200,reserved)
    output["bridge_test"] = ud_pairs("data/test.jsonl",200,reserved)
    for split in ["bridge_train", "bridge_validation", "bridge_test"]:
        write(f"data/grammar/{split}.jsonl",output[split])
    seen_groups, seen_texts = set(), set()
    for split in output:
        current_groups = {r["group"] for r in output[split]}
        current_texts = {normalize(r["text"]).casefold() for r in output[split]}
        assert not current_groups & seen_groups,split
        assert not current_texts & seen_texts,split
        seen_groups.update(current_groups)
        seen_texts.update(current_texts)
    provenance = {"repository":REPOSITORY,"revision":REVISION,"seed":SEED,
                  "curation_version":"v1-derived-corrections" if args.allow_unconfirmed_corrections else "v3-source-confirmed-spacing",
                  "license":"CC-BY-4.0 (DaLAJ); CC-BY-SA-4.0 (UD bridge)",
                  "source_sha256":hashes,"rejected":dict(rejected),"filter_counts":dict(counts),
                  "eligible_families":{s:len(r) for s,r in candidates.items()},
                  "rows":{s:len(r) for s,r in output.items()},
                  "error_types_train":dict(Counter(r["error_type"] for r in output["train"] if r["score"]==0)),
                  "prepared_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(Path("data/grammar").glob("*.jsonl"))},
                  "group_and_text_overlap_checks_passed":True}
    Path("data/grammar/provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps(provenance,indent=2),flush=True)


if __name__ == "__main__":
    main()
