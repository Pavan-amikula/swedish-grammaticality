import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import GrammaticalityModel, fit_statistics, features
from prepare_data import normalized


class ProjectTests(unittest.TestCase):
    def test_language_model_probabilities_never_exceed_one(self):
        statistics = fit_statistics(["Samma mening."]*20)
        self.assertTrue(all(x >= 0 for x in features("Samma mening.",statistics)))

    def test_partitions_have_no_document_or_text_overlap(self):
        seen_docs, seen_texts, seen_groups = set(),set(),set()
        for split in ["lm_train","calibration","validation","test"]:
            rows = [json.loads(l) for l in (ROOT/f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            docs = {r["document"] for r in rows}
            texts = {normalized(r["text"]) for r in rows}
            groups = {r["source_id"] for r in rows}
            self.assertFalse(docs & seen_docs,split)
            self.assertFalse(texts & seen_texts,split)
            self.assertFalse(groups & seen_groups,split)
            seen_docs.update(docs)
            seen_texts.update(texts)
            seen_groups.update(groups)

    def test_hub_save_reload_preserves_scores(self):
        model = GrammaticalityModel.from_pretrained(str(ROOT/"model")).eval()
        texts = ["", "   ", "åäö", "En svensk mening.","🙂", "word "*1000]
        before = model(texts).numpy()
        self.assertTrue(np.isfinite(before).all())
        self.assertTrue(((before>=0)&(before<=1)).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            after = GrammaticalityModel.from_pretrained(directory)(texts).numpy()
        np.testing.assert_array_equal(before,after)

    def test_required_cli_and_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            inp,out = Path(directory)/"input.jsonl",Path(directory)/"scores.jsonl"
            rows = [{"id":None,"text":""},{"id":7,"text":"Hon läser en bok."},{"id":7,"text":"🙂"},{"id":"å","text":"a"}]
            for batch in [rows,[]]:
                inp.write_text("".join(json.dumps(r)+"\n" for r in batch),encoding="utf-8")
                subprocess.run([sys.executable,str(ROOT/"predict.py"),"--input",str(inp),"--output",str(out)],cwd=directory,check=True)
                results = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["id"] for r in batch],[r["id"] for r in results])
                self.assertTrue(all(set(r)=={"id","score"} and math.isfinite(r["score"]) and 0<=r["score"]<=1 for r in results))


if __name__ == "__main__":
    unittest.main()
