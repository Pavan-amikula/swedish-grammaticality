"""Checks for correction-family separation, paired training, and score ordering."""
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_grammar_data import normalize
from train_grammar import paired_batches


class GrammarTests(unittest.TestCase):
    def test_correct_members_have_a_published_positive_label(self):
        positive = {}
        for split in ["train","dev","test"]:
            path = ROOT/f"data/grammar/source/{split}.jsonl"
            released = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            positive[split] = {normalize(r["sentence"]) for r in released if r["label"]=="correct"}
        for split in ["train","calibration","validation","test"]:
            rows = [json.loads(l) for l in (ROOT/f"data/grammar/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in rows:
                if row["score"]==1:
                    self.assertIn(normalize(row["text"]),positive[row["source_split"]])

    def test_families_and_sentences_do_not_cross_partitions(self):
        seen_groups,seen_texts = set(),set()
        for split in ["train","calibration","validation","test","bridge_train","bridge_validation","bridge_test"]:
            rows = [json.loads(l) for l in (ROOT/f"data/grammar/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            groups = defaultdict(list)
            for row in rows:
                groups[row["group"]].append(row)
            for family in groups.values():
                self.assertEqual(len(family),2)
                self.assertEqual({r["score"] for r in family},{0.0,1.0})
                self.assertNotEqual(family[0]["text"],family[1]["text"])
            texts = {normalize(r["text"]).casefold() for r in rows}
            self.assertFalse(set(groups)&seen_groups,split)
            self.assertFalse(texts&seen_texts,split)
            seen_groups.update(groups)
            seen_texts.update(texts)

    def test_pair_batches_keep_correction_before_error(self):
        class Padding:
            def pad(self,items,**kwargs):
                return {"size":len(items)}
        rows = [json.loads(l) for l in (ROOT/"data/grammar/train.jsonl").read_text(encoding="utf-8").splitlines()[:40]]
        encoded = [{"input_ids":list(range(i%9+2))} for i in range(len(rows))]
        encountered = []
        for indices,batch in paired_batches(rows,encoded,Padding(),8,42):
            self.assertEqual(batch["size"],len(indices))
            self.assertEqual(len(indices)%2,0)
            for a,b in zip(indices[0::2],indices[1::2]):
                self.assertEqual(rows[a]["group"],rows[b]["group"])
                self.assertEqual((rows[a]["score"],rows[b]["score"]),(1.0,0.0))
            encountered.extend(indices)
        self.assertEqual(sorted(encountered),list(range(len(rows))))

    def test_scoring_is_stable_when_length_sort_changes_order(self):
        from predict import EncoderScorer
        scorer = EncoderScorer(str(ROOT/"grammar_model"))
        texts = ["Hon läser en bok.","Det här är en längre mening som används för att kontrollera ordningen.",""]
        forward = scorer(texts).numpy()
        backward = scorer(texts[::-1]).numpy()[::-1]
        alone = np.array([scorer([text]).item() for text in texts])
        np.testing.assert_allclose(forward,backward,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(forward,alone,atol=1e-5,rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
