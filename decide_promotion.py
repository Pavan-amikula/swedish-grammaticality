"""Compare a fixed candidate with the previous submission on validation only."""
import json
from pathlib import Path


def main():
    old = json.loads(Path("results/legacy_evaluation.json").read_text(encoding="utf-8"))
    new = json.loads(Path("results/candidate_evaluation.json").read_text(encoding="utf-8"))
    human = new["validation"]["roc_auc"]-old["validation"]["roc_auc"]
    bridge = new["bridge_validation"]["roc_auc"]-old["bridge_validation"]["roc_auc"]
    checks = {"human_auc_improves_at_least_0.02":human>=0.02,
              "bridge_auc_drops_no_more_than_0.02":bridge>=-0.02,
              "weighted_objective_improves":new["objective"]>old["objective"]}
    result = {"eligible":all(checks.values()),"checks":checks,
              "human_auc_difference":human,"bridge_auc_difference":bridge,
              "legacy":old,"candidate":new,
              "test_scores_used_for_decision":False}
    Path("results/promotion_decision.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
