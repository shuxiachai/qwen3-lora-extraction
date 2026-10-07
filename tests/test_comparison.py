import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import _output_path, compare_results


GOLD = {"name": "A", "address": None, "email": "a@example.test", "question": "help"}


def result(raw, *, data="sha", max_new_tokens=8):
    metrics = {
        "sample_count": 1, "json_parse_rate": 1.0, "schema_valid_rate": 1.0,
        "field_exact_match_rate": {field: 1.0 for field in GOLD}, "all_fields_exact_match_rate": 1.0,
        "null_false_fill_rate": 0.0, "null_failure_rate": 0.0, "null_gold_cells": 1,
    }
    return {**metrics, "rows": [{"id": "r1", "input": "input", "gold": GOLD, "raw": raw}], "provenance": {
        "adapter": None, "dataset_sha256": data, "split": "test", "selected_ids": ["r1"],
        "model": {"id": "model", "revision": "rev"}, "prompt": {"system": "extract"}, "seed": 7,
        "dtype": "torch.float32", "device": "cuda:0", "generation_effective": {"do_sample": False, "max_new_tokens": max_new_tokens, "_from_model_config": False, "transformers_version": "x"},
        "generation_mode": "greedy_search", "use_model_defaults": False,
    }}


class ComparisonTests(unittest.TestCase):
    def test_output_rejects_parent_traversal_before_any_write(self):
        with self.assertRaisesRegex(ValueError, "inside the project root"):
            _output_path("../outside.json")

    def test_comparable_results_recompute_metrics_and_record_negative_delta(self):
        base = result(json.dumps(GOLD))
        adapter = result('{"name":"wrong","address":"filled","email":"a@example.test","question":"help"}')
        adapter["provenance"]["adapter"] = "artifacts/lora"
        # Saved summary is deliberately made correct for its raw prediction.
        adapter.update({"field_exact_match_rate": {"name": 0.0, "address": 0.0, "email": 1.0, "question": 1.0},
                        "all_fields_exact_match_rate": 0.0, "null_false_fill_rate": 1.0, "null_failure_rate": 1.0})
        comparison = compare_results(base, adapter)
        self.assertEqual(comparison["metrics"]["all_fields_exact_match_rate"]["delta"], -1.0)
        self.assertEqual(comparison["rows"][0]["null_false_fill_fields"]["adapter"], ["address"])

    def test_incomparable_data_generation_and_gold_are_rejected(self):
        base = result(json.dumps(GOLD))
        adapter = result(json.dumps(GOLD))
        adapter["provenance"]["adapter"] = "adapter"
        for mutate in (
            lambda value: value["provenance"].update({"dataset_sha256": "other"}),
            lambda value: value["provenance"]["generation_effective"].update({"max_new_tokens": 9}),
            lambda value: value["rows"][0].update({"gold": {**GOLD, "name": "other"}}),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                changed = copy.deepcopy(adapter)
                mutate(changed)
                compare_results(base, changed)

    def test_invalid_rows_remain_in_denominator_and_zero_null_denominator_is_none(self):
        gold = {field: "x" for field in GOLD}
        base = result("invalid")
        base["rows"][0]["gold"] = gold
        base.update({"json_parse_rate": 0.0, "schema_valid_rate": 0.0,
                     "field_exact_match_rate": {field: 0.0 for field in gold}, "all_fields_exact_match_rate": 0.0,
                     "null_false_fill_rate": None, "null_failure_rate": None, "null_gold_cells": 0})
        adapter = copy.deepcopy(base)
        adapter["provenance"]["adapter"] = "adapter"
        comparison = compare_results(base, adapter)
        self.assertEqual(comparison["metrics"]["sample_count"]["base"], 1)
        self.assertIsNone(comparison["metrics"]["null_failure_rate"]["delta"])


if __name__ == "__main__":
    unittest.main()
