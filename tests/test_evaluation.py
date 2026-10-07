import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import FIELDS, evaluate_predictions, parse_prediction

GOLD = {"name": "林晓雨", "address": None, "email": "a@example.com", "question": "漏水"}


class EvaluationTests(unittest.TestCase):
    def test_metrics_count_invalid_and_missing_schema_as_failures(self):
        good = json.dumps(GOLD)
        result = evaluate_predictions(
            [GOLD] * 4,
            [good, '{"name":"林晓雨"}', chr(96) * 3 + "json\n{}\n" + chr(96) * 3,
             '{"name":"林晓雨","address":"乱填","email":"a@example.com","question":"漏水"}'])
        self.assertEqual(result["json_parse_rate"], 0.75)
        self.assertEqual(result["schema_valid_rate"], 0.5)
        self.assertEqual(result["all_fields_exact_match_rate"], 0.25)
        self.assertEqual(result["field_exact_match_rate"], {"name": 0.5, "address": 0.25, "email": 0.5, "question": 0.5})
        self.assertEqual(result["null_false_fill_rate"], 0.25)
        self.assertEqual(result["null_failure_rate"], 0.75)
        self.assertEqual(len(result["rows"]), 4)

    def test_strict_parser_rejects_duplicate_nonfinite_and_trailing(self):
        malformed = (
            '{"name":null,"name":null,"address":null,"email":null,"question":null}',
            '{"name":NaN,"address":null,"email":null,"question":null}',
            '{"name":Infinity,"address":null,"email":null,"question":null}',
            '{"name":-Infinity,"address":null,"email":null,"question":null}',
            '{"name":null,"address":null,"email":null,"question":null} trailing',
            '{"name":{"a":1,"a":2},"address":null,"email":null,"question":null}',
        )
        for raw in malformed:
            with self.subTest(raw=raw):
                result = parse_prediction(raw)
                self.assertFalse(result["json_valid"])
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["raw"], raw)

    def test_zero_denominators_are_none(self):
        result = evaluate_predictions([], [])
        for metric in ("json_parse_rate", "schema_valid_rate", "all_fields_exact_match_rate",
                       "null_false_fill_rate", "null_failure_rate"):
            self.assertIsNone(result[metric])
        self.assertTrue(all(rate is None for rate in result["field_exact_match_rate"].values()))
        nonnull_gold = {field: "x" for field in FIELDS}
        result = evaluate_predictions([nonnull_gold], [json.dumps(nonnull_gold)])
        self.assertEqual(result["all_fields_exact_match_rate"], 1.0)
        self.assertIsNone(result["null_false_fill_rate"])
        self.assertIsNone(result["null_failure_rate"])

    def test_length_mismatch_and_bad_gold_raise(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([GOLD], [])
        for bad_gold in (None, [], {}, {"name": 3, "address": None, "email": None, "question": None}):
            with self.subTest(gold=bad_gold), self.assertRaises(ValueError):
                evaluate_predictions([bad_gold], ["{}"])

    def test_leading_whitespace_and_partial_schema_false_fill(self):
        raw = " \t\n" + json.dumps(GOLD) + "\r\n"
        self.assertTrue(parse_prediction(raw)["schema_valid"])
        partial = '{"name":"林晓雨","address":"乱填"}'
        result = evaluate_predictions([GOLD], [partial])
        self.assertEqual(result["null_false_fill_rate"], 1.0)
        self.assertEqual(result["null_failure_rate"], 1.0)
        self.assertTrue(all(rate == 0.0 for rate in result["field_exact_match_rate"].values()))
        self.assertEqual(result["rows"][0]["raw"], partial)
        self.assertEqual(result["rows"][0]["parsed"], {"name": "林晓雨", "address": "乱填"})

    def test_null_cells_are_counted_once_and_invalid_schema_fails_whole_row(self):
        gold = {field: None for field in FIELDS}
        cases = [
            ('{"name":null,"address":null,"email":null,"question":null}', 0.0, 0.0, 1.0),
            ('{"name":"x","address":null,"email":null,"question":null}', 0.25, 0.25, 0.0),
            ('{"name":"x"}', 0.25, 1.0, 0.0),
            ('{"name":null}', 0.0, 1.0, 0.0),
            ('{"name":0,"address":[],"email":{},"question":false}', 1.0, 1.0, 0.0),
            ('{"name":null,"address":null,"email":null,"question":null,"extra":0}', 0.0, 1.0, 0.0),
            ("[]", 0.0, 1.0, 0.0),
            ("invalid", 0.0, 1.0, 0.0),
        ]
        for raw, false_fill, failure, exact in cases:
            with self.subTest(raw=raw):
                result = evaluate_predictions([gold], [raw])
                self.assertEqual(result["null_gold_cells"], 4)
                self.assertEqual(result["null_false_fill_rate"], false_fill)
                self.assertEqual(result["null_failure_rate"], failure)
                self.assertEqual(result["all_fields_exact_match_rate"], exact)
                if not result["rows"][0]["schema_valid"]:
                    self.assertTrue(all(rate == 0.0 for rate in result["field_exact_match_rate"].values()))


    def test_overflow_number_is_rejected_without_nonfinite_parsed_output(self):
        for token in ("1e999", "-1e999"):
            raw = '{"name":' + token + ',"address":null,"email":null,"question":null}'
            result = parse_prediction(raw)
            self.assertFalse(result["json_valid"])
            self.assertFalse(result["schema_valid"])
            self.assertIsNone(result["parsed"])
            self.assertIn("finite parser range", result["error"])
            self.assertEqual(result["raw"], raw)
            json.dumps(evaluate_predictions([GOLD], [raw]), allow_nan=False)
