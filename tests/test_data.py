import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import NAMES, ISSUES, SEED, build_records, split_records, template_signature
from check_data import check


def write_fixture(root, splits):
    data = root / "data"
    data.mkdir()
    for split, rows in splits.items():
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "seed": SEED, "synthetic": True,
        "fields": ["name", "address", "email", "question"],
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "group_counts": {split: len({row["group_id"] for row in rows}) for split, rows in splits.items()},
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class DataTests(unittest.TestCase):
    def test_fixed_group_split_has_60_20_20_and_no_group_leakage(self):
        splits = split_records(build_records())
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 60, "val": 20, "test": 20})
        seen = set()
        for rows in splits.values():
            groups = {row["group_id"] for row in rows}
            self.assertFalse(seen & groups)
            seen |= groups
        reversed_splits = split_records(list(reversed(build_records())))
        for split in splits:
            self.assertEqual({r["id"] for r in splits[split]}, {r["id"] for r in reversed_splits[split]})

    def test_all_variants_retain_one_template_and_templates_are_disjoint(self):
        templates_by_group = {}
        for row in build_records():
            templates_by_group.setdefault(row["group_id"], set()).add(template_signature(row["input"]))
        self.assertEqual(len(templates_by_group), 25)
        self.assertTrue(all(len(templates) == 1 for templates in templates_by_group.values()))
        self.assertEqual(len(set.union(*templates_by_group.values())), 25)

    def test_questions_preserve_original_punctuation_and_fictional_domains(self):
        for row in build_records():
            group = int(row["id"][1:3]) - 1
            for field, value in row["output"].items():
                if value is not None:
                    self.assertIn(value, row["input"], row["id"] + ": " + field)
            question = row["output"]["question"]
            if question is not None:
                self.assertEqual(question, ISSUES[group])
                self.assertTrue(question.endswith("。"))
            email = row["output"]["email"]
            if email is not None:
                self.assertTrue(email.endswith("@example.com"))

    def test_identity_and_contact_ownership_are_explicit(self):
        records = {row["id"]: row for row in build_records()}
        self.assertIn("转述人：", records["c04-1"]["input"])
        self.assertIn("并非投诉人", records["c04-1"]["input"])
        self.assertIsNone(records["c04-1"]["output"]["name"])
        self.assertIn("未说明谁是投诉人", records["c07-1"]["input"])
        self.assertIsNone(records["c07-1"]["output"]["name"])
        self.assertIn("承办点地址", records["c06-1"]["input"])
        self.assertIn("service06@example.com", records["c06-1"]["input"])
        self.assertIsNone(records["c06-1"]["output"]["address"])
        self.assertIsNone(records["c06-1"]["output"]["email"])
        self.assertTrue(all(value is None for value in records["c08-1"]["output"].values()))
        self.assertIsNone(records["c05-1"]["output"]["question"])

    def test_checker_accepts_generated_data_and_committed_files_match(self):
        splits = split_records(build_records())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, splits)
            self.assertEqual(check(root), [])
        self.assertEqual(check(ROOT), [])
        for split, rows in splits.items():
            actual = [json.loads(line) for line in (ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(actual, rows)

    def test_checker_catches_shared_legacy_template_across_groups(self):
        splits = copy.deepcopy(split_records(build_records()))
        for split in ("train", "val"):
            row = splits[split][0]
            group = int(row["id"][1:3]) - 1
            row["input"] = f"转述人{NAMES[(group + 7) % 25]}说有人遇到如下情况：{ISSUES[group]}未留下投诉人姓名、地址或邮箱。"
            row["output"] = {"name": None, "address": None, "email": None, "question": ISSUES[group]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, splits)
            self.assertTrue(any("normalized template crosses groups" in error for error in check(root)))

    def test_checker_catches_wrong_party_and_string_null(self):
        splits = copy.deepcopy(split_records(build_records()))
        row = next(row for rows in splits.values() for row in rows if row["id"] == "c06-1")
        row["output"]["email"] = "service06@example.com"
        row["output"]["address"] = "null"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, splits)
            errors = check(root)
            self.assertTrue(any("email does not match generated complainant/source" in error for error in errors))
            self.assertTrue(any("string null" in error for error in errors))

    def test_checker_reports_malformed_rows_without_crashing(self):
        malformed = [
            "[]", "null", '{"id":"x","id":"y"}',
            json.dumps({"id": [], "group_id": {}, "input": 1, "output": []}),
            json.dumps({"id": "c01-1", "group_id": "source-01", "input": "x",
                        "output": {"name": [], "address": None, "email": None, "question": 5}}),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, split_records(build_records()))
            (root / "data" / "train.jsonl").write_text("\n".join(malformed) + "\n", encoding="utf-8")
            errors = check(root)
            self.assertTrue(any("record keys invalid" in error for error in errors))
            self.assertTrue(any("duplicate key" in error for error in errors))
            self.assertTrue(any("invalid id" in error for error in errors))
            self.assertTrue(any("must be string/null" in error for error in errors))
