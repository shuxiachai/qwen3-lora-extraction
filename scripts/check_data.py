"""检查本项目 JSONL 合成集；发现问题时返回错误列表而非因畸形行崩溃。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from prepare_data import ISSUES, NAMES, SEED, template_signature

FIELDS = {"name", "address", "email", "question"}
ROOT = Path(__file__).resolve().parents[1]
SPLIT_COUNTS = {"train": 60, "val": 20, "test": 20}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def check(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    ids, inputs = set(), set()
    groups: dict[str, str] = {}
    group_sizes: dict[str, int] = {}
    templates: dict[str, str] = {}
    group_templates: dict[str, str] = {}
    actual_counts = {}
    actual_group_counts = {}

    for split, expected_count in SPLIT_COUNTS.items():
        path = root / "data" / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        actual_counts[split] = len(lines)
        if len(lines) != expected_count:
            errors.append(f"{split}: expected {expected_count} rows, got {len(lines)}")
        split_groups = set()
        for line_no, line in enumerate(lines, 1):
            where = f"{split}:{line_no}"
            if not line.strip():
                errors.append(f"{where}: empty line")
                continue
            try:
                item = json.loads(line, object_pairs_hook=_object, parse_constant=_reject_constant)
            except ValueError as exc:
                errors.append(f"{where}: invalid JSON: {exc}")
                continue
            if not isinstance(item, dict) or set(item) != {"id", "group_id", "input", "output"}:
                errors.append(f"{where}: record keys invalid")
                continue
            valid_strings = True
            for field in ("id", "group_id", "input"):
                if not isinstance(item[field], str) or not item[field].strip():
                    errors.append(f"{where}: invalid {field}")
                    valid_strings = False
            if not valid_strings:
                continue
            if item["id"] in ids:
                errors.append(f"{where}: duplicate id {item['id']}")
            ids.add(item["id"])
            if item["input"] in inputs:
                errors.append(f"{where}: duplicate input")
            inputs.add(item["input"])
            group_id = item["group_id"]
            if groups.get(group_id, split) != split:
                errors.append(f"{where}: group crosses splits")
            groups[group_id] = split
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
            split_groups.add(group_id)
            signature = template_signature(item["input"])
            if signature in templates and templates[signature] != group_id:
                errors.append(f"{where}: normalized template crosses groups")
            templates[signature] = group_id
            if group_id in group_templates and group_templates[group_id] != signature:
                errors.append(f"{where}: variants do not retain one group template")
            group_templates[group_id] = signature

            output = item["output"]
            if not isinstance(output, dict) or set(output) != FIELDS:
                errors.append(f"{where}: output keys invalid")
                continue
            for field, value in output.items():
                if value is not None and not isinstance(value, str):
                    errors.append(f"{where}: {field} must be string/null")
                    continue
                if value == "null":
                    errors.append(f"{where}: {field} uses string null")
                if value is not None and (not value or value not in item["input"]):
                    errors.append(f"{where}: {field} is not a nonempty verbatim input substring")
            if len(item["input"]) > 5120:
                errors.append(f"{where}: input exceeds 5120 characters")

            # 仅适用于本虚构数据集；候选值可从固定 id 推导。
            match = re.fullmatch(r"c(0[1-9]|1[0-9]|2[0-5])-([1-4])", item["id"])
            if match is None:
                errors.append(f"{where}: synthetic id cannot be checked")
                continue
            group, variant = map(int, match.groups())
            if group_id != f"source-{group:02d}":
                errors.append(f"{where}: id and group_id disagree")
            expected_sources = {
                "name": NAMES[group - 1],
                "address": f"云岚市星桥区幻月街{group}号雾港苑{variant + 1}室",
                "email": f"complain{group:02d}_{variant}@example.com",
                "question": ISSUES[group - 1],
            }
            for field, source in expected_sources.items():
                if output[field] is None and source in item["input"]:
                    errors.append(f"{where}: null {field} leaks generated source value")
                if output[field] is not None and output[field] != source:
                    errors.append(f"{where}: {field} does not match generated complainant/source")
        actual_group_counts[split] = len(split_groups)

    if len(group_sizes) != 25 or any(size != 4 for size in group_sizes.values()):
        errors.append("expected 25 groups of four records")
    manifest_path = root / "data" / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"missing {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=_reject_constant)
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            if manifest.get("seed") != SEED or manifest.get("synthetic") is not True:
                errors.append("manifest seed/synthetic mismatch")
            if manifest.get("fields") != ["name", "address", "email", "question"]:
                errors.append("manifest fields mismatch")
            if manifest.get("split_counts") != actual_counts or manifest.get("group_counts") != actual_group_counts:
                errors.append("manifest counts mismatch")
        except ValueError as exc:
            errors.append(f"manifest: invalid JSON: {exc}")
    return errors


if __name__ == "__main__":
    found = check()
    print("OK" if not found else "\n".join(found))
    sys.exit(bool(found))
