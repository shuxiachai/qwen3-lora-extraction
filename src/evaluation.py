"""严格的四字段抽取评测；不依赖第三方库。"""
from __future__ import annotations

import json
import math
from typing import Any

FIELDS = ("name", "address", "email", "question")


class _DuplicateKey(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number exceeds finite parser range: {value}")
    return number


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate key: {key}")
        result[key] = value
    return result


def parse_prediction(raw: str) -> dict[str, Any]:
    """解析完整 JSON；拒绝重复键、非有限/溢出数值、围栏及尾随内容。"""
    result: dict[str, Any] = {"raw": raw, "parsed": None, "json_valid": False,
                              "schema_valid": False, "error": None}
    if not isinstance(raw, str):
        result["error"] = "prediction must be a string"
        return result
    try:
        # json.loads 接受合法 JSON 的首尾空白，并仍拒绝尾随非空白文本。
        value = json.loads(raw, object_pairs_hook=_reject_duplicates,
                           parse_constant=_reject_constant, parse_float=_finite_float)
        result["parsed"] = value
        result["json_valid"] = True
    except (ValueError, json.JSONDecodeError, _DuplicateKey) as exc:
        result["error"] = str(exc)
        return result
    if not isinstance(value, dict):
        result["error"] = "JSON root must be an object"
    elif set(value) != set(FIELDS):
        result["error"] = "schema requires exactly name, address, email, question"
    elif any(item is not None and not isinstance(item, str) for item in value.values()):
        result["error"] = "schema values must be strings or null"
    else:
        result["schema_valid"] = True
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_predictions(golds: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """计算指标。每项主指标均以全部 N 为分母；None 表示零分母。"""
    if len(golds) != len(predictions):
        raise ValueError("golds and predictions must have equal lengths")
    parsed_rows = [parse_prediction(raw) for raw in predictions]
    n = len(golds)
    json_count = sum(row["json_valid"] for row in parsed_rows)
    schema_count = sum(row["schema_valid"] for row in parsed_rows)
    exact_by_field = {field: 0 for field in FIELDS}
    all_correct = 0
    null_total = null_false_fill = null_failures = 0
    rows: list[dict[str, Any]] = []
    for index, (gold, parsed) in enumerate(zip(golds, parsed_rows)):
        if not isinstance(gold, dict) or set(gold) != set(FIELDS) or any(v is not None and not isinstance(v, str) for v in gold.values()):
            raise ValueError(f"invalid gold schema at index {index}")
        prediction = parsed["parsed"] if parsed["schema_valid"] else None
        observed_object = parsed["parsed"] if isinstance(parsed["parsed"], dict) else None
        field_matches: dict[str, bool] = {}
        for field in FIELDS:
            match = prediction is not None and prediction[field] == gold[field]
            field_matches[field] = match
            exact_by_field[field] += int(match)
            if gold[field] is None:
                null_total += 1
                # 即使同一对象缺其它键，显式非 null 仍是可观察的错误填入。
                explicit_fill = observed_object is not None and field in observed_object and observed_object[field] is not None
                if explicit_fill:
                    null_false_fill += 1
                if not parsed["schema_valid"] or explicit_fill:
                    null_failures += 1
        row_all_correct = all(field_matches.values())
        all_correct += int(row_all_correct)
        rows.append({"index": index, "gold": gold, **parsed,
                     "field_matches": field_matches, "all_correct": row_all_correct})
    return {
        "sample_count": n,
        "json_parse_rate": _rate(json_count, n),
        "schema_valid_rate": _rate(schema_count, n),
        "field_exact_match_rate": {field: _rate(exact_by_field[field], n) for field in FIELDS},
        "all_fields_exact_match_rate": _rate(all_correct, n),
        "null_false_fill_rate": _rate(null_false_fill, null_total),
        "null_failure_rate": _rate(null_failures, null_total),
        "null_gold_cells": null_total,
        "rows": rows,
    }
