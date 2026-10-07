"""Compare strictly comparable base-model and LoRA evaluation JSON results."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.evaluation import FIELDS, evaluate_predictions


DEFAULT_OUTPUT = "artifacts/reproduction_comparison.json"
_GENERATION_METADATA = {"_from_model_config", "transformers_version"}
_METRIC_KEYS = (
    "sample_count", "json_parse_rate", "schema_valid_rate", "field_exact_match_rate",
    "all_fields_exact_match_rate", "null_false_fill_rate", "null_failure_rate", "null_gold_cells",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        _fail(f"missing {label}.{key}")
    return mapping[key]


def _canonical_generation(provenance: dict[str, Any], label: str) -> dict[str, Any]:
    generation = _mapping(_required(provenance, "generation_effective", label), f"{label}.generation_effective")
    return {key: value for key, value in generation.items() if key not in _GENERATION_METADATA}


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _row_null_false_fill_fields(row: dict[str, Any]) -> list[str]:
    gold = _mapping(_required(row, "gold", "row"), "row.gold")
    parsed = row.get("parsed")
    observed = parsed if isinstance(parsed, dict) else {}
    return [field for field in FIELDS if gold.get(field) is None and field in observed and observed[field] is not None]


def _recompute_and_validate(result: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = _required(result, "rows", label)
    if not isinstance(rows, list):
        _fail(f"{label}.rows must be a list")
    golds, raws = [], []
    for index, row in enumerate(rows):
        row = _mapping(row, f"{label}.rows[{index}]")
        _required(row, "id", f"{label}.rows[{index}]")
        _required(row, "input", f"{label}.rows[{index}]")
        golds.append(_required(row, "gold", f"{label}.rows[{index}]"))
        raws.append(_required(row, "raw", f"{label}.rows[{index}]"))
    recomputed = evaluate_predictions(golds, raws)
    for key in _METRIC_KEYS:
        expected = _required(result, key, label)
        if expected != recomputed[key]:
            _fail(f"{label}.{key} does not match metrics recomputed from raw predictions")
    # Use the freshly calculated rows so stale saved parse fields cannot affect comparison.
    for original, fresh in zip(rows, recomputed["rows"]):
        fresh.update({"id": original["id"], "input": original["input"],
                      "generation_error": original.get("generation_error")})
    return recomputed["rows"]


def _validate_comparable(base: dict[str, Any], adapter: dict[str, Any], base_rows: list[dict[str, Any]], adapter_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_provenance = _mapping(_required(base, "provenance", "base"), "base.provenance")
    adapter_provenance = _mapping(_required(adapter, "provenance", "adapter"), "adapter.provenance")
    if base_provenance.get("adapter"):
        _fail("base provenance.adapter must be empty")
    if not adapter_provenance.get("adapter"):
        _fail("adapter provenance.adapter must name a LoRA adapter")
    for key in ("dataset_sha256", "split", "selected_ids", "prompt", "seed", "dtype", "device"):
        base_value = _required(base_provenance, key, "base.provenance")
        adapter_value = _required(adapter_provenance, key, "adapter.provenance")
        if base_value != adapter_value:
            _fail(f"incomparable provenance.{key}")
    if base_provenance["split"] != "test":
        _fail("comparison requires provenance.split == 'test'")
    for key in ("id", "revision"):
        base_model = _mapping(_required(base_provenance, "model", "base.provenance"), "base.provenance.model")
        adapter_model = _mapping(_required(adapter_provenance, "model", "adapter.provenance"), "adapter.provenance.model")
        if _required(base_model, key, "base.provenance.model") != _required(adapter_model, key, "adapter.provenance.model"):
            _fail(f"incomparable provenance.model.{key}")
    base_generation = _canonical_generation(base_provenance, "base.provenance")
    adapter_generation = _canonical_generation(adapter_provenance, "adapter.provenance")
    if base_generation != adapter_generation:
        _fail("incomparable provenance.generation_effective")
    if _hash(base_generation) != _hash(adapter_generation):  # Defensive and recorded as evidence.
        _fail("incomparable generation configuration hash")
    for provenance, label in ((base_provenance, "base"), (adapter_provenance, "adapter")):
        if _required(provenance, "generation_mode", f"{label}.provenance") != "greedy_search":
            _fail(f"{label} must use generation_mode 'greedy_search'")
        if _required(provenance, "use_model_defaults", f"{label}.provenance") is not False:
            _fail(f"{label} must set use_model_defaults to false")
        if base_generation.get("do_sample") is not False:
            _fail(f"{label} generation_effective.do_sample must be false")
    selected_ids = base_provenance["selected_ids"]
    if not isinstance(selected_ids, list):
        _fail("provenance.selected_ids must be a list")
    for index, (base_row, adapter_row) in enumerate(zip(base_rows, adapter_rows)):
        if base_row["id"] != adapter_row["id"] or base_row["id"] != (selected_ids[index] if index < len(selected_ids) else None):
            _fail(f"row id mismatch at index {index}")
        if base_row["input"] != adapter_row["input"]:
            _fail(f"row input mismatch at id {base_row['id']}")
        if base_row["gold"] != adapter_row["gold"]:
            _fail(f"row gold mismatch at id {base_row['id']}")
    if len(base_rows) != len(adapter_rows) or len(base_rows) != len(selected_ids):
        _fail("rows and selected_ids must have equal lengths")
    return base_provenance, adapter_provenance, base_generation


def _metric_columns(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        base_value, adapter_value = base[key], adapter[key]
        if isinstance(base_value, dict):
            columns[key] = {
                field: {"base": base_value[field], "adapter": adapter_value[field],
                        "delta": None if base_value[field] is None or adapter_value[field] is None else adapter_value[field] - base_value[field]}
                for field in FIELDS
            }
        else:
            columns[key] = {"base": base_value, "adapter": adapter_value,
                            "delta": None if base_value is None or adapter_value is None else adapter_value - base_value}
    return columns


def compare_results(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    """Validate two saved evaluation results and return a row- and metric-level comparison."""
    base_rows = _recompute_and_validate(base, "base")
    adapter_rows = _recompute_and_validate(adapter, "adapter")
    base_provenance, adapter_provenance, generation = _validate_comparable(base, adapter, base_rows, adapter_rows)
    row_comparison = []
    for base_row, adapter_row in zip(base_rows, adapter_rows):
        row_comparison.append({
            "id": base_row["id"],
            "all_correct": {"base": base_row["all_correct"], "adapter": adapter_row["all_correct"],
                            "delta": int(adapter_row["all_correct"]) - int(base_row["all_correct"])},
            "field_correctness": {field: {"base": base_row["field_matches"][field], "adapter": adapter_row["field_matches"][field],
                                             "delta": int(adapter_row["field_matches"][field]) - int(base_row["field_matches"][field])}
                                  for field in FIELDS},
            "error": {"base": base_row["error"], "adapter": adapter_row["error"]},
            "generation_error": {"base": base_row["generation_error"], "adapter": adapter_row["generation_error"]},
            "null_false_fill_fields": {"base": _row_null_false_fill_fields(base_row), "adapter": _row_null_false_fill_fields(adapter_row)},
        })
    return {
        "comparison": "base_vs_lora",
        "comparability": {
            "dataset_sha256": base_provenance["dataset_sha256"], "split": "test", "selected_ids": base_provenance["selected_ids"],
            "model": {key: base_provenance["model"][key] for key in ("id", "revision")}, "prompt": base_provenance["prompt"],
            "seed": base_provenance["seed"], "dtype": base_provenance["dtype"], "device": base_provenance["device"],
            "generation_effective_sha256": _hash(generation), "generation_mode": "greedy_search", "do_sample": False,
            "use_model_defaults": False, "adapter_path": adapter_provenance["adapter"],
        },
        "metrics": _metric_columns(base, adapter),
        "rows": row_comparison,
        "limitations": ["This comparison reports observed deltas only; it does not establish improvement.", "A synthetic evaluation with fewer than 20 samples has limited evidentiary value."],
    }


def _read_result(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} result {path}: {exc}") from exc
    return _mapping(data, label)


def _output_path(value: str) -> Path:
    path = Path(value)
    target = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("--output must be inside the project root") from exc
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base-model evaluation JSON")
    parser.add_argument("--adapter", required=True, help="LoRA evaluation JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"project-relative output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    output = _output_path(args.output)
    if output.exists():
        _fail(f"refusing to overwrite existing output: {output}")
    comparison = compare_results(_read_result(Path(args.base), "base"), _read_result(Path(args.adapter), "adapter"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": len(comparison["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
