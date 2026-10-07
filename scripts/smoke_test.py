"""默认仅检查 tokenizer/labels；--forward/--backward 才加载模型，无优化器。"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path, write_json
from src.preprocessing import encode_example, pad_examples, label_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--backward", action="store_true", help="一次反传，隐含 --forward；绝不更新参数")
    parser.add_argument("--samples", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    local_cache_environment()
    config = load_config(args.config)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_id": config["model"]["id"], "revision": config["model"]["revision"],
              "optimizer_steps": 0, "weights_saved": False,
              "mode": "backward" if args.backward else "forward" if args.forward else "dry_run",
              "status": "running", "stage": "tokenizer", "config": config}
    output_path = f"artifacts/smoke_{report['mode']}.json"
    try:
        from src.modeling import load_tokenizer, load_base_model, attach_lora, gradient_audit
        tokenizer = load_tokenizer(config)
        # smoke 固定取 train 开头，不打开测试集选案例或调 prompt。
        train_bytes = project_path(config["data"]["train"]).read_bytes()
        records = [json.loads(line) for line in train_bytes.decode("utf-8").splitlines()]
        report["train_sha256"] = hashlib.sha256(train_bytes).hexdigest()
        selected = records[:args.samples]
        if len(selected) != args.samples:
            raise ValueError("训练集不足所需样本数")
        encoded = [encode_example(record, tokenizer, config) for record in selected]
        for record, item in zip(selected, encoded):
            write_json(f"artifacts/token_labels_{record['id']}.json", label_trace(record, item, tokenizer))
        # 在 CPU 上校验整份 train/val 的长度；不对 test 做监督分析或调参。
        length_summary = {}
        from datasets import Dataset
        for split in ("train", "val"):
            rows = [json.loads(line) for line in project_path(config["data"][split]).read_text(encoding="utf-8").splitlines()]
            ds = Dataset.from_list(rows).map(lambda row: encode_example(row, tokenizer, config), remove_columns=list(rows[0]))
            lengths = [len(row["input_ids"]) for row in ds]
            length_summary[split] = {"count": len(lengths), "min_tokens": min(lengths), "max_tokens": max(lengths)}
        batch = pad_examples(encoded, tokenizer.pad_token_id)
        report.update({"sample_ids": [row["id"] for row in selected], "sequence_lengths": [len(row["input_ids"]) for row in encoded],
                       "sample_snapshot": selected, "length_summary": length_summary, "eos_token_id": tokenizer.eos_token_id,
                       "pad_token_id": tokenizer.pad_token_id, "tokenizer_checks": "passed"})
        write_json("artifacts/smoke_batch.json", batch)
        if args.forward or args.backward:
            import torch
            from transformers import set_seed
            set_seed(config["seed"])
            report["stage"] = "model_and_lora"
            model, audit = attach_lora(load_base_model(config), config)
            write_json("artifacts/lora_audit.json", audit)
            model.print_trainable_parameters()
            print("LoRA modules:\n" + "\n".join(audit["adapter_modules"]))
            report.update({"device": str(model.device), "dtype": str(next(model.parameters()).dtype),
                           "total_parameters": audit["total_parameters"], "trainable_parameters": audit["trainable_parameters"],
                           "base_all_frozen": audit["base_all_frozen"]})
            model.train()
            model.zero_grad(set_to_none=True)
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            report["stage"] = "forward"
            # 单设备 batch_size=1；若选择两条仅做拼接检查，模型前传仍固定一条。
            single = pad_examples(encoded[:1], tokenizer.pad_token_id)
            inputs = {key: torch.tensor(value, dtype=torch.long, device=model.device) for key, value in single.items()}
            with torch.set_grad_enabled(args.backward):
                output = model(**inputs, use_cache=False)
            if not torch.isfinite(output.loss).item():
                raise AssertionError("loss 不是有限数")
            report["loss"] = output.loss.item()
            report["forward"] = "passed"
            if args.backward:
                report["stage"] = "backward"
                output.loss.backward()
                report["gradient_audit"] = gradient_audit(model)
                report["backward"] = "passed"
            else:
                report["backward"] = "not_executed"
            if model.device.type == "cuda":
                torch.cuda.synchronize()
                report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            model.zero_grad(set_to_none=True)
        else:
            report.update({"forward": "not_executed", "backward": "not_executed"})
        report.update({"status": "passed", "stage": "complete"})
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(output_path, report)
        raise
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
