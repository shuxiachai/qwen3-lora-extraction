"""原模型/未来适配器共用评测入口，默认只显示计划；显式 --run 才推理。"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path, write_json
from src.evaluation import evaluate_predictions
from src.preprocessing import prompt_ids


def evaluation_generation_arguments(model, tokenizer, settings):
    """通过公开 generate 参数固定评估配置，不继承模型的采样默认值。"""
    from transformers import GenerationConfig

    # use_model_defaults=False 仍允许补齐缺失的 special token IDs；在这里
    # 显式完成这一步，使保存的配置与传给 generate 的配置一致。
    resolved = dict(settings)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
        if resolved.get(name) is None:
            resolved[name] = getattr(model.generation_config, name)
    for name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name)
        if token_id is not None:
            resolved[name] = token_id
    resolved["use_cache"] = True
    generation = GenerationConfig(**resolved)
    # kwargs 优先级最高；False/1/1.0 等显式值不能被模型默认值覆盖。
    return {"generation_config": generation, **resolved, "use_model_defaults": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--all", action="store_true", help="显式选择完整所选集合")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--adapter", help="下一阶段本地 adapter 路径")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    local_cache_environment()
    config = load_config(args.config)
    plan = {"mode": "run" if args.run else "dry_run", "model": config["model"],
            "adapter": args.adapter, "split": args.split, "limit": "all" if args.all else args.limit,
            "generation": config["evaluation"], "enable_thinking": False}
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if project_path(args.output).exists():
        raise FileExistsError(f"保留已有评测结果，请指定新的 --output：{project_path(args.output)}")
    import torch
    from transformers import __version__ as transformers_version, set_seed
    from src.modeling import load_tokenizer, load_base_model, load_adapter_for_inference
    set_seed(config["seed"])
    path = project_path(config["data"][args.split])
    data = path.read_bytes()
    records = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    if not args.all:
        records = records[:args.limit]
    tokenizer = load_tokenizer(config)
    model = load_adapter_for_inference(config, args.adapter) if args.adapter else load_base_model(config)
    model.eval()
    generation_arguments = evaluation_generation_arguments(model, tokenizer, config["evaluation"])
    generation = generation_arguments["generation_config"]
    raw_predictions, generated_tokens, generation_errors = [], [], []
    for record in records:
        ids = prompt_ids(tokenizer, record["input"], config)
        if len(ids) > config["data"]["max_length"]:
            raise ValueError(f"{record['id']}: 推理 prompt 超长；不做静默截断")
        inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
        try:
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **generation_arguments)
            suffix = output[0, len(ids):].tolist()
            # 只去掉生成末尾EOS，不 strip 围栏/解释，也不隐藏其它 special tokens。
            content = suffix[:-1] if suffix and suffix[-1] == tokenizer.eos_token_id else suffix
            raw_predictions.append(tokenizer.decode(content, skip_special_tokens=False))
            generated_tokens.append(suffix)
            generation_errors.append(None)
        except RuntimeError as exc:
            # 推理失败同样保留分母，以空字符串明确计为解析/字段失败。
            raw_predictions.append("")
            generated_tokens.append([])
            generation_errors.append(f"{type(exc).__name__}: {exc}")
    result = evaluate_predictions([row["output"] for row in records], raw_predictions)
    for record, row, tokens, error in zip(records, result["rows"], generated_tokens, generation_errors):
        row.update({"id": record["id"], "input": record["input"], "generated_token_ids": tokens,
                    "generation_error": error})
    result["provenance"] = {**plan, "date_utc": datetime.now(timezone.utc).isoformat(),
                            "dataset_sha256": hashlib.sha256(data).hexdigest(),
                            "selected_ids": [record["id"] for record in records],
                            "device": str(model.device), "dtype": str(next(model.parameters()).dtype),
                            "transformers_version": transformers_version,
                            "generation_requested": dict(config["evaluation"]),
                            "generation_effective": generation.to_dict(),
                            "generation_mode": generation.get_generation_mode().value,
                            "use_model_defaults": generation_arguments["use_model_defaults"],
                            "generation_model_defaults": model.generation_config.to_diff_dict(),
                            "generation_config_evidence": "显式 generate 参数；禁止补入模型生成默认值。max_length 在内部根据 prompt 长度和 max_new_tokens 推导。",
                            "prompt": config["prompt"], "seed": config["seed"],
                            "scope": "接口检查，非完整效果结论" if not args.all else "完整所选集合"}
    write_json(args.output, result)
    summary = {key: value for key, value in result.items() if key not in ("rows", "provenance")}
    summary["run"] = {"output": str(project_path(args.output)), "split": args.split,
                      "selected_ids": result["provenance"]["selected_ids"],
                      "generation_mode": generation.get_generation_mode().value,
                      "do_sample": generation.do_sample, "num_beams": generation.num_beams,
                      "max_new_tokens": generation.max_new_tokens, "use_model_defaults": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
