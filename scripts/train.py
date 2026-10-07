"""首轮单 epoch LoRA 训练；默认仅核验 train/val，显式 --run 才训练。"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path, write_json
from src.training import prepare_training, training_plan, validate_training_settings


def cuda_memory(torch, device):
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def save_reload_reference(model, tokenizer, record, config, output):
    import torch
    from scripts.evaluate import evaluation_generation_arguments
    from src.preprocessing import prompt_ids

    model.eval()
    ids = prompt_ids(tokenizer, record["input"], config)
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    arguments = evaluation_generation_arguments(model, tokenizer, config["evaluation"])
    with torch.inference_mode():
        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **arguments)
    suffix = generated[0, len(ids):].tolist()
    content = suffix[:-1] if suffix and suffix[-1] == tokenizer.eos_token_id else suffix
    result = {"sample_id": record["id"], "input": record["input"], "split": "train",
              "prompt_token_ids": ids, "generated_token_ids": suffix,
              "raw": tokenizer.decode(content, skip_special_tokens=False),
              "generation_effective": arguments["generation_config"].to_dict(),
              "use_model_defaults": False, "dtype": str(next(model.parameters()).dtype),
              "adapter_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})}
    write_json(output / "reload_reference.json", result)


def run_training(config, config_path):
    """Reserve one new output directory and retain diagnostic files on every failure."""
    settings = validate_training_settings(config)
    output = project_path(settings["output_dir"])
    if output.exists():
        raise FileExistsError(f"训练输出已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history = {"microbatches": [], "updates": []}
    summary = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "output_dir": str(output), "training_loop_seconds": None,
               "reload_reference_generation_seconds": None, "test_accessed": False}
    stage = "preflight"
    torch = None
    cuda_device = None
    total_peak_allocated = total_peak_reserved = 0
    loop_started = None
    try:
        config_bytes = project_path(config_path).read_bytes()
        (output / "config.yaml").write_bytes(config_bytes)
        write_json(output / "run_config.json", config)
        summary["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        summary["model"] = config["model"]
        tokenizer, datasets, provenance = prepare_training(config)
        plan = training_plan(config, provenance, run=True)
        summary["plan"] = plan
        write_json(output / "training_summary.json", summary)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        import torch
        import peft
        import transformers
        from src.modeling import attach_lora, choose_device_dtype, load_base_model
        from src.training import (capture_weight_audit, finish_weight_audit, shuffled_batches,
                                  train_epoch, validation_loss, assert_lora_only)

        device, _ = choose_device_dtype(config)
        if device != "cuda":
            raise RuntimeError("正式训练只允许已验证的 CUDA 环境；CPU 仅用于单元测试与 dry-run")
        cuda_device = device
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats(device)
        summary["environment"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                  "peft": peft.__version__, "cuda": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(device)}
        stage = "load_model"
        model, parameter_audit = attach_lora(load_base_model(config), config)
        for adapter_config in model.peft_config.values():
            adapter_config.base_model_name_or_path = config["model"]["id"]
            adapter_config.revision = config["model"]["revision"]
        summary["device"] = str(model.device)
        summary["model_dtype"] = str(next(model.parameters()).dtype)
        summary["adapter_dtypes"] = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
        before = capture_weight_audit(model)
        write_json(output / "lora_audit.json", {"parameter_audit": parameter_audit,
                                                "base_versions_before": before["base_versions"]})
        stage = "validation_before"
        summary["validation_before"] = validation_loss(model, datasets["val"], tokenizer, device)
        parameters = assert_lora_only(model)
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                     betas=tuple(settings["adam_betas"]), eps=settings["adam_epsilon"],
                                     weight_decay=settings["weight_decay"])
        summary["optimizer_parameter_count"] = sum(p.numel() for p in parameters)
        pre_memory = cuda_memory(torch, device)
        total_peak_allocated = pre_memory["peak_allocated_bytes"]
        total_peak_reserved = pre_memory["peak_reserved_bytes"]
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        loop_started = time.perf_counter()
        stage = "training_loop"

        def progress(current):
            write_json(output / "training_history.json", current)
            if current["updates"] and current["updates"][-1]["ending_microbatch"] == len(current["microbatches"]):
                update = current["updates"][-1]
                print(f"update {update['update']}/{plan['optimizer_updates']}: "
                      f"loss={update['mean_microbatch_loss']:.6f}, "
                      f"grad_norm={update['gradient_norm_before_clip']:.6f}", flush=True)

        gradients = train_epoch(
            model, shuffled_batches(datasets["train"], tokenizer, device, config["seed"]),
            microbatch_count=provenance["train"]["count"], optimizer=optimizer,
            accumulation_steps=settings["gradient_accumulation_steps"],
            max_grad_norm=settings["max_grad_norm"], history=history, on_progress=progress)
        torch.cuda.synchronize(device)
        summary["training_loop_seconds"] = time.perf_counter() - loop_started
        summary["training_loop_cuda_memory"] = cuda_memory(torch, device)
        stage = "weight_audit"
        weights = finish_weight_audit(model, before)
        write_json(output / "lora_audit.json", {"parameter_audit": parameter_audit,
                                                "first_gradient_audit": gradients,
                                                "weight_audit": weights})
        stage = "validation_after"
        summary["validation_after"] = validation_loss(model, datasets["val"], tokenizer, device)
        stage = "save_adapter"
        adapter_path = output / "adapter"
        model.save_pretrained(adapter_path, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(adapter_path)
        for filename in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json"):
            if not (adapter_path / filename).is_file():
                raise AssertionError(f"保存后缺少文件: {filename}")
        # Separate a training-input behavioral reference from validation and test evaluation.
        stage = "reload_reference"
        reference_started = time.perf_counter()
        save_reload_reference(model, tokenizer, datasets["train"]["records"][0], config, output)
        torch.cuda.synchronize(device)
        summary["reload_reference_generation_seconds"] = time.perf_counter() - reference_started
        summary.update({"status": "completed", "completed_microbatches": len(history["microbatches"]),
                        "completed_optimizer_updates": len(history["updates"]),
                        "adapter_path": str(adapter_path),
                        "adapter_sha256": hashlib.sha256((adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
                        "reload_verification": "pending_separate_process", "merged_base_model_saved": False})
    except Exception as error:
        if stage == "training_loop" and loop_started is not None:
            summary["training_loop_seconds"] = time.perf_counter() - loop_started
        summary.update({"status": "failed", "failure_stage": stage,
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if torch is not None and cuda_device is not None:
            memory = cuda_memory(torch, cuda_device)
            memory["peak_allocated_bytes"] = max(total_peak_allocated, memory["peak_allocated_bytes"])
            memory["peak_reserved_bytes"] = max(total_peak_reserved, memory["peak_reserved_bytes"])
            summary["total_cuda_memory"] = memory
        summary["total_seconds"] = time.perf_counter() - started
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "training_history.json", history)
        write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--run", action="store_true", help="显式执行固定首轮训练；不覆盖已有输出目录")
    args = parser.parse_args(argv)
    local_cache_environment()
    config = load_config(args.config)
    validate_training_settings(config)
    if args.run:
        return run_training(config, args.config)
    _, _, provenance = prepare_training(config)
    plan = training_plan(config, provenance)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


if __name__ == "__main__":
    main()
