"""在新进程验证保存的 LoRA/Tokenizer，并复现训练结束时的生成结果。"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path, write_json
from src.modeling import load_adapter_for_inference, load_tokenizer
from src.preprocessing import prompt_ids
from scripts.evaluate import evaluation_generation_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--reference", help="默认读取 adapter 上一级的 reload_reference.json")
    parser.add_argument("--output", help="默认写 adapter 上一级的 reload_verification.json")
    args = parser.parse_args()
    local_cache_environment()
    config = load_config(args.config)
    adapter = project_path(args.adapter)
    reference_path = project_path(args.reference) if args.reference else adapter.parent / "reload_reference.json"
    output_path = project_path(args.output) if args.output else adapter.parent / "reload_verification.json"
    if output_path.exists():
        raise FileExistsError(f"保留现有验证结果，请指定新的 --output：{output_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    report = {"status": "running", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "adapter": str(adapter), "reference": str(reference_path),
              "model_revision": config["model"]["revision"], "optimizer_steps": 0}
    try:
        import torch
        from transformers import AutoTokenizer, set_seed
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        set_seed(config["seed"])
        official_tokenizer = load_tokenizer(config)
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, token=False,
                                                   trust_remote_code=False)
        if (tokenizer.get_vocab() != official_tokenizer.get_vocab()
                or tokenizer.chat_template != official_tokenizer.chat_template
                or tokenizer.eos_token_id != official_tokenizer.eos_token_id
                or tokenizer.pad_token_id != official_tokenizer.pad_token_id):
            raise AssertionError("保存的 Tokenizer/模板与固定基础模型不一致")
        report["tokenizer_roundtrip"] = "passed"
        model = load_adapter_for_inference(config, adapter)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("推理重载后所有参数应冻结")
        report["all_parameters_frozen"] = True

        # 对比全部适配器张量，不能只根据文件存在来声称加载成功。
        expected = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
        loaded = get_peft_model_state_dict(model)
        if set(expected) != set(loaded):
            raise AssertionError("重载的适配器参数名称不一致")
        mismatch = [name for name in expected
                    if not torch.equal(expected[name], loaded[name].detach().cpu())]
        if mismatch:
            raise AssertionError(f"重载适配器参数值不一致：{mismatch[:5]}")
        report["adapter_tensor_count"] = len(expected)
        report["adapter_tensors_exact_match"] = True
        ids = prompt_ids(tokenizer, reference["input"], config)
        if ids != reference["prompt_token_ids"]:
            raise AssertionError("重载后的 prompt 编码与内存模型参考不一致")
        dtype = str(next(model.parameters()).dtype)
        if dtype != reference["dtype"]:
            raise AssertionError("生成比较所用精度不一致")
        arguments = evaluation_generation_arguments(model, tokenizer, config["evaluation"])
        effective = arguments["generation_config"].to_dict()
        if effective != reference["generation_effective"]:
            raise AssertionError("重载前后的实际生成配置不一致")
        inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **arguments)
        suffix = output[0, len(ids):].tolist()
        if suffix != reference["generated_token_ids"]:
            raise AssertionError("保存前后的生成 token 不完全一致")
        content = suffix[:-1] if suffix and suffix[-1] == tokenizer.eos_token_id else suffix
        raw = tokenizer.decode(content, skip_special_tokens=False)
        if raw != reference["raw"]:
            raise AssertionError("保存前后的原始回答不一致")
        report.update({"status": "passed", "sample_id": reference["sample_id"],
                       "generated_token_ids_exact_match": True, "raw_exact_match": True,
                       "raw": raw, "device": str(model.device), "dtype": dtype,
                       "generation_mode": arguments["generation_config"].get_generation_mode().value})
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.perf_counter() - started})
        write_json(output_path, report)
        raise
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
