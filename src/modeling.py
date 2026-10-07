"""只加载官方本地快照；先核对真实模块再注入 LoRA。"""
import json
from .config import project_path


def verified_model_dir(config):
    directory = project_path(config["model"]["local_dir"])
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if (manifest["model_id"], manifest["revision"]) != (config["model"]["id"], config["model"]["revision"]):
        raise ValueError("模型快照与配置的固定 revision 不一致")
    return directory


def load_tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(verified_model_dir(config), local_files_only=True,
                                         trust_remote_code=False, token=False)


def choose_device_dtype(config):
    import torch
    requested = config["model"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device not in ("cuda", "cpu"):
        raise ValueError("本阶段仅支持单设备 cuda 或 cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 Python 环境无法访问 GPU")
    precision = config["model"]["precision"]
    if precision == "auto":
        precision = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if precision not in ("bfloat16", "float32"):
        raise ValueError("本阶段支持 bfloat16 或 float32；不未经验证启用 fp16")
    if precision == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("当前设备不满足本项目 BF16 条件")
    return device, getattr(torch, precision)


def load_base_model(config):
    from transformers import AutoModelForCausalLM
    device, dtype = choose_device_dtype(config)
    model = AutoModelForCausalLM.from_pretrained(
        verified_model_dir(config), local_files_only=True, trust_remote_code=False,
        token=False, use_safetensors=True, dtype=dtype,
        attn_implementation=config["model"]["attn_implementation"],
    ).to(device)
    model.config.use_cache = False
    return model


def attach_lora(base_model, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    settings = config["lora"]
    targets = settings["target_modules"]
    modules = [(name, module) for name, module in base_model.named_modules()
               if name.rsplit(".", 1)[-1] in targets]
    present = {name.rsplit(".", 1)[-1] for name, _ in modules}
    if present != set(targets) or not modules:
        raise ValueError(f"真实模块缺少配置的目标: {set(targets) - present}")
    if any(not isinstance(module, torch.nn.Linear) for _, module in modules):
        raise TypeError("第一版 LoRA 仅接入检查过的 Linear 模块")
    base_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    model = get_peft_model(base_model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        revision=config["model"]["revision"], **settings,
    ))
    audit = parameter_audit(model)
    audit["base_parameters_before_lora"] = base_parameters
    audit["matched_base_modules"] = [name for name, _ in modules]
    audit["module_shapes"] = [{"name": name, "in_features": module.in_features,
                                "out_features": module.out_features} for name, module in modules]
    if audit["unexpected_trainable"] or not audit["trainable_parameters"] or audit["frozen_adapter_names"]:
        raise AssertionError("参数冻结状态不符合 LoRA-only 约束")
    return model, audit


def is_adapter(name):
    return ".lora_A." in name or ".lora_B." in name


def parameter_audit(model):
    params = list(model.named_parameters())
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percent": 100 * trainable / total,
            "unexpected_trainable": [n for n, p in params if p.requires_grad and not is_adapter(n)],
            "frozen_adapter_names": [n for n, p in params if is_adapter(n) and not p.requires_grad],
            "base_all_frozen": all(not p.requires_grad for n, p in params if not is_adapter(n)),
            "adapter_modules": [n for n, m in model.named_modules() if hasattr(m, "lora_A")],
            "trainable_names": [n for n, p in params if p.requires_grad]}


def gradient_audit(model):
    import torch
    adapters = [(n, p) for n, p in model.named_parameters() if is_adapter(n)]
    missing = [n for n, p in adapters if p.grad is None]
    nonfinite = [n for n, p in adapters if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    nonzero = [n for n, p in adapters if p.grad is not None and torch.count_nonzero(p.grad).item() > 0]
    base_gradients = [n for n, p in model.named_parameters() if not is_adapter(n) and p.grad is not None]
    # 默认 B=0 初始化会令第一次 backward 的 A 梯度为零，这是预期行为。
    if missing or nonfinite or base_gradients or not nonzero:
        raise AssertionError(f"梯度异常: missing={missing}, nonfinite={nonfinite}, base={base_gradients}")
    return {"adapter_tensor_count": len(adapters), "missing_gradients": missing,
            "nonfinite_gradients": nonfinite, "nonzero_gradient_tensor_count": len(nonzero),
            "base_with_gradients": base_gradients,
            "note": "默认 LoRA B 初始为零，第一次反传 A 梯度可为零；仍要求所有梯度存在且有限。"}


def load_adapter_for_inference(config, adapter_path):
    """加载并冻结固定基础模型上的本地适配器，用于重载验证和评测。"""
    from peft import PeftConfig, PeftModel
    path = project_path(adapter_path)
    adapter_config = PeftConfig.from_pretrained(path, local_files_only=True, token=False)
    if adapter_config.revision != config["model"]["revision"]:
        raise ValueError("适配器必须记录相同 base revision")
    model = PeftModel.from_pretrained(load_base_model(config), path, is_trainable=False,
                                     local_files_only=True, token=False)
    model.eval()
    return model
