"""One-epoch LoRA training primitives; no model, optimizer or CUDA work on import."""
import hashlib
import json
import math
import random
import time

from .config import project_path
from .modeling import gradient_audit, is_adapter
from .preprocessing import encode_example, pad_examples


def validate_training_settings(config):
    settings = config["training"]
    if settings["num_train_epochs"] != 1 or settings["per_device_train_batch_size"] != 1:
        raise ValueError("首轮固定为 1 epoch、每设备 batch size 1")
    accumulation = settings["gradient_accumulation_steps"]
    if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation < 1:
        raise ValueError("gradient_accumulation_steps 必须是正整数")
    if settings["optimizer"] != "AdamW":
        raise ValueError("首轮只支持 AdamW")
    if settings["loss_reduction"] != "mean_of_microbatch_token_means":
        raise ValueError("训练目标必须明确为 microbatch token mean loss 的等权平均")
    for key in ("learning_rate", "adam_epsilon", "max_grad_norm"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} 必须是有限正数")
    if not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0:
        raise ValueError("weight_decay 必须是有限非负数")
    if len(settings["adam_betas"]) != 2 or any(not 0 <= beta < 1 for beta in settings["adam_betas"]):
        raise ValueError("adam_betas 必须是两个 [0, 1) 内的数")
    return settings


def prepare_training(config):
    """Validate all train/val labels locally. The test path is never opened."""
    from .modeling import load_tokenizer
    validate_training_settings(config)
    tokenizer = load_tokenizer(config)
    if tokenizer.pad_token_id is None:
        raise ValueError("训练需要明确的 pad_token_id")
    datasets, provenance = {}, {}
    for split in ("train", "val"):
        path = project_path(config["data"][split])
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"{split} 数据集为空")
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} 样本 id 重复")
        encoded = [encode_example(record, tokenizer, config) for record in records]
        datasets[split] = {"records": records, "encoded": encoded}
        provenance[split] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                             "count": len(records), "sample_ids": ids,
                             "max_tokens": max(len(row["input_ids"]) for row in encoded),
                             "target_tokens": sum(sum(x != -100 for x in row["labels"][1:])
                                                  for row in encoded)}
    train_ids = set(provenance["train"]["sample_ids"])
    if train_ids.intersection(provenance["val"]["sample_ids"]):
        raise ValueError("train 与 val 样本 id 不能重叠")
    return tokenizer, datasets, provenance


def training_plan(config, provenance, *, run=False):
    settings = validate_training_settings(config)
    count = provenance["train"]["count"]
    return {"mode": "run" if run else "dry_run", "seed": config["seed"],
            "model": config["model"], "lora": config["lora"], "training": settings,
            "datasets": provenance, "forward_backward_calls": count,
            "optimizer_updates": math.ceil(count / settings["gradient_accumulation_steps"]),
            "validation_forward_calls": 2 * provenance["val"]["count"],
            "test_accessed": False, "output_dir": str(project_path(settings["output_dir"])),
            "loss_semantics": "每条样本先对其被监督 token 求均值；累积组再对各条样本等权平均。最后不足一组按实际条数缩放。"}


def assert_lora_only(model):
    """Reject any newly unfrozen base weight or any base gradient at every step."""
    trainable = []
    for name, parameter in model.named_parameters():
        if is_adapter(name):
            if not parameter.requires_grad:
                raise AssertionError(f"adapter 意外冻结: {name}")
            trainable.append(parameter)
        elif parameter.requires_grad or parameter.grad is not None:
            raise AssertionError(f"基础参数必须冻结且无梯度: {name}")
    if not trainable:
        raise AssertionError("未发现可训练 LoRA 参数")
    return trainable


def capture_weight_audit(model):
    assert_lora_only(model)
    return {"base_versions": {name: parameter._version for name, parameter in model.named_parameters()
                              if not is_adapter(name)},
            "adapter_before": {name: parameter.detach().cpu().clone()
                               for name, parameter in model.named_parameters() if is_adapter(name)}}


def finish_weight_audit(model, before):
    import torch
    assert_lora_only(model)
    current = dict(model.named_parameters())
    if set(current) != set(before["base_versions"]) | set(before["adapter_before"]):
        raise AssertionError("训练前后参数名称发生变化")
    mutated = [name for name, version in before["base_versions"].items()
               if current[name]._version != version]
    if mutated:
        raise AssertionError(f"基础参数版本改变: {mutated}")
    changed, unchanged = [], []
    max_delta = 0.0
    for name, old in before["adapter_before"].items():
        new = current[name].detach().cpu()
        if not torch.isfinite(new).all().item():
            raise FloatingPointError(f"adapter 参数非有限值: {name}")
        (unchanged if torch.equal(old, new) else changed).append(name)
        max_delta = max(max_delta, (new.float() - old.float()).abs().max().item())
    if not changed:
        raise AssertionError("训练后没有任何 LoRA 参数改变")
    return {"base_all_frozen": True, "base_with_gradients": [],
            "base_tensor_count": len(before["base_versions"]),
            "base_version_changes": mutated, "base_versions_before": before["base_versions"],
            "base_versions_after": {name: current[name]._version for name in before["base_versions"]},
            "base_unchanged_evidence": "全部基础参数 _version 训练前后相同；未复制基础模型权重。",
            "adapter_tensor_count": len(before["adapter_before"]),
            "adapter_changed_tensor_count": len(changed), "adapter_changed_names": changed,
            "adapter_unchanged_names": unchanged, "adapter_max_absolute_change": max_delta,
            "adapter_all_finite": True}


def tensor_batch(encoded, pad_token_id, device):
    import torch
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in pad_examples([encoded], pad_token_id).items()}


def shuffled_batches(dataset, tokenizer, device, seed):
    """Iterator keeps only one microbatch on the GPU; shuffle is repeatable."""
    indices = list(range(len(dataset["encoded"])))
    random.Random(seed).shuffle(indices)
    for index in indices:
        yield dataset["records"][index]["id"], tensor_batch(
            dataset["encoded"][index], tokenizer.pad_token_id, device)


def train_epoch(model, batches, *, microbatch_count, optimizer, accumulation_steps,
                max_grad_norm, history, on_progress=None):
    """Train using mean of microbatch means, including a correctly scaled partial group."""
    import torch
    if microbatch_count < 1 or accumulation_steps < 1:
        raise ValueError("microbatch_count 与 accumulation_steps 必须为正数")
    parameters = assert_lora_only(model)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_ids) != len(parameters) or set(optimizer_ids) != {id(p) for p in parameters}:
        raise AssertionError("优化器必须恰好只包含全部可训练 LoRA 参数")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    first_gradient_audit = None
    group_losses = []
    observed_count = 0
    for index, (sample_id, batch) in enumerate(batches):
        if index >= microbatch_count:
            raise AssertionError("microbatch 数超过预先验证的训练样本数")
        group_start = (index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, microbatch_count - group_start)
        loss = model(**batch).loss
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError(f"microbatch {index + 1} loss 非有限标量")
        value = loss.detach().float().item()
        (loss / group_size).backward()
        assert_lora_only(model)
        if first_gradient_audit is None:
            first_gradient_audit = gradient_audit(model)
            history["first_gradient_audit"] = first_gradient_audit
        for parameter in parameters:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"microbatch {index + 1} adapter 梯度缺失或非有限")
        observed_count += 1
        group_losses.append(value)
        history["microbatches"].append({"microbatch": index + 1, "sample_id": sample_id,
                                        "loss": value, "accumulation_group_size": group_size,
                                        "backward_loss_scale": 1.0 / group_size,
                                        "elapsed_seconds": time.perf_counter() - started})
        if (index + 1) % accumulation_steps == 0 or index + 1 == microbatch_count:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            norm_value = float(grad_norm.item())
            if not math.isfinite(norm_value):
                raise FloatingPointError("clip 前梯度范数非有限")
            optimizer.step()
            assert_lora_only(model)
            if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
                raise FloatingPointError("optimizer.step 后 adapter 参数非有限")
            history["updates"].append({"update": len(history["updates"]) + 1,
                                       "ending_microbatch": index + 1,
                                       "microbatch_count": group_size,
                                       "mean_microbatch_loss": sum(group_losses) / group_size,
                                       "gradient_norm_before_clip": norm_value,
                                       "max_grad_norm": max_grad_norm,
                                       "learning_rates": [group["lr"] for group in optimizer.param_groups],
                                       "elapsed_seconds": time.perf_counter() - started})
            group_losses = []
            optimizer.zero_grad(set_to_none=True)
        if on_progress is not None:
            on_progress(history)
    if observed_count != microbatch_count:
        raise AssertionError(f"预期 {microbatch_count} 个 microbatch，实际 {observed_count}")
    return first_gradient_audit


def validation_loss(model, dataset, tokenizer, device):
    import torch
    model.eval()
    losses = []
    with torch.inference_mode():
        for record, encoded in zip(dataset["records"], dataset["encoded"]):
            loss = model(**tensor_batch(encoded, tokenizer.pad_token_id, device)).loss
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError(f"验证 loss 非有限标量: {record['id']}")
            losses.append({"sample_id": record["id"], "loss": loss.float().item()})
    return {"mean_per_example_loss": sum(row["loss"] for row in losses) / len(losses),
            "sample_count": len(losses), "rows": losses,
            "purpose": "固定首轮训练前后诊断；不用于超参或检查点选择。"}
