"""用实际 token 前缀验证 assistant 边界；不以字符串长度猜 token 数。"""
import json

FIELDS = ("name", "address", "email", "question")


class SampleTooLong(ValueError):
    pass


def build_messages(text, config):
    if not isinstance(text, str):
        raise TypeError("input 必须为字符串")
    return [{"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": text}]


def prompt_ids(tokenizer, text, config):
    return tokenizer.apply_chat_template(
        build_messages(text, config), tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )


def encode_example(example, tokenizer, config):
    output = example["output"]
    if not isinstance(output, dict) or set(output) != set(FIELDS):
        raise ValueError("output 必须恰好包含四个字段")
    if any(value is not None and (not isinstance(value, str) or value == "null") for value in output.values()):
        raise ValueError("output 值必须为字符串或真正的 null")
    target = json.dumps({key: output[key] for key in FIELDS}, ensure_ascii=False, separators=(",", ":"))
    messages = build_messages(example["input"], config)
    prefix = prompt_ids(tokenizer, example["input"], config)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": target}], tokenize=True,
        add_generation_prompt=False, enable_thinking=False,
    )
    # BPE 可能在拼接处合并 token。必须核对完整对话的真实 token 前缀，否则拒收。
    if full[:len(prefix)] != prefix:
        raise ValueError("模板 token 前缀不一致，不能安全构造 labels；需重新核验 tokenizer revision")
    suffix = full[len(prefix):]
    eos = tokenizer.eos_token_id
    if eos is None or suffix.count(eos) != 1:
        raise ValueError("目标回复必须有且仅有一个官方 EOS")
    end = len(prefix) + suffix.index(eos) + 1
    # 模板在 EOS 后附加换行；只保留至 EOS，模型学习结束回复而非继续生成模板换行。
    if tokenizer.decode(full[end:], skip_special_tokens=False).strip():
        raise ValueError("EOS 后存在非空白内容")
    input_ids = full[:end]
    decoded_target = tokenizer.decode(input_ids[len(prefix):-1], skip_special_tokens=False)
    if decoded_target != target:
        raise ValueError("目标 JSON token 解码不一致，拒绝错误监督")
    maximum = config["data"]["max_length"]
    if len(input_ids) > maximum:
        raise SampleTooLong(f"{example.get('id', '?')}: {len(input_ids)} tokens > {maximum}; 未截断，样本拒收")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prefix) + input_ids[len(prefix):]}


def pad_examples(examples, pad_token_id):
    """训练右侧动态 padding；attention_mask=0，labels=-100，EOS 仍有监督。"""
    if not examples or pad_token_id is None:
        raise ValueError("需要非空 batch 及有效 pad_token_id")
    width = max(len(item["input_ids"]) for item in examples)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for item in examples:
        size = len(item["input_ids"])
        if any(len(item[key]) != size for key in batch):
            raise ValueError("三个张量长度必须相同")
        if not any(label != -100 for label in item["labels"]):
            raise ValueError("不能训练没有目标 token 的样本")
        for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key].append(item[key] + [padding] * (width - size))
    return batch


def label_trace(example, encoded, tokenizer):
    rows = []
    for index, (token, label, mask) in enumerate(zip(encoded["input_ids"], encoded["labels"], encoded["attention_mask"])):
        rows.append({"position": index, "token_id": token,
                     "token": tokenizer.convert_ids_to_tokens(token),
                     "label": label, "attention_mask": mask,
                     "contributes_to_loss": label != -100})
    return {"sample_id": example["id"], "input": example["input"], "gold": example["output"],
            "prompt_token_count": sum(label == -100 for label in encoded["labels"]),
            "target_token_count": sum(label != -100 for label in encoded["labels"]),
            "supervised_text": tokenizer.decode([x for x in encoded["labels"] if x != -100], skip_special_tokens=False),
            "tokens": rows}
