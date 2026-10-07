"""第一步实操：读取一条训练样本，查看真实 token 和 labels；不加载模型权重。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path
from src.modeling import load_tokenizer
from src.preprocessing import encode_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--sample-id", help="可选：只选择训练集中的指定 id，默认第一条")
    args = parser.parse_args()
    local_cache_environment()
    config = load_config(args.config)

    # 只读训练集，练习时不使用测试集来选择样本或调整提示词。
    example = None
    with project_path(config["data"]["train"]).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.sample_id is None or row["id"] == args.sample_id:
                example = row
                break
    if example is None:
        raise ValueError(f"训练集中找不到样本：{args.sample_id!r}")

    print(f"样本 ID：{example['id']}")
    print("\n1. 投诉原文（题目）")
    print(example["input"])
    print("\n2. 标准 JSON（参考答案）")
    print(json.dumps(example["output"], ensure_ascii=False, indent=2))

    # 使用真实项目处理函数，保证练习展示与后续训练采用相同的模板和标签规则。
    tokenizer = load_tokenizer(config)
    encoded = encode_example(example, tokenizer, config)
    labels = encoded["labels"]
    boundary = next(index for index, label in enumerate(labels) if label != -100)

    print("\n3. Tokenizer 和 labels 检查")
    print(f"总 token 数：{len(encoded['input_ids'])}")
    print(f"不计分的前缀 token 数：{boundary}")
    print(f"参与 loss 的答案及结束符 token 数：{len(labels) - boundary}")
    print(f"input_ids 开头：{encoded['input_ids'][:12]}")
    print(f"labels 开头：{labels[:12]}")
    print("\n真正参与 loss 的内容：")
    print(tokenizer.decode([label for label in labels if label != -100],
                           skip_special_tokens=False))

    print("\n题目和答案的边界（位置从 0 开始）：")
    print("位置 | token ID | 单个 token 的解码 | label | 是否计分")
    positions = list(range(max(0, boundary - 2), min(boundary + 3, len(labels))))
    positions.append(len(labels) - 1)
    for index in sorted(set(positions)):
        token_id = encoded["input_ids"][index]
        piece = repr(tokenizer.decode([token_id], skip_special_tokens=False))
        scored = "是" if labels[index] != -100 else "否"
        print(f"{index} | {token_id} | {piece} | {labels[index]} | {scored}")

    print("\n检查完成。参数更新次数：0。")


if __name__ == "__main__":
    main()
