"""CPU-only coverage for supervised chat-token construction and dynamic padding."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.modeling import load_tokenizer
from src.preprocessing import (
    SampleTooLong,
    build_messages,
    encode_example,
    pad_examples,
    prompt_ids,
)


def example_with_escapes():
    return {
        "id": "escaped-local-fixture",
        "input": "投诉人王\"小雨\"说：路径 C:\\\\服务\\\\工单。\n请处理断网问题。",
        "output": {
            "name": "王\"小雨\"",
            "address": "云岚市\\星桥区\n幻月街 8 号",
            "email": "xiaoyu@example.com",
            "question": "断网问题：\"无法登录\"，请尽快处理。",
        },
    }


class EndingTokenizer:
    """Minimal deterministic tokenizer for rejection paths that need no HF assets."""

    eos_token_id = 9

    def __init__(self, *, mismatch=False, trailing=""):
        self.mismatch = mismatch
        self.trailing = trailing

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.assert_template_arguments(tokenize, enable_thinking)
        if add_generation_prompt:
            return [1, 2]
        return ([7, 2] if self.mismatch else [1, 2]) + [3, self.eos_token_id, 8]

    def assert_template_arguments(self, tokenize, enable_thinking):
        if not tokenize or enable_thinking:
            raise AssertionError("unexpected chat-template invocation")

    def decode(self, ids, skip_special_tokens=False):
        if ids == [3]:
            return json.dumps({"name": None, "address": None, "email": None, "question": None},
                              ensure_ascii=False, separators=(",", ":"))
        if ids == [8]:
            return self.trailing
        return ""


class PaddingTests(unittest.TestCase):
    def test_right_padding_masks_multiple_short_rows_and_preserves_eos_supervision(self):
        rows = [
            {"input_ids": [10, 11, 99], "attention_mask": [1, 1, 1], "labels": [-100, 11, 99]},
            {"input_ids": [20, 99], "attention_mask": [1, 1], "labels": [-100, 99]},
            {"input_ids": [30], "attention_mask": [1], "labels": [30]},
        ]
        batch = pad_examples(rows, pad_token_id=0)
        self.assertEqual(batch["input_ids"], [[10, 11, 99], [20, 99, 0], [30, 0, 0]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        self.assertEqual(batch["labels"], [[-100, 11, 99], [-100, 99, -100], [30, -100, -100]])
        self.assertEqual([row[2] for row in batch["labels"]], [99, -100, -100])


class PreprocessingTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        try:
            cls.tokenizer = load_tokenizer(cls.config)
        except Exception as error:  # A missing snapshot must not hide offline padding coverage.
            raise unittest.SkipTest(f"local Qwen3 tokenizer unavailable: {error}")

    def test_training_and_inference_prefixes_match_and_only_json_eos_is_labeled(self):
        item = example_with_escapes()
        prefix = prompt_ids(self.tokenizer, item["input"], self.config)
        direct_inference_prefix = self.tokenizer.apply_chat_template(
            build_messages(item["input"], self.config), tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        encoded = encode_example(item, self.tokenizer, self.config)
        target = json.dumps(item["output"], ensure_ascii=False, separators=(",", ":"))

        self.assertEqual(prefix, direct_inference_prefix)
        self.assertEqual(encoded["input_ids"][:len(prefix)], prefix)
        self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
        supervised = [token for token in encoded["labels"] if token != -100]
        self.assertEqual(supervised, encoded["input_ids"][len(prefix):])
        self.assertEqual(supervised[-1], self.tokenizer.eos_token_id)
        self.assertEqual(supervised.count(self.tokenizer.eos_token_id), 1)
        self.assertEqual(self.tokenizer.decode(supervised[:-1], skip_special_tokens=False), target)
        self.assertIn("\\n", target)
        self.assertIn("\\\\", target)
        self.assertIn('\\"', target)

    def test_overlong_sample_raises_instead_of_silent_truncation(self):
        item = example_with_escapes()
        config = deepcopy(self.config)
        config["data"]["max_length"] = 1
        with self.assertRaisesRegex(SampleTooLong, r"未截断"):
            encode_example(item, self.tokenizer, config)


class PreprocessingRejectionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.item = {
            "id": "rejection-fixture",
            "input": "虚构投诉",
            "output": {"name": None, "address": None, "email": None, "question": None},
        }

    def test_rejects_template_token_prefix_mismatch(self):
        with self.assertRaisesRegex(ValueError, "前缀不一致"):
            encode_example(self.item, EndingTokenizer(mismatch=True), self.config)

    def test_rejects_nonempty_special_token_after_official_eos(self):
        with self.assertRaisesRegex(ValueError, "EOS 后存在非空白"):
            encode_example(self.item, EndingTokenizer(trailing="<|im_end|>"), self.config)
