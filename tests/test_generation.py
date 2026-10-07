"""Exercise the installed public generate API without downloads or training."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from scripts.evaluate import evaluation_generation_arguments


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )).cpu().eval()
        # Reproduce the local checkpoint's sampling defaults and version gate.
        cls.model.generation_config = GenerationConfig(
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
            do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
            transformers_version="4.51.0",
        )
        cls.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
        cls.settings = {"max_new_tokens": 4, "do_sample": False,
                        "num_beams": 1, "repetition_penalty": 1.0}
        cls.inputs = torch.tensor([[1, 4, 8]], dtype=torch.long)

    def generate_and_observe(self, **arguments):
        observed = []
        original_prepare = self.model._prepare_generation_config

        def observe(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            observed.append(result[0].to_dict())
            return result

        # The production code uses public APIs only. This test observes the
        # installed library's resolved config at its real generation boundary.
        with patch.object(self.model, "_prepare_generation_config", side_effect=observe), \
                patch("torch.multinomial", wraps=torch.multinomial) as sampling, \
                torch.inference_mode():
            output = self.model.generate(
                input_ids=self.inputs, attention_mask=torch.ones_like(self.inputs),
                return_dict_in_generate=True, output_scores=True, **arguments,
            )
        return output, observed[0], sampling.call_count

    def test_old_call_inherits_sampling_despite_requested_false(self):
        requested = GenerationConfig(**self.settings, pad_token_id=0, use_cache=True)
        _, effective, sample_calls = self.generate_and_observe(generation_config=requested)
        self.assertFalse(requested.do_sample)
        self.assertTrue(effective["do_sample"])
        self.assertEqual(effective["temperature"], 0.6)
        self.assertGreater(sample_calls, 0)

    def test_explicit_arguments_keep_greedy_and_provenance_matches(self):
        arguments = evaluation_generation_arguments(self.model, self.tokenizer, self.settings)
        saved = arguments["generation_config"].to_dict()
        output, effective, sample_calls = self.generate_and_observe(**arguments)
        self.assertFalse(arguments["use_model_defaults"])
        self.assertFalse(effective["do_sample"])
        self.assertEqual(effective["num_beams"], 1)
        self.assertEqual(effective["repetition_penalty"], 1.0)
        self.assertEqual(effective["temperature"], 1.0)
        self.assertEqual(sample_calls, 0)
        # Output controls are test-only kwargs; every persisted config field
        # otherwise equals what generate actually resolved.
        for key, value in saved.items():
            if key not in ("return_dict_in_generate", "output_scores"):
                self.assertEqual(effective[key], value, key)
        generated = output.sequences[0, self.inputs.shape[1]:]
        self.assertEqual(len(generated), self.settings["max_new_tokens"])
        for token, scores in zip(generated, output.scores):
            self.assertEqual(token.item(), scores[0].argmax().item())
        self.assertTrue(self.model.generation_config.do_sample)


if __name__ == "__main__":
    unittest.main()
