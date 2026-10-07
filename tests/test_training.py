"""CPU checks of real autograd/AdamW accumulation, frozen weights and safe dry-run."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train
from src.config import load_config
from src.training import capture_weight_audit, finish_weight_audit, train_epoch


class TinyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Module()
        self.proj.base_layer = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        self.proj.base_layer.requires_grad_(False)
        self.proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)})
        self.proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)})
        torch.nn.init.zeros_(self.proj.lora_B["default"].weight)

    def forward(self, input_ids, labels):
        prediction = self.proj.base_layer(input_ids) + self.proj.lora_B["default"](
            self.proj.lora_A["default"](input_ids))
        return SimpleNamespace(loss=torch.nn.functional.mse_loss(prediction, labels))


class TrainingTests(unittest.TestCase):
    def test_accumulation_matches_large_batches_including_partial_final_group(self):
        torch.manual_seed(42)
        model = TinyLora()
        large_batch_model = deepcopy(model)
        inputs = torch.tensor([[1, 2], [3, 1], [-1, 2], [2, -2], [1, -3], [4, 2]], dtype=torch.float64)
        labels = torch.tensor([[1], [-1], [2], [0], [3], [-2]], dtype=torch.float64)
        parameters = [p for p in model.parameters() if p.requires_grad]
        reference_parameters = [p for p in large_batch_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
        reference_optimizer = torch.optim.AdamW(reference_parameters, lr=0.01, weight_decay=0)
        before = capture_weight_audit(model)
        base_before = model.proj.base_layer.weight.detach().clone()
        history = {"microbatches": [], "updates": []}
        batches = [(str(i), {"input_ids": inputs[i:i + 1], "labels": labels[i:i + 1]})
                   for i in range(len(inputs))]

        gradients = train_epoch(model, batches, microbatch_count=6, optimizer=optimizer,
                                accumulation_steps=4, max_grad_norm=1.0, history=history)
        for begin, end in ((0, 4), (4, 6)):
            reference_optimizer.zero_grad(set_to_none=True)
            large_batch_model(input_ids=inputs[begin:end], labels=labels[begin:end]).loss.backward()
            torch.nn.utils.clip_grad_norm_(reference_parameters, 1.0, error_if_nonfinite=True)
            reference_optimizer.step()

        for parameter, reference in zip(model.parameters(), large_batch_model.parameters()):
            torch.testing.assert_close(parameter, reference, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(model.proj.base_layer.weight, base_before))
        audit = finish_weight_audit(model, before)
        self.assertEqual(audit["base_version_changes"], [])
        self.assertGreater(audit["adapter_changed_tensor_count"], 0)
        self.assertEqual(gradients["base_with_gradients"], [])
        self.assertEqual(len(history["microbatches"]), 6)
        self.assertEqual([row["microbatch_count"] for row in history["updates"]], [4, 2])
        self.assertEqual([row["backward_loss_scale"] for row in history["microbatches"]],
                         [0.25, 0.25, 0.25, 0.25, 0.5, 0.5])

    def test_rejects_optimizer_that_includes_base_weights_before_any_forward(self):
        model = TinyLora()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with patch.object(model, "forward", side_effect=AssertionError("must not run")) as forward:
            with self.assertRaisesRegex(AssertionError, "优化器必须"):
                train_epoch(model, [], microbatch_count=1, optimizer=optimizer,
                            accumulation_steps=1, max_grad_norm=1.0,
                            history={"microbatches": [], "updates": []})
        forward.assert_not_called()

    def test_dry_run_validates_local_train_val_without_model_optimizer_or_output(self):
        config = deepcopy(load_config())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            config["training"]["output_dir"] = str(output)
            # An unreadable test path proves the dry-run uses only train and val.
            config["data"]["test"] = str(Path(temporary) / "no-test-file.jsonl")
            with (patch.object(train, "load_config", return_value=config),
                  patch.object(train, "run_training", side_effect=AssertionError("must not train")) as run,
                  patch("src.modeling.load_base_model", side_effect=AssertionError("must not load model")) as load,
                  patch("torch.optim.AdamW", side_effect=AssertionError("must not create optimizer")) as optimizer,
                  redirect_stdout(io.StringIO())):
                plan = train.main([])
            self.assertFalse(output.exists())
            self.assertEqual(plan["forward_backward_calls"], 60)
            self.assertEqual(plan["optimizer_updates"], 15)
            self.assertEqual(plan["validation_forward_calls"], 40)
            self.assertEqual(set(plan["datasets"]), {"train", "val"})
            self.assertFalse(plan["test_accessed"])
            run.assert_not_called()
            load.assert_not_called()
            optimizer.assert_not_called()

    def test_existing_output_is_refused_without_modifying_evidence(self):
        config = deepcopy(load_config())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evidence = directory / "evidence.txt"
            evidence.write_text("preserve", encoding="utf-8")
            config["training"]["output_dir"] = str(directory)
            with patch.object(train, "prepare_training", side_effect=AssertionError("must not prepare")) as prepare:
                with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                    train.run_training(config, "configs/lora.yaml")
            prepare.assert_not_called()
            self.assertEqual(evidence.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(directory.iterdir()), [evidence])


if __name__ == "__main__":
    unittest.main()
