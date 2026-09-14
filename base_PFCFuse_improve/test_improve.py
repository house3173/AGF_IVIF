import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

from .base_PFCFuse import clear_model_cache, pfcfuse_fuse_gray
from .data import CachedPairs, paired_files, prepare_pair
from .evaluation import eligible, evaluate, paired_intervals
from .fusion_model import DEFAULT_CHECKPOINT, FORMAT, ResidualFusion, load_model
from .objective import FusionObjective
from .train_residual import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class ImproveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(3407)

    def setUp(self):
        self.visible = torch.rand(1, 1, 16, 20)
        self.infrared = torch.rand_like(self.visible)

    def batch(self):
        return dict(visible=self.visible, infrared=self.infrared,
                    base_visible=self.visible, base_infrared=self.infrared,
                    detail=torch.zeros_like(self.visible), id=["fixture"])

    def test_identity_against_independent_original_net(self):
        reference_path = ROOT / "base_PFCFuse_original/net.py"
        if not reference_path.exists():
            self.skipTest(f"{reference_path} not deployed alongside base_PFCFuse_improve; skipping cross-check")
        spec = importlib.util.spec_from_file_location("original_reference", reference_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        modules = {
            "DIDF_Encoder": original.Restormer_Encoder(),
            "DIDF_Decoder": original.Restormer_Decoder(),
            "BaseFuseLayer": original.BaseFeatureExtraction(dim=64),
            "DetailFuseLayer": original.DetailFeatureExtraction(num_layers=1),
        }
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        for name, module in modules.items():
            state = {key[len("module."):] if key.startswith("module.") else key: value
                     for key, value in checkpoint[name].items()}
            module.load_state_dict(state)
            module.eval()
        with torch.no_grad():
            visible_base, visible_detail, _ = modules["DIDF_Encoder"](self.visible)
            infrared_base, infrared_detail, _ = modules["DIDF_Encoder"](self.infrared)
            result, _ = modules["DIDF_Decoder"](
                self.visible, modules["BaseFuseLayer"](visible_base + infrared_base),
                modules["DetailFuseLayer"](visible_detail + infrared_detail))
            candidate = load_model()(self.visible, self.infrared)
        torch.testing.assert_close(candidate, result, rtol=0, atol=0)

    def test_residual_bound_and_equal_inputs(self):
        adapter = ResidualFusion(dim=4, hidden=8)
        with torch.no_grad():
            adapter.output.weight.normal_()
            adapter.output.bias.normal_()
        visible, infrared = torch.randn(2, 4, 9, 11), torch.randn(2, 4, 9, 11)
        delta = adapter(visible, infrared) - (visible + infrared)
        self.assertTrue(torch.all(delta.abs() <= 0.25 * (visible - infrared).abs() + 1e-6))
        torch.testing.assert_close(adapter(visible, visible), visible + visible, rtol=0, atol=0)

    def test_only_adapters_update(self):
        model = load_model().train()
        frozen = {name: parameter.detach().clone() for name, parameter in model.backbone.named_parameters()}
        optimizer = torch.optim.Adam(model.adapters.parameters(), lr=1e-3)
        with torch.no_grad():
            reference = model(self.visible, self.infrared, baseline=True)
        candidate = model(self.visible, self.infrared)
        loss = FusionObjective()(candidate, self.batch(), reference)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertGreater(model.adapters["base"].output.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.adapters["detail"].output.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in model.backbone.named_parameters():
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model(self.visible, self.infrared), reference))

    def test_round_trip_and_strict_loading(self):
        model = load_model()
        with torch.no_grad():
            model.adapters["base"].output.bias.fill_(0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(path, model, {"mode": "raw"}, 0, {})
            restored = load_model(path)
            with torch.no_grad():
                torch.testing.assert_close(model(self.visible, self.infrared),
                                           restored(self.visible, self.infrared), rtol=0, atol=0)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["model"].pop(next(iter(checkpoint["model"])))
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_model(path)

    def test_wrapper_shape_range_and_invalid_inputs(self):
        visible, infrared = self.visible[0, 0].numpy(), self.infrared[0, 0].numpy()
        result = pfcfuse_fuse_gray(infrared, visible, device="cpu")
        self.assertEqual(result.shape, visible.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1)
        for invalid in (visible * 255, visible[:4], visible[None], visible * np.nan):
            with self.assertRaises(ValueError):
                pfcfuse_fuse_gray(invalid, visible)
        with self.assertRaises(FileNotFoundError):
            pfcfuse_fuse_gray(infrared, visible, ckpt_path="nonexistent-improved-checkpoint.pth")
        clear_model_cache()

    def test_evaluation_identity_and_gate(self):
        means, rows = evaluate(load_model(), [self.batch()], "cpu")
        self.assertFalse(eligible(means))
        for result in paired_intervals(rows).values():
            self.assertEqual(result["delta"], 0)
            self.assertEqual(result["ci95"], [0, 0])
        improved = dict(means)
        improved["improved_Qabf"] += 0.01
        improved["improved_loss"] -= 0.01
        self.assertTrue(eligible(improved))
        improved["improved_SSIM"] -= 0.01
        self.assertFalse(eligible(improved))

    def test_constant_loss_is_finite(self):
        for value in (0.0, 1.0):
            image = torch.full((2, 1, 16, 20), value)
            candidate = image.clone().requires_grad_()
            batch = dict(visible=image, infrared=image, detail=torch.zeros_like(image))
            loss = FusionObjective()(candidate, batch, image)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(candidate.grad).all())

    def test_finetune_fusion_lr_unfreezes_only_those_two_layers(self):
        model = load_model()
        for name, adapter in model.adapters.items():
            adapter.requires_grad_(True)
        fusion_layers = [model.backbone["BaseFuseLayer"], model.backbone["DetailFuseLayer"]]
        for layer in fusion_layers:
            layer.requires_grad_(True)
        frozen_names = {"DIDF_Encoder", "DIDF_Decoder"}
        for name, module in model.backbone.items():
            expected = name not in frozen_names
            for parameter in module.parameters():
                self.assertEqual(parameter.requires_grad, expected, name)
        with torch.no_grad():
            reference = model(self.visible, self.infrared, baseline=True)
        candidate = model(self.visible, self.infrared)
        FusionObjective()(candidate, self.batch(), reference).backward()
        for layer in fusion_layers:
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for parameter in layer.parameters()))
        for name in frozen_names:
            self.assertTrue(all(parameter.grad is None for parameter in model.backbone[name].parameters()))

    def test_cli_training_agf_evaluation_and_leakage(self):
        def run(module, *arguments, succeeds=True):
            result = subprocess.run([sys.executable, "-m", f"base_PFCFuse_improve.{module}",
                                     *map(str, arguments)], cwd=ROOT, capture_output=True, text=True, timeout=180)
            if succeeds:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for modality in ("vi", "ir"):
                (source / modality).mkdir(parents=True)
                for index in range(4):
                    generator = np.random.default_rng(index + (100 if modality == "ir" else 0))
                    cv2.imwrite(str(source / modality / f"{index}.png"),
                                generator.integers(0, 256, (24, 28), dtype=np.uint8))
            self.assertEqual(len(paired_files(source)), 4)
            pair = prepare_pair(source / "vi/0.png", source / "ir/0.png", "agf")
            self.assertEqual(pair["detail"].shape, (24, 28))
            self.assertTrue(all(np.isfinite(image).all() for image in pair.values()))
            cache, trained = root / "cache", root / "trained"
            run("data", "--source", source, "--output", cache, "--mode", "agf")
            self.assertEqual(len(CachedPairs(cache, "train")), 3)
            self.assertEqual(len(CachedPairs(cache, "val")), 1)
            run("train_residual", "--cache", cache, "--output", trained, "--epochs", 1,
                "--batch-size", 2, "--crop", 16, "--max-train-batches", 1, "--threads", 2, "--device", "cpu")
            selected = torch.load(trained / "selected.pth", weights_only=True)
            self.assertEqual(selected["format"], FORMAT)
            self.assertTrue((trained / "last.pth").exists())
            finetuned = root / "trained_finetune"
            run("train_residual", "--cache", cache, "--output", finetuned, "--epochs", 1,
                "--batch-size", 2, "--crop", 16, "--max-train-batches", 1, "--threads", 2,
                "--device", "cpu", "--finetune-fusion-lr", 1e-5)
            self.assertTrue((finetuned / "last.pth").exists())
            run("evaluation", "--cache", cache, "--checkpoint", trained / "selected.pth",
                "--split", "val", "--output", root / "report", "--device", "cpu", "--threads", 2)
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["evidence"], "not_established")
            run("infer", "--source", source, "--checkpoint", trained / "selected.pth",
                "--output", root / "fused", "--limit", 1, "--threads", 2, "--device", "cpu")
            output = cv2.imread(str(root / "fused/0.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(output.shape, (24, 28))
            test_cache = root / "test_cache"
            run("data", "--source", source, "--output", test_cache, "--split", "test", "--mode", "agf")
            result = run("evaluation", "--cache", test_cache, "--checkpoint", trained / "selected.pth",
                         "--output", root / "leakage", "--device", "cpu", succeeds=False)
            self.assertIn("overlaps", result.stderr)


if __name__ == "__main__":
    unittest.main()