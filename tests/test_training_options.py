"""Check CPU/GPU CLI wiring without importing torch or downloading weights."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_detector


class TrainingOptionsTest(unittest.TestCase):
    def run_training(self, options=(), resume=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "runs" / "detector" / "weights"
            weights.mkdir(parents=True)
            if resume:
                (weights / "last.pt").touch()
            factory = MagicMock()
            argv = ["train_detector.py", "--data", str(root / "dataset.yaml"),
                    "--project", str(root / "runs"), "--name", "detector", *options]
            with patch.object(sys, "argv", argv), patch.dict(
                sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}
            ):
                train_detector.main()
            self.assertTrue((root / "runs" / "detector-status.json").is_file())
            return factory.return_value.train.call_args.kwargs

    def test_cpu_defaults_are_conservative(self):
        args = self.run_training()
        self.assertEqual(args["device"], "cpu")
        self.assertEqual(args["batch"], 1)
        self.assertFalse(args["amp"])
        self.assertEqual(args["workers"], 0)
        self.assertFalse(args["cache"])

    def test_gpu_options(self):
        args = self.run_training(["--device", "0", "--batch", "8"])
        self.assertEqual(args["device"], "0")
        self.assertEqual(args["batch"], 8)
        self.assertTrue(args["amp"])

    def test_resume_sets_device_and_batch(self):
        args = self.run_training(["--resume", "--device", "0", "--batch", "8"], resume=True)
        self.assertEqual(args, {"resume": True, "device": "0", "batch": 8})
