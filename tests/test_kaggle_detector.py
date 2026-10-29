from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from webapp import inference


class KaggleDetectorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name) / "model.onnx"
        self.model.write_bytes(b"synthetic model, never loaded as real ONNX")
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.session = MagicMock()
        self.session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 416, 416], type="tensor(float)", name="images")]
        self.session.get_outputs.return_value = [SimpleNamespace(shape=[1, 5, 3549])]
        self.ort = SimpleNamespace(SessionOptions=MagicMock(side_effect=SimpleNamespace),
                                   ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
                                   InferenceSession=MagicMock(return_value=self.session))
        self.enterContext(patch.dict(sys.modules, {"onnxruntime": self.ort}))

    def detector(self):
        return inference.KaggleDetector(str(self.model), self.digest, 0.35)

    @staticmethod
    def output(*rows):
        output = np.zeros((1, 5, 3549), dtype=np.float32)
        for index, row in enumerate(rows):
            output[0, :, index] = row
        return output

    def test_pinned_hash_and_cpu_threads(self):
        self.detector()
        kwargs = self.ort.InferenceSession.call_args.kwargs
        self.assertEqual(kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(kwargs["sess_options"].intra_op_num_threads, 1)
        self.assertEqual(kwargs["sess_options"].inter_op_num_threads, 1)

    def test_missing_file_hash_and_relative_path_fail_closed(self):
        for path, digest in ((str(self.model), "f" * 64), (str(self.model), ""),
                             ("relative.onnx", self.digest), (str(self.model) + ".missing", self.digest)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                inference.KaggleDetector(path, digest, 0.35)
        self.ort.InferenceSession.assert_not_called()

    def test_oversized_model_rejected_before_read(self):
        with self.model.open("wb") as output:
            output.truncate(32 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            self.detector()
        self.ort.InferenceSession.assert_not_called()

    def test_wrong_input_or_output_contract_rejected(self):
        self.session.get_inputs.return_value[0].shape = [1, 3, 640, 640]
        with self.assertRaises(ValueError):
            self.detector()
        self.session.get_inputs.return_value[0].shape = [1, 3, 416, 416]
        self.session.get_outputs.return_value[0].shape = [1, 300, 6]
        with self.assertRaises(ValueError):
            self.detector()

    def test_letterbox_bgr_to_rgb_normalization(self):
        frame = np.zeros((416, 832, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        tensor, gain, left, top = inference.KaggleDetector.preprocess(frame)
        self.assertEqual((gain, left, top), (0.5, 0, 104))
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tensor[0, 0, 208, 208], 1)
        self.assertEqual(tensor[0, 2, 208, 208], 0)
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)

    def test_odd_dimensions_still_produce_square_tensor(self):
        tensor, gain, left, top = inference.KaggleDetector.preprocess(np.zeros((201, 101, 3), dtype=np.uint8))
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(left, 103)
        self.assertEqual(top, 0)

    def test_boxes_restore_original_coordinates(self):
        detections = self.detector().postprocess(self.output([208, 208, 104, 52, .9]), 832, 416, .5, 0, 104)
        self.assertEqual(detections[0][0], [312, 156, 520, 260])
        self.assertAlmostEqual(detections[0][1], .9)

    def test_nms_confidence_and_nonfinite_filtering(self):
        output = self.output([208, 208, 104, 52, .9], [209, 208, 104, 52, .8],
                             [50, 50, 20, 20, .1], [50, 50, 20, 20, float("nan")],
                             [100, 100, -10, 20, .9], [100, 100, 20, 20, 2])
        detections = self.detector().postprocess(output, 416, 416, 1, 0, 0)
        self.assertEqual(len(detections), 1)

    def test_clipping_and_ocr_call_limit(self):
        output = self.output([0, 208, 100, 100, .99], [140, 208, 20, 20, .9],
                             [230, 208, 20, 20, .8], [320, 208, 20, 20, .7])
        detections = self.detector().postprocess(output, 416, 416, 1, 0, 0)
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0][0][0], 0)

    def test_blank_and_degenerate_boxes_return_empty(self):
        output = self.output([208, 10, 30, 10, .9])
        self.assertEqual(self.detector().postprocess(output, 832, 416, .5, 0, 104), [])
        self.assertEqual(self.detector().postprocess(self.output(), 416, 416, 1, 0, 0), [])
        with self.assertRaises(ValueError):
            self.detector().postprocess(np.zeros((1, 6, 3549)), 416, 416, 1, 0, 0)

    def test_predict_returns_fastalpr_compatible_results(self):
        output = self.output([208, 208, 104, 52, .9])
        self.session.run.return_value = [output]
        result_type = SimpleNamespace(from_detection_data=lambda box, confidence, label:
                                      SimpleNamespace(box=box, confidence=confidence, label=label))
        module = SimpleNamespace(DetectionResult=result_type)
        with patch.dict(sys.modules, {"open_image_models.detection.core.base": module}):
            result = self.detector().predict(np.zeros((416, 832, 3), dtype=np.uint8))
        self.assertEqual(result[0].box, (312, 156, 520, 260))
        self.assertEqual(result[0].label, "plate")

    def test_backend_selection_preserves_ocr_and_never_silently_falls_back(self):
        factory = MagicMock()
        with patch.dict(sys.modules, {"fast_alpr": SimpleNamespace(ALPR=factory)}):
            with patch.object(inference, "DETECTOR_BACKEND", "hub"):
                inference.create_alpr()
                self.assertEqual(factory.call_args.kwargs["ocr_model"], inference.OCR_MODEL)
                self.assertNotIn("detector", factory.call_args.kwargs)
            with patch.object(inference, "DETECTOR_BACKEND", "kaggle"):
                with patch.object(inference, "KaggleDetector", return_value="custom"):
                    inference.create_alpr()
                    self.assertEqual(factory.call_args.kwargs["detector"], "custom")
                    self.assertEqual(factory.call_args.kwargs["ocr_device"], "cpu")
                with patch.object(inference, "KaggleDetector", side_effect=ValueError("missing model")):
                    with self.assertRaises(ValueError):
                        inference.create_alpr()
            with patch.object(inference, "DETECTOR_BACKEND", "typo"):
                with self.assertRaises(ValueError):
                    inference.create_alpr()
