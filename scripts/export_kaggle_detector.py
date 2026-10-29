#!/usr/bin/env python3
"""Export a trusted YOLO checkpoint to the fixed CPU serving contract.

Requires the training environment plus onnx==1.23.2; never run in the web venv.
Artifacts stay outside Git. No overwrite of existing versioned models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO
    import onnx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(dir=args.out_dir, prefix="export-") as temporary:
        checkpoint = Path(temporary) / "detector.pt"
        shutil.copyfile(args.weights, checkpoint)
        exported = Path(YOLO(str(checkpoint)).export(format="onnx", imgsz=416, batch=1,
                        dynamic=False, simplify=False, opset=17, nms=False, half=False, device="cpu"))
        graph = onnx.load(exported)
        onnx.checker.check_model(graph)
        inputs = [[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.input]
        outputs = [[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.output]
        if inputs != [[1, 3, 416, 416]] or outputs != [[1, 5, 3549]]:
            raise ValueError("Checkpoint does not match the single-class serving contract")
        digest = hashlib.sha256(exported.read_bytes()).hexdigest()
        destination = args.out_dir / f"kaggle-v6-yolo11n-416-{digest[:16]}.onnx"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(exported, destination)
        destination.chmod(0o444)
        manifest = {"dataset": "piotrstefaskiue/poland-vehicle-license-plate-dataset/versions/6",
                    "source_checkpoint_sha256": source_sha, "onnx_sha256": digest,
                    "onnx_path": str(destination.resolve()), "imgsz": 416, "opset": 17,
                    "input_shape": inputs[0], "output_shape": outputs[0]}
        destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
