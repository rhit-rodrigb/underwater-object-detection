#!/usr/bin/env python3
"""Export the team's trained YOLO11s .pt checkpoints to browser-ready ONNX files.

Run inside the team's Ultralytics Python environment, on a machine with the
four .pt checkpoints. No GPU is needed for export.
"""

import argparse
import shutil
from pathlib import Path


MODEL_NAMES = ("baseline", "builtin", "weighted", "offline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Directory containing baseline.pt, builtin.pt, weighted.pt, offline.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models",
                        help="Directory for resulting ONNX files (default: website/models)")
    parser.add_argument("--model", choices=("all",) + MODEL_NAMES, default="all",
                        help="Export one model or all four")
    args = parser.parse_args()

    names = MODEL_NAMES if args.model == "all" else (args.model,)
    checkpoints = {name: args.weights_dir.resolve() / f"{name}.pt" for name in names}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        parser.error(f"Missing trained weights: {', '.join(missing)}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error(f"Ultralytics is not installed in this Python environment: {error}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, source in checkpoints.items():
        print(f"Exporting {name}: {source}", flush=True)
        model = YOLO(str(source))
        if len(model.names) != 22:
            raise ValueError(f"{name}: expected 22 TrashCan-Instance classes, got {len(model.names)}")

        # Raw YOLO outputs are handled by app.js (which performs per-class NMS).
        # Avoiding ONNX-embedded NMS improves compatibility with ONNX Runtime Web.
        exported = Path(model.export(format="onnx", imgsz=640, dynamic=False,
                                     nms=False, half=False, opset=17))
        target = output / f"{name}.onnx"
        if exported.resolve() != target:
            shutil.copy2(exported, target)
        print(f"  → {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
