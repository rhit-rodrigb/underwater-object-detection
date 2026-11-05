# Model files go here

The webpage expects these four ONNX files, exported from your **trained**
TrashCan-Instance YOLO11s checkpoints (not a generic pretrained model):

| Web dropdown | File |
| --- | --- |
| Baseline | `baseline.onnx` |
| Built-in augmentation | `builtin.onnx` |
| Class-weighted | `weighted.onnx` |
| Offline augmentation | `offline.onnx` |

Your newer `models.zip` contains all four original `.pt` checkpoints.
Those `.pt` files are deliberately **not** included in the GitHub Pages
website package, because a browser cannot run PyTorch checkpoints directly.
Use `tools/export_models.py` with the extracted weights and Ultralytics to
create these `.onnx` files.

Once added, these files are downloaded directly by the visitor's browser.
No Python server runs on GitHub Pages.
