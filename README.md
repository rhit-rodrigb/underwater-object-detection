# Submerged — GitHub Pages demo starter

This is a **static website** starter for the team’s TrashCan underwater object
detection project. It is intended to live at
`https://rhit-rodrigb.github.io/underwater-object-detection/`.

It was built from the supplied `TrashCan-code.zip` and `models.zip`. The team's existing
`demo/predict.py` defines these **four YOLO11s models trained on
TrashCan-Instance**: baseline, built-in augmentation, class-weighted, and
offline augmentation. The second archive contains the **four trained `.pt`
checkpoints** and ten curated examples (original images, ground-truth images,
and saved predictions from all four models). All ten curated examples are
included in this website starter. Real inference still needs ONNX export.

## 1. Copy the site into GitHub (no integration required)

The file **`index.html` must be at the root of the GitHub Pages repository**.
Upload or commit the **contents** of this starter directory—not the containing
`submerged_github_pages_starter` folder—into the root of
`rhit-rodrigb/underwater-object-detection`.

```text
underwater-object-detection/
├── index.html              <-- REQUIRED at repository root
├── app.js
├── styles.css
├── README.md
├── models/
│   ├── README.md
│   ├── baseline.onnx        <-- add after exporting trained weights
│   ├── builtin.onnx         <-- add after exporting trained weights
│   ├── weighted.onnx        <-- add after exporting trained weights
│   └── offline.onnx         <-- add after exporting trained weights
├── samples/
│   ├── manifest.json        <-- 10 included curated examples
│   └── images/             <-- supplied originals, reference boxes, saved predictions
└── tools/
    ├── prepare_examples.py
    └── export_models.py
```

If the repository already has an `index.html`, back it up or inspect it before
replacing it. In the repository's **Settings → Pages**, configure the intended
branch and **`/(root)`** as the publishing source. GitHub Pages updates may take
several minutes to become visible.

## 2. Preview locally

Open a terminal inside this website directory and run:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/`. Do **not** open `index.html` directly
through `file://`: the site fetches JSON and model files and requires HTTP.

You can already select and display a personal image. Select **Dataset sample →
TrashCan-Instance**, pick one of the ten examples, choose a model, and click
**Run detection** to see the **saved** annotated output from your team. The
website explicitly identifies these as saved images, not fresh inference.

On a **personal upload**, Run detection will report missing ONNX model files
until Step 3 is complete. The saved examples contain annotated JPGs but **no
machine-readable detection records or COCO bounding-box annotations**, so
per-object correctness rows are not yet available for those examples.

## 3. Export the actual trained models

The new `models.zip` you provided contains these checkpoints:

- `baseline.pt`
- `builtin.pt`
- `weighted.pt`
- `offline.pt`

Extract `models.zip` somewhere on your computer or the school server, then
run the following using an environment with **Ultralytics and ONNX installed**
from the website directory. Adjust the `--weights-dir` path to wherever you
extracted the `models` folder:

```bash
python tools/export_models.py --weights-dir /path/to/extracted/models
```

To export just one model while others are still being trained:

```bash
python tools/export_models.py --weights-dir /path/to/extracted/models --model baseline
```

This creates `models/baseline.onnx`, etc. The export uses 640×640 inference
and **raw YOLO outputs**; the JavaScript handles confidence thresholds,
non-maximum suppression, and drawing boxes. The models are loaded directly
in each visitor's browser through ONNX Runtime Web. No backend is necessary.

The `.pt` checkpoints are present in your **new** ZIP, but I could not export
or test them here because the environment does not have Ultralytics, PyTorch,
or ONNX installed. After
export, test each model with representative images, compare at least a few
predictions against the existing Python `demo/predict.py`, and confirm the
class names and coordinates agree. Do not replace these weights with a
generic pretrained YOLO model: its output classes would be incompatible.

**File size:** Do not put `.pt` files or whole Python virtual environments in
the Pages repository. Each `.onnx` should be checked with `ls -lh`; GitHub
warns on files over 50 MiB and blocks regular uploads over 100 MiB. If ONNX
files are too large for practical hosting, ask before using an external model
host or smaller quantized model.

## 4. Prepare annotated dataset examples

On the school server, where the **original** dataset images and annotation
JSON files are available, run from the website directory:

```bash
python tools/prepare_examples.py \
  --instance-dir /path/to/TrashCan/data/kaggle/dataset/instance_version \
  --material-dir /path/to/TrashCan/data/kaggle/dataset/material_version \
  --limit 16
```

This copies up to 16 **additional** annotated validation images from each
dataset into `samples/images/` and updates `samples/manifest.json` with their original COCO
bounding boxes, classes, and image filenames. It does not modify the dataset.
The ten curated examples are preserved. It uses only the standard library,
so it doesn't require the YOLO venv.

You may run it with only `--instance-dir` if the Material files are unavailable.
Be mindful of image sizes before committing the sample folder to GitHub; 16
per version is a starting point, not a fixed requirement.

If the school server disconnects on large downloads, create a small ZIP of
just the website assets, exported ONNX files (if they fit), and selected sample
images. There is **no need to download the entire 1.60 GB dataset** just to
build the webpage.

## 5. How the results are interpreted

- **Model selector:** Four variants named in the team’s `demo/predict.py`.
- **Your image:** Browser upload; predicted boxes and classes are shown, but
  actual classes/correctness remain *Not available* (no ground truth).
- **Ten curated TrashCan-Instance samples:** The left image shows your team's
  saved ground-truth image, and the right shows the selected model's saved
  predicted boxes until ONNX inference is available. These saved JPGs do not
  contain extractable confidence and ground-truth records, so the object table
  explains that numerical comparison is not yet possible.
- **Additional COCO-annotated Instance samples:** With ONNX available, a
  predicted box is *Correct* when it overlaps one labeled object with
  IoU ≥ 0.50 **and** matches its category. Wrong class, missed annotations,
  and extra detections are listed separately. This is a per-image
  demonstration, **not an mAP50 evaluation**.
- **TrashCan-Material sample:** The models still predict **Instance** classes
  (22 categories), whereas Material labels use **16 different categories**.
  Both boxes are displayed, but actual/predicted classes are currently marked
  *Not comparable*. Comparing across these two label schemes would require an
  explicit, defensible class mapping or a Material-trained model.
- **Multiple objects:** Each match, missed object, or extra detection appears
  as a separate row in the results table.

## Deployment checklist

- [ ] `index.html`, `styles.css`, `app.js` in GitHub repository root
- [ ] GitHub Pages enabled for the correct branch and root folder
- [ ] At least one **trained** `.onnx` checkpoint exported and uploaded
- [x] Ten curated Instance samples included, with saved results for four models
- [ ] Optional additional annotated samples prepared with `prepare_examples.py`
- [ ] Personal-image upload tested in Chrome/Edge
- [ ] Instance sample tested; ground truth and prediction boxes align
- [ ] All four models tested once weights are available
- [ ] Model outputs cross-checked with `demo/predict.py`
- [ ] Mobile layout checked
- [ ] Team credits, labels, and copy reviewed by the group

## Attribution

The interface and initial browser-inference integration were drafted with
assistance from OpenAI ChatGPT. The trained models, dataset, evaluation work,
and any later code changes should be attributed to their respective authors
and the TrashCan dataset creators in the final course submission.
