#!/usr/bin/env python3
"""Copy a small set of TrashCan COCO validation images into the static demo.

This script reads the *original* COCO annotation JSON and images, not the
YOLO symlink dataset. It does not modify the original TrashCan dataset.
Only Python's standard library is required.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def find_validation_annotations(root):
    matches = []
    for path in root.rglob("*.json"):
        if "val" not in path.stem.lower():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one validation COCO JSON under {root}, found {len(matches)}: "
            f"{[str(path) for path, _ in matches]}"
        )
    return matches[0]


def find_image(root, filename):
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image filename: {filename}")
    for candidate in (root / relative, root / "val" / relative,
                      root / "val" / relative.name, root / relative.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image {filename!r} under {root}")


def valid_annotations(image, annotations, class_names):
    width, height = image["width"], image["height"]
    result = []
    for ann in annotations:
        if ann.get("iscrowd", 0) or ann["category_id"] not in class_names:
            continue
        x, y, w, h = ann["bbox"]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 <= x1 or y2 <= y1:
            continue
        result.append({
            "class": class_names[ann["category_id"]],
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
        })
    return result


def build_samples(kind, root, count, rng, output):
    annotations_path, data = find_validation_annotations(root)
    categories = sorted(data["categories"], key=lambda category: category["id"])
    expected = 22 if kind == "instance" else 16
    if len(categories) != expected:
        raise ValueError(f"Expected {expected} {kind} categories in {annotations_path}; got {len(categories)}")
    class_names = {c["id"]: c["name"] for c in categories}
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    candidates = []
    for image in data["images"]:
        labels = valid_annotations(image, by_image[image["id"]], class_names)
        if labels:
            candidates.append((image, labels))

    # Mix representative objects into the sample set; favor some images with
    # multiple annotated objects so the website's results table is useful.
    rng.shuffle(candidates)
    candidates.sort(key=lambda pair: min(4, len(pair[1])), reverse=True)
    chosen = candidates[:count]
    image_dir = output / "samples" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for index, (image, labels) in enumerate(chosen, 1):
        source = find_image(root, image["file_name"])
        extension = source.suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        filename = f"{kind}_{index:03d}_{source.name}"
        destination = image_dir / filename
        shutil.copy2(source, destination)
        samples.append({
            "id": f"{kind}_{index:03d}",
            "label": f"{kind.title()} #{index:02d} · {source.stem}",
            "dataset": kind,
            "split": "val",
            "image": f"samples/images/{filename}",
            "annotations": labels,
        })
    print(f"{kind}: {len(samples)} examples from {annotations_path}")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, help="Original data/kaggle/dataset/instance_version")
    parser.add_argument("--material-dir", type=Path, help="Original data/kaggle/dataset/material_version")
    parser.add_argument("--limit", type=int, default=16, help="Samples per available dataset (default: 16)")
    parser.add_argument("--seed", type=int, default=416, help="Reproducible selection seed")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Website root with index.html")
    args = parser.parse_args()
    if not args.instance_dir and not args.material_dir:
        parser.error("Specify --instance-dir, --material-dir, or both")
    if args.limit <= 0:
        parser.error("--limit must be positive")

    output = args.output_dir.resolve()
    if not (output / "index.html").exists():
        parser.error(f"Website index.html not found under {output}")
    rng = random.Random(args.seed)
    manifest = output / "samples" / "manifest.json"
    # Preserve the ten curated examples and their saved model predictions.
    existing = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
    samples = [sample for sample in existing if sample.get("split") == "curated"]
    for kind, root in (("instance", args.instance_dir), ("material", args.material_dir)):
        if root is None:
            continue
        root = root.resolve()
        if not root.is_dir():
            parser.error(f"Dataset directory not found: {root}")
        samples.extend(build_samples(kind, root, args.limit, rng, output))

    manifest.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest} with {len(samples)} entries")


if __name__ == "__main__":
    main()
