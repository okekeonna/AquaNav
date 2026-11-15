"""Run the trained RF-DETR crab detector on a test image.

The checkpoint (models/weights.pt) is a Roboflow RF-DETR Small model with 3
classes, but it does not store the class names. To label detections, the
script first runs the model on named reference images (tests/<species>.png) to
learn which class id corresponds to which species.

Usage:
    python src/test_model.py
    python src/test_model.py --image tests/mycrabs.png --conf 0.4 --show
    python src/test_model.py --classes "1=Green crab,2=Jonah crab,3=Rock crab"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reference images, each showing one known species. The class name is taken
# from the filename, e.g. "european_green_crab.png" -> "European green crab".
# Transparent backgrounds are flattened onto white to match the test sheet.
REFERENCE_IMAGES = [
    "tests/european_green_crab.png",
    "tests/jonah_crab.png",
    "tests/rock_crab.png",
]
CALIBRATION_THRESHOLD = 0.05


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        sys.exit(f"Error: image not found: {path}")
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def species_name(path: str) -> str:
    return Path(path).stem.replace("_", " ").capitalize()


def parse_classes(spec: str) -> dict[int, str]:
    names = {}
    for item in spec.split(","):
        key, _, value = item.partition("=")
        if not value:
            sys.exit(f"Error: bad --classes entry '{item}', expected id=name")
        names[int(key.strip())] = value.strip()
    return names


def calibrate(model) -> dict[int, str]:
    """Map class ids to species names using the reference images.

    Each reference image shows one known species. For every (reference, class)
    pair we take the best detection confidence, then assign greedily from the
    most confident pair down, so each species gets a distinct class id.
    """
    print("Calibrating class names from reference images:")
    scores: list[tuple[float, int, str, str]] = []  # (conf, class_id, species, path)
    for rel_path in REFERENCE_IMAGES:
        species = species_name(rel_path)
        path = resolve(rel_path)
        if not path.is_file():
            print(f"  warning: {rel_path} missing, skipping")
            continue
        dets = model.predict(load_image(path), threshold=CALIBRATION_THRESHOLD)
        if len(dets) == 0:
            print(f"  warning: no detection on {rel_path} ({species})")
            continue
        per_class: dict[int, float] = {}
        for cid, conf in zip(dets.class_id, dets.confidence):
            per_class[int(cid)] = max(per_class.get(int(cid), 0.0), float(conf))
        ranked = ", ".join(f"{c}:{v:.2f}" for c, v in
                           sorted(per_class.items(), key=lambda kv: -kv[1]))
        print(f"  {rel_path:<30} ({species}) class scores -> {ranked}")
        scores += [(conf, cid, species, rel_path) for cid, conf in per_class.items()]

    names: dict[int, str] = {}
    for conf, cid, species, rel_path in sorted(scores, reverse=True):
        if cid in names or species in names.values():
            continue
        names[cid] = species
    for species in {species_name(r) for r in REFERENCE_IMAGES} - set(names.values()):
        print(f"  warning: could not assign a class id to '{species}'")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--image", default="tests/mycrabs.png")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--out", default="runs/mycrabs_pred.png")
    parser.add_argument("--classes", help='manual map, e.g. "0=a,1=b,2=c"')
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip reference-image calibration")
    parser.add_argument("--show", action="store_true", help="display the result")
    args = parser.parse_args()

    weights = resolve(args.weights)
    if not weights.is_file():
        sys.exit(f"Error: weights not found: {weights}")
    image = load_image(resolve(args.image))

    # Heavy imports after argument checks so errors show up fast.
    import supervision as sv
    from rfdetr import RFDETRSmall

    print(f"Loading model from {weights} ...")
    model = RFDETRSmall(pretrain_weights=str(weights), num_classes=3)

    if args.classes:
        names = parse_classes(args.classes)
    elif args.no_calibrate:
        names = {}
    else:
        names = calibrate(model)
    print("Class map:", {k: names[k] for k in sorted(names)} or "(none, using ids)")

    def label_of(class_id: int) -> str:
        return names.get(int(class_id), f"class_{int(class_id)}")

    detections = model.predict(image, threshold=args.conf)

    print(f"\n{len(detections)} detection(s) in {args.image} (conf >= {args.conf}):")
    for i, (box, cid, conf) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence), 1):
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"  {i:2d}. {label_of(cid):<22} {conf:.2f}  box=({x1}, {y1}, {x2}, {y2})")
    counts = Counter(label_of(c) for c in detections.class_id)
    for name, n in counts.most_common():
        print(f"  total {name}: {n}")

    labels = [f"{label_of(c)} {s:.2f}"
              for c, s in zip(detections.class_id, detections.confidence)]
    annotated = image.copy()
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    annotated = sv.LabelAnnotator(text_scale=0.5, smart_position=True).annotate(annotated, detections, labels)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out)
    print(f"\nSaved annotated image to {out}")

    if args.show:
        import cv2
        cv2.imshow("Crab detections", cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
