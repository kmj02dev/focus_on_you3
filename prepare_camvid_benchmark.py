#!/usr/bin/env python3
"""Prepare one CamVid video-style segmentation benchmark with GT masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMVID_CLASS_IDS = {
    "sky": 0,
    "building": 1,
    "pole": 2,
    "road": 3,
    "sidewalk": 4,
    "tree": 5,
    "signsymbol": 6,
    "fence": 7,
    "car": 8,
    "pedestrian": 9,
    "bicyclist": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small CamVid benchmark video and binary GT masks.")
    parser.add_argument("--camvid-root", type=Path, default=Path("datasets/CamVid"))
    parser.add_argument("--split-file", default="camvid_test.txt")
    parser.add_argument("--sequence-prefix", default="0001TP")
    parser.add_argument("--class-name", default="road", choices=sorted(CAMVID_CLASS_IDS))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_data"))
    parser.add_argument("--gt-dir", type=Path, default=Path("benchmark_gt"))
    return parser.parse_args()


def load_pairs(camvid_root: Path, split_file: str, sequence_prefix: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    split_path = camvid_root / split_file
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_rel, label_rel = line.split()
        image_path = camvid_root / image_rel
        label_path = camvid_root / label_rel
        if image_path.name.startswith(sequence_prefix):
            pairs.append((image_path, label_path))
    return sorted(pairs, key=lambda pair: pair[0].name)


def resize_rgb(image: np.ndarray, width: int) -> np.ndarray:
    height = round(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, width: int) -> np.ndarray:
    height = round(mask.shape[0] * (width / mask.shape[1]))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    args = parse_args()
    class_id = CAMVID_CLASS_IDS[args.class_name.lower()]
    pairs = load_pairs(args.camvid_root, args.split_file, args.sequence_prefix)
    if not pairs:
        raise RuntimeError(f"No frames found for prefix {args.sequence_prefix!r} in {args.split_file}")
    pairs = pairs[: args.frames]

    video_stem = f"camvid_{args.class_name}_{args.sequence_prefix}"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.gt_dir / video_stem
    mask_dir.mkdir(parents=True, exist_ok=True)

    first_rgb = np.array(Image.open(pairs[0][0]).convert("RGB"))
    first_rgb = resize_rgb(first_rgb, args.width)
    height, width = first_rgb.shape[:2]
    video_path = args.data_dir / f"{video_stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

    frame_records = []
    for frame_index, (image_path, label_path) in enumerate(pairs):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path))
        rgb = resize_rgb(rgb, args.width)
        label = resize_mask(label, args.width)
        binary_mask = np.uint8(label == class_id) * 255

        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mask_path = mask_dir / f"frame_{frame_index:06d}_mask.png"
        cv2.imwrite(str(mask_path), binary_mask)
        frame_records.append(
            {
                "frame": frame_index,
                "image": str(image_path),
                "label": str(label_path),
                "gt_mask": str(mask_path),
                "positive_ratio": float((binary_mask > 0).mean()),
            }
        )

    writer.release()
    metadata = {
        "dataset": "CamVid",
        "source": "https://github.com/lih627/CamVid",
        "video_path": str(video_path),
        "gt_dir": str(args.gt_dir),
        "video_stem": video_stem,
        "class_name": args.class_name,
        "class_id": class_id,
        "frames": len(frame_records),
        "fps": args.fps,
        "width": width,
        "height": height,
        "frame_records": frame_records,
    }
    metadata_path = args.gt_dir / f"{video_stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Video: {video_path}")
    print(f"GT masks: {mask_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
