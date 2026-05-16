#!/usr/bin/env python3
"""Run text-prompted semantic segmentation for images and videos in data/."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass
class SegmentationStats:
    input_path: str
    output_path: str
    media_type: str
    prompt: str
    threshold: float
    frames_processed: int
    mean_mask_score: float
    mean_mask_coverage: float
    fps: float
    latency_ms: float
    miou: float | None
    non_target_leakage: float | None
    target_damage: float | None
    true_positive: int | None
    false_positive: int | None
    true_negative: int | None
    false_negative: int | None


class PromptSegmenter:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_mask(self, image: Image.Image, prompt: str) -> np.ndarray:
        """Return a float mask in [0, 1] resized to the original image size."""
        rgb_image = image.convert("RGB")
        inputs = self.processor(
            text=[prompt],
            images=[rgb_image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 3:
            logits = logits.unsqueeze(1)

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(rgb_image.height, rgb_image.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.sigmoid(upsampled)[0, 0].detach().cpu().numpy()
        return mask.astype(np.float32)


def iter_media_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
            yield path


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    heat = np.uint8(np.clip(mask, 0, 1) * 255)
    return cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)


def make_overlay(rgb: np.ndarray, mask: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heatmap = colorize_mask(mask)
    blended = cv2.addWeighted(bgr, 1.0 - alpha, heatmap, alpha, 0)
    binary = mask >= threshold
    return np.where(binary[..., None], blended, bgr)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def binary_confusion_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, tn, fn


def rates_from_counts(tp: int, fp: int, tn: int, fn: int) -> tuple[float | None, float | None]:
    background = fp + tn
    target = tp + fn
    non_target_leakage = float(fp / background) if background > 0 else None
    target_damage = float(fn / target) if target > 0 else None
    return non_target_leakage, target_damage


def read_ground_truth_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read ground-truth mask: {mask_path}")
    width, height = size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def find_image_ground_truth(path: Path, gt_dir: Path | None) -> Path | None:
    if gt_dir is None:
        return None

    relative = path.relative_to(path.parents[0])
    candidates = [
        gt_dir / relative,
        gt_dir / f"{path.stem}_mask.png",
        gt_dir / f"{path.stem}.png",
        gt_dir / path.stem / f"{path.stem}_mask.png",
        gt_dir / path.stem / "mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_video_frame_ground_truth(path: Path, gt_dir: Path | None, frame_index: int) -> Path | None:
    if gt_dir is None:
        return None

    stem = path.stem
    frame_name = f"frame_{frame_index:06d}"
    candidates = [
        gt_dir / stem / f"{frame_name}_mask.png",
        gt_dir / stem / f"{frame_name}.png",
        gt_dir / stem / f"{frame_index:06d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_image(
    segmenter: PromptSegmenter,
    path: Path,
    output_dir: Path,
    gt_dir: Path | None,
    prompt: str,
    threshold: float,
    alpha: float,
) -> SegmentationStats:
    started_at = time.perf_counter()
    image = Image.open(path).convert("RGB")
    infer_started_at = time.perf_counter()
    mask = segmenter.predict_mask(image, prompt)
    latency_ms = (time.perf_counter() - infer_started_at) * 1000.0
    rgb = np.array(image)
    binary = np.uint8(mask >= threshold) * 255
    overlay = make_overlay(rgb, mask, threshold, alpha)

    stem = safe_stem(path)
    image_dir = output_dir / "images" / stem
    image_dir.mkdir(parents=True, exist_ok=True)

    mask_path = image_dir / f"{stem}_mask.png"
    overlay_path = image_dir / f"{stem}_overlay.png"
    heatmap_path = image_dir / f"{stem}_heatmap.png"

    cv2.imwrite(str(mask_path), binary)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(heatmap_path), colorize_mask(mask))

    gt_path = find_image_ground_truth(path, gt_dir)
    miou = None
    non_target_leakage = None
    target_damage = None
    tp = fp = tn = fn = None
    if gt_path is not None:
        gt_mask = read_ground_truth_mask(gt_path, image.size)
        pred_mask = binary > 0
        miou = binary_iou(pred_mask, gt_mask)
        tp, fp, tn, fn = binary_confusion_counts(pred_mask, gt_mask)
        non_target_leakage, target_damage = rates_from_counts(tp, fp, tn, fn)

    elapsed = time.perf_counter() - started_at
    return SegmentationStats(
        input_path=str(path),
        output_path=str(overlay_path),
        media_type="image",
        prompt=prompt,
        threshold=threshold,
        frames_processed=1,
        mean_mask_score=float(mask.mean()),
        mean_mask_coverage=float((mask >= threshold).mean()),
        fps=float(1.0 / elapsed) if elapsed > 0 else 0.0,
        latency_ms=float(latency_ms),
        miou=miou,
        non_target_leakage=non_target_leakage,
        target_damage=target_damage,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def process_video(
    segmenter: PromptSegmenter,
    path: Path,
    output_dir: Path,
    gt_dir: Path | None,
    prompt: str,
    threshold: float,
    alpha: float,
    frame_stride: int,
    preview_frames: int,
) -> SegmentationStats:
    started_at = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = safe_stem(path)
    video_dir = output_dir / "videos" / stem
    frames_dir = video_dir / "preview_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = video_dir / f"{stem}_overlay.mp4"
    mask_path = video_dir / f"{stem}_mask.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (width, height), isColor=False)

    scores: list[float] = []
    coverages: list[float] = []
    latencies_ms: list[float] = []
    ious: list[float] = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    gt_frames = 0
    processed = 0
    frame_index = 0
    last_mask: np.ndarray | None = None

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_index % frame_stride == 0 or last_mask is None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            infer_started_at = time.perf_counter()
            last_mask = segmenter.predict_mask(pil_image, prompt)
            latencies_ms.append((time.perf_counter() - infer_started_at) * 1000.0)
            processed += 1
            scores.append(float(last_mask.mean()))
            coverages.append(float((last_mask >= threshold).mean()))

            gt_path = find_video_frame_ground_truth(path, gt_dir, frame_index)
            if gt_path is not None:
                gt_mask = read_ground_truth_mask(gt_path, (width, height))
                pred_mask = last_mask >= threshold
                ious.append(binary_iou(pred_mask, gt_mask))
                tp, fp, tn, fn = binary_confusion_counts(pred_mask, gt_mask)
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                gt_frames += 1

            if processed <= preview_frames:
                cv2.imwrite(
                    str(frames_dir / f"frame_{frame_index:06d}_overlay.png"),
                    make_overlay(rgb, last_mask, threshold, alpha),
                )
                cv2.imwrite(
                    str(frames_dir / f"frame_{frame_index:06d}_mask.png"),
                    np.uint8(last_mask >= threshold) * 255,
                )

        overlay_writer.write(make_overlay(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), last_mask, threshold, alpha))
        mask_writer.write(np.uint8(last_mask >= threshold) * 255)
        frame_index += 1
        if frame_index % 50 == 0:
            if total_frames > 0:
                print(f"  {path.name}: wrote {frame_index}/{total_frames} frames", flush=True)
            else:
                print(f"  {path.name}: wrote {frame_index} frames", flush=True)

    cap.release()
    overlay_writer.release()
    mask_writer.release()

    elapsed = time.perf_counter() - started_at
    non_target_leakage, target_damage = (
        rates_from_counts(total_tp, total_fp, total_tn, total_fn) if gt_frames > 0 else (None, None)
    )
    return SegmentationStats(
        input_path=str(path),
        output_path=str(overlay_path),
        media_type="video",
        prompt=prompt,
        threshold=threshold,
        frames_processed=processed,
        mean_mask_score=float(np.mean(scores)) if scores else 0.0,
        mean_mask_coverage=float(np.mean(coverages)) if coverages else 0.0,
        fps=float(frame_index / elapsed) if elapsed > 0 else 0.0,
        latency_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        miou=float(np.mean(ious)) if ious else None,
        non_target_leakage=non_target_leakage,
        target_damage=target_damage,
        true_positive=total_tp if gt_frames > 0 else None,
        false_positive=total_fp if gt_frames > 0 else None,
        true_negative=total_tn if gt_frames > 0 else None,
        false_negative=total_fn if gt_frames > 0 else None,
    )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment images and videos in data/ with a text prompt and save visualizations to outputs/."
    )
    parser.add_argument("prompt", help="Text prompt describing the object or region to segment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Optional ground-truth mask directory for leakage/damage metrics. Images use matching names; videos use gt/<video_stem>/frame_000000_mask.png.",
    )
    parser.add_argument("--model", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run segmentation every Nth video frame. Reuses the last mask between segmented frames.",
    )
    parser.add_argument("--preview-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = PromptSegmenter(args.model, args.device)

    stats: list[SegmentationStats] = []
    for path in iter_media_files(args.data_dir):
        suffix = path.suffix.lower()
        guessed_type = mimetypes.guess_type(path)[0] or ""
        print(f"Processing {path} ...", flush=True)
        if suffix in IMAGE_EXTENSIONS or guessed_type.startswith("image/"):
            stats.append(process_image(segmenter, path, args.output_dir, args.gt_dir, args.prompt, args.threshold, args.alpha))
        elif suffix in VIDEO_EXTENSIONS or guessed_type.startswith("video/"):
            stats.append(
                process_video(
                    segmenter,
                    path,
                    args.output_dir,
                    args.gt_dir,
                    args.prompt,
                    args.threshold,
                    args.alpha,
                    args.frame_stride,
                    args.preview_frames,
                )
            )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(item) for item in stats], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(stats)} result(s) to {args.output_dir}")
    print("Core metrics:")
    for item in stats:
        leakage = (
            "N/A (ground-truth mask not found)"
            if item.non_target_leakage is None
            else f"{item.non_target_leakage:.4f}"
        )
        damage = "N/A (ground-truth mask not found)" if item.target_damage is None else f"{item.target_damage:.4f}"
        print(
            f"  {item.input_path}: non_target_leakage={leakage}, "
            f"target_damage={damage}, FPS={item.fps:.2f}, latency={item.latency_ms:.2f} ms",
            flush=True,
        )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
