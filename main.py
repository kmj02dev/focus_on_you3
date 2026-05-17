#!/usr/bin/env python3
"""PySide6 single-file MVP for real-time prompt-based video filtering.

Install:
    pip install -r requirements.txt

Run:
    python main.py

The MVP keeps all application code in this file. It supports camera/video input,
text-prompted segmentation, non-target blur/removal, live FPS/latency metrics,
parameter sweeps, and a CamVid road GT benchmark when benchmark_data/ and
benchmark_gt/ are present.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

try:
    from PySide6.QtCore import Qt, QThread, Signal, Slot
    from PySide6.QtGui import QAction, QImage, QPixmap
    from PySide6.QtWidgets import (
        QAbstractScrollArea,
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSizePolicy,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - only used when dependency is missing.
    print("PySide6 is required. Install it with: pip install PySide6", file=sys.stderr)
    raise


CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"
BACKEND_CLIPSEG = "CLIPSeg"
BACKEND_YOLO_WORLD_BOX = "YOLO-World small box-only"
BACKEND_YOLO11N_SEG = "YOLO11n-seg"
BACKEND_YOLO_WORLD_SAM2 = "YOLO-World small + SAM2 tiny tracking"
BACKEND_SAM3 = "SAM 3"
BACKEND_GDINO_SAM2 = "Grounding DINO tiny + SAM2 tiny"
BACKEND_PRESETS = [
    BACKEND_CLIPSEG,
    BACKEND_YOLO_WORLD_BOX,
    BACKEND_YOLO11N_SEG,
    BACKEND_YOLO_WORLD_SAM2,
    BACKEND_SAM3,
    BACKEND_GDINO_SAM2,
]
RUNNABLE_BACKENDS = {
    BACKEND_CLIPSEG,
    BACKEND_YOLO_WORLD_BOX,
    BACKEND_YOLO11N_SEG,
}
YOLO_WORLD_MODEL = "yolov8s-world.pt"
YOLO11N_SEG_MODEL = str(Path(__file__).resolve().parent / "weights" / "yolo11n-seg.pt")
YOLOWorld = None
YOLO = None
CAMVID_VIDEO = Path("benchmark_data/camvid_road_0001TP.mp4")
CAMVID_GT_DIR = Path("benchmark_gt/camvid_road_0001TP")

DEFAULT_SETTINGS = {
    "prompt": "road",
    "mode": "blur",
    "threshold": 50,
    "infer_scale": "0.50",
    "sweep_prompts": "road",
    "sweep_models": BACKEND_CLIPSEG,
    "sweep_scales": "0.25,0.50,0.75,1.00",
    "sweep_thresholds": "0.35,0.50,0.65",
    "sweep_skip_frames": "0",
    "blur_kernel": 35,
    "skip_frames": 0,
    "benchmark_frames": 0,
}


@dataclass
class RuntimeSettings:
    prompt: str = "road"
    mode: str = "blur"
    threshold: float = 0.5
    infer_scale: float = 0.5
    blur_kernel: int = 35
    skip_frames: int = 0


@dataclass
class SweepConfig:
    prompts: list[str]
    model_ids: list[str]
    scales: list[float]
    thresholds: list[float]
    skip_frames: list[int]

    @property
    def combination_count(self) -> int:
        return (
            len(self.prompts)
            * len(self.model_ids)
            * len(self.scales)
            * len(self.thresholds)
            * len(self.skip_frames)
        )


@dataclass
class FrameMetrics:
    fps: float
    latency_ms: float
    model_latency_ms: float
    mask_coverage: float
    processed_frames: int
    skipped_frames: int
    infer_scale: float


@dataclass
class CapturedFrame:
    frame: np.ndarray
    frame_index: int
    total_frames: int
    captured_at: float
    sequence: int


@dataclass
class MaskSnapshot:
    mask: np.ndarray
    threshold: float
    mode: str
    blur_kernel: int
    sequence: int


class LatestFrameBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: CapturedFrame | None = None
        self.sequence = 0

    def update(self, frame: np.ndarray, frame_index: int, total_frames: int) -> None:
        with self.lock:
            self.sequence += 1
            self.latest = CapturedFrame(
                frame=frame.copy(),
                frame_index=frame_index,
                total_frames=total_frames,
                captured_at=time.perf_counter(),
                sequence=self.sequence,
            )

    def get_latest(self) -> CapturedFrame | None:
        with self.lock:
            if self.latest is None:
                return None
            latest = self.latest
            return CapturedFrame(
                frame=latest.frame.copy(),
                frame_index=latest.frame_index,
                total_frames=latest.total_frames,
                captured_at=latest.captured_at,
                sequence=latest.sequence,
            )


class LatestMaskBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: MaskSnapshot | None = None
        self.sequence = 0

    def update(self, mask: np.ndarray, threshold: float, mode: str, blur_kernel: int) -> None:
        with self.lock:
            self.sequence += 1
            self.latest = MaskSnapshot(
                mask=mask.astype(np.float32, copy=True),
                threshold=float(threshold),
                mode=mode,
                blur_kernel=int(blur_kernel),
                sequence=self.sequence,
            )

    def set_visualization(self, threshold: float, mode: str, blur_kernel: int) -> None:
        with self.lock:
            if self.latest is None:
                return
            self.latest.threshold = float(threshold)
            self.latest.mode = mode
            self.latest.blur_kernel = int(blur_kernel)

    def get_latest(self) -> MaskSnapshot | None:
        with self.lock:
            if self.latest is None:
                return None
            latest = self.latest
            return MaskSnapshot(
                mask=latest.mask.copy(),
                threshold=latest.threshold,
                mode=latest.mode,
                blur_kernel=latest.blur_kernel,
                sequence=latest.sequence,
            )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_yolo_world_cls():
    global YOLOWorld
    if YOLOWorld is None:
        try:
            from ultralytics import YOLOWorld as YOLOWorldClass
        except ImportError as exc:  # pragma: no cover - optional backend.
            raise RuntimeError("YOLO-World backends require ultralytics. Install it with: pip install ultralytics") from exc
        YOLOWorld = YOLOWorldClass
    return YOLOWorld


def get_yolo_cls():
    global YOLO
    if YOLO is None:
        try:
            from ultralytics import YOLO as YOLOClass
        except ImportError as exc:  # pragma: no cover - optional backend.
            raise RuntimeError("YOLO11n-seg backend requires ultralytics. Install it with: pip install ultralytics") from exc
        YOLO = YOLOClass
    return YOLO


def prompt_to_classes(prompt: str) -> list[str]:
    classes = [item.strip() for item in prompt.split(",") if item.strip()]
    if not classes and prompt.strip():
        classes = [prompt.strip()]
    if not classes:
        raise ValueError("Prompt must contain at least one target class")
    return classes


COCO_CLASS_ALIASES: dict[str, list[int]] = {
    "person": [0],
    "people": [0],
    "human": [0],
    "man": [0],
    "woman": [0],
    "pedestrian": [0],
    "사람": [0],
    "bicycle": [1],
    "bike": [1],
    "car": [2],
    "vehicle": [1, 2, 3, 5, 7],
    "자동차": [2],
    "motorcycle": [3],
    "bus": [5],
    "truck": [7],
    "cat": [15],
    "dog": [16],
    "강아지": [16],
    "개": [16],
    "animal": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}


def coco_classes_for_prompt(prompt: str) -> list[int]:
    normalized = normalize_prompt(prompt)
    direct = COCO_CLASS_ALIASES.get(normalized)
    if direct is not None:
        return direct

    classes: list[int] = []
    for word in normalized.split():
        classes.extend(COCO_CLASS_ALIASES.get(word, []))
    if not classes:
        supported = ", ".join(sorted(COCO_CLASS_ALIASES))
        raise ValueError(f"YOLO11n-seg supports COCO class prompts only. Supported aliases: {supported}")
    return sorted(set(classes))


def normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip().lower().replace("_", " ").replace("-", " ")
    words = [word for word in normalized.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def backend_unavailable_message(backend_name: str) -> str | None:
    if backend_name in RUNNABLE_BACKENDS:
        return None
    if backend_name == BACKEND_YOLO_WORLD_SAM2:
        return (
            "YOLO-World + SAM2 tiny tracking is not runnable yet. "
            "SAM2 runtime and a tiny checkpoint need to be added first."
        )
    if backend_name == BACKEND_SAM3:
        return (
            "SAM 3 is not runnable yet. "
            "SAM 3 runtime and checkpoint need to be added first."
        )
    if backend_name == BACKEND_GDINO_SAM2:
        return (
            "Grounding DINO tiny + SAM2 tiny is not runnable yet. "
            "Grounding DINO, SAM2, and tiny checkpoints need to be added first."
        )
    return f"Unsupported backend: {backend_name}"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def parse_float_csv(text: str, minimum: float, maximum: float, label: str) -> list[float]:
    values: list[float] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-number: {item!r}") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} value {value:g} is outside [{minimum:g}, {maximum:g}]")
        values.append(value)
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return list(dict.fromkeys(values))


def parse_int_csv(text: str, minimum: int, maximum: int, label: str) -> list[int]:
    values: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-integer: {item!r}") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} value {value:d} is outside [{minimum:d}, {maximum:d}]")
        values.append(value)
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return list(dict.fromkeys(values))


def parse_text_csv(text: str, label: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return list(dict.fromkeys(values))


def apply_effect(frame: np.ndarray, mask: np.ndarray, threshold: float, mode: str, blur_kernel: int) -> np.ndarray:
    binary = mask >= threshold
    if mode == "mask":
        heat = np.uint8(np.clip(mask, 0.0, 1.0) * 255)
        return cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    if mode == "remove":
        background = np.zeros_like(frame)
    elif mode == "dim":
        background = np.uint8(frame * 0.18)
    else:
        kernel = max(3, int(blur_kernel) | 1)
        background = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    return np.where(binary[..., None], frame, background)


def overlay_effect_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    mode: str,
    blur_kernel: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return apply_effect(frame, mask, threshold, mode, blur_kernel)


def binary_confusion_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, tn, fn


def metric_rates(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float]:
    non_target = fp + tn
    target = tp + fn
    leakage = float(fp / non_target) if non_target else 0.0
    damage = float(fn / target) if target else 0.0
    return leakage, damage


def read_gt_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read GT mask: {path}")
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def fit_table_to_panel(table: QTableWidget, default_section_size: int = 72) -> None:
    table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    table.setWordWrap(False)
    table.setMinimumWidth(0)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Interactive)
    header.setStretchLastSection(False)
    header.setDefaultSectionSize(default_section_size)


class PromptSegmenter:
    def __init__(self, backend_name: str, device: str) -> None:
        self.model_name = backend_name
        self.device = torch.device(device)
        self.processor: CLIPSegProcessor | None = None
        self.model: CLIPSegForImageSegmentation | None = None
        self.yolo_model = None
        self.yolo_seg_model = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.model_name == BACKEND_YOLO11N_SEG:
            self.load_yolo11n_seg()
            return
        if self.model_name != BACKEND_CLIPSEG:
            self.load_yolo_world() if self.model_name == BACKEND_YOLO_WORLD_BOX else self.ensure_backend_available()
            return
        if self.model is not None and self.processor is not None:
            return
        with self.lock:
            if self.model is not None and self.processor is not None:
                return
            self.processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_NAME)
            self.model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_NAME).to(self.device)
            self.model.eval()
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True

    def load_yolo_world(self) -> None:
        if self.yolo_model is not None:
            return
        with self.lock:
            if self.yolo_model is not None:
                return
            yolo_cls = get_yolo_world_cls()
            self.yolo_model = yolo_cls(YOLO_WORLD_MODEL)

    def load_yolo11n_seg(self) -> None:
        if self.yolo_seg_model is not None:
            return
        with self.lock:
            if self.yolo_seg_model is not None:
                return
            yolo_cls = get_yolo_cls()
            self.yolo_seg_model = yolo_cls(YOLO11N_SEG_MODEL)

    def ensure_backend_available(self) -> None:
        message = backend_unavailable_message(self.model_name)
        if message is not None:
            raise RuntimeError(message)

    def set_model(self, model_name: str, device: str) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Backend name cannot be empty")
        if model_name not in BACKEND_PRESETS:
            raise ValueError(f"Unsupported backend: {model_name}")
        with self.lock:
            self.processor = None
            self.model = None
            self.yolo_model = None
            self.yolo_seg_model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.model_name = model_name
            self.device = torch.device(device)
        gc.collect()

    def close(self) -> None:
        with self.lock:
            self.processor = None
            self.model = None
            self.yolo_model = None
            self.yolo_seg_model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    @torch.inference_mode()
    def predict_mask(self, frame: np.ndarray, prompt: str, infer_scale: float) -> tuple[np.ndarray, float]:
        if self.model_name == BACKEND_YOLO11N_SEG:
            return self.predict_yolo11n_seg_mask(frame, prompt, infer_scale)
        if self.model_name == BACKEND_YOLO_WORLD_BOX:
            return self.predict_yolo_world_box_mask(frame, prompt, infer_scale)
        if self.model_name != BACKEND_CLIPSEG:
            self.ensure_backend_available()
        self.load()
        assert self.processor is not None
        assert self.model is not None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(1.0, max(0.05, infer_scale))
        if scale < 1.0:
            resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb

        image = Image.fromarray(resized)
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(self.device)

        started = time.perf_counter()
        logits = self.model(**inputs).logits
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_latency_ms = (time.perf_counter() - started) * 1000.0

        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 3:
            logits = logits.unsqueeze(1)

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.sigmoid(upsampled)[0, 0].detach().float().cpu().numpy()
        return mask.astype(np.float32), model_latency_ms

    @torch.inference_mode()
    def predict_yolo_world_box_mask(self, frame: np.ndarray, prompt: str, infer_scale: float) -> tuple[np.ndarray, float]:
        self.load_yolo_world()
        assert self.yolo_model is not None

        classes = prompt_to_classes(prompt)
        height, width = frame.shape[:2]
        scale = min(1.0, max(0.05, infer_scale))
        if scale < 1.0:
            resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = frame

        self.yolo_model.set_classes(classes)
        started = time.perf_counter()
        results = self.yolo_model.predict(resized, verbose=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_latency_ms = (time.perf_counter() - started) * 1000.0

        mask = np.zeros((height, width), dtype=np.float32)
        if not results:
            return mask, model_latency_ms
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return mask, model_latency_ms
        xyxy = boxes.xyxy.detach().cpu().numpy()
        inv_scale = 1.0 / scale
        for x1, y1, x2, y2 in xyxy:
            left = int(max(0, round(x1 * inv_scale)))
            top = int(max(0, round(y1 * inv_scale)))
            right = int(min(width, round(x2 * inv_scale)))
            bottom = int(min(height, round(y2 * inv_scale)))
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 1.0
        return mask, model_latency_ms

    @torch.inference_mode()
    def predict_yolo11n_seg_mask(self, frame: np.ndarray, prompt: str, infer_scale: float) -> tuple[np.ndarray, float]:
        self.load_yolo11n_seg()
        assert self.yolo_seg_model is not None

        class_ids = coco_classes_for_prompt(prompt)
        height, width = frame.shape[:2]
        scale = min(1.0, max(0.05, infer_scale))
        imgsz = max(320, int(round(max(width, height) * scale)))

        started = time.perf_counter()
        results = self.yolo_seg_model.predict(
            frame,
            imgsz=imgsz,
            conf=0.25,
            iou=0.7,
            classes=class_ids,
            retina_masks=True,
            device=str(self.device),
            verbose=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_latency_ms = (time.perf_counter() - started) * 1000.0

        mask = np.zeros((height, width), dtype=np.float32)
        if not results or results[0].masks is None:
            return mask, model_latency_ms
        raw_masks = results[0].masks.data.detach().cpu().numpy()
        for raw_mask in raw_masks:
            mask_part = raw_mask.astype(np.float32)
            if mask_part.shape != (height, width):
                mask_part = cv2.resize(mask_part, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = np.maximum(mask, mask_part)
        return np.clip(mask, 0.0, 1.0).astype(np.float32), model_latency_ms


class CaptureWorker(QThread):
    input_frame = Signal(QImage)
    output_frame = Signal(QImage)
    status = Signal(str)
    position_changed = Signal(int, int)
    video_info = Signal(int, float)

    def __init__(self, frame_buffer: LatestFrameBuffer, mask_buffer: LatestMaskBuffer) -> None:
        super().__init__()
        self.frame_buffer = frame_buffer
        self.mask_buffer = mask_buffer
        self.source: int | str = 0
        self.running = False
        self.paused = False
        self.seek_frame: int | None = None
        self.control_lock = threading.Lock()
        self.cap: cv2.VideoCapture | None = None
        self.cap_lock = threading.Lock()

    def set_source(self, source: int | str) -> None:
        self.source = source
        with self.control_lock:
            self.paused = False
            self.seek_frame = None

    def stop(self) -> None:
        self.running = False
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()

    def pause(self) -> None:
        with self.control_lock:
            self.paused = True

    def resume(self) -> None:
        with self.control_lock:
            self.paused = False

    def seek(self, frame_index: int) -> None:
        with self.control_lock:
            self.seek_frame = max(0, frame_index)

    def run(self) -> None:
        self.running = True
        cap = cv2.VideoCapture(self.source)
        with self.cap_lock:
            self.cap = cap
        if not cap.isOpened():
            self.status.emit(f"Could not open source: {self.source}")
            with self.cap_lock:
                if self.cap is cap:
                    self.cap = None
            cap.release()
            return

        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(self.source, str) else 0
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        processed = 0
        skipped = 0
        last_emit = time.perf_counter()
        self.status.emit("source opened")
        self.video_info.emit(total_frames, float(source_fps))

        while self.running:
            loop_started = time.perf_counter()
            with self.control_lock:
                paused = self.paused
                requested_seek = self.seek_frame
                self.seek_frame = None

            if requested_seek is not None and isinstance(self.source, str):
                frame_index = min(requested_seek, max(0, total_frames - 1)) if total_frames > 0 else requested_seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            elif paused:
                time.sleep(0.03)
                continue

            ok, frame = cap.read()
            if not ok:
                break

            self.frame_buffer.update(frame, frame_index, total_frames)
            self.input_frame.emit(bgr_to_qimage(frame))
            mask_snapshot = self.mask_buffer.get_latest()
            if mask_snapshot is None:
                self.output_frame.emit(bgr_to_qimage(frame))
            else:
                self.output_frame.emit(
                    bgr_to_qimage(
                        overlay_effect_mask(
                            frame,
                            mask_snapshot.mask,
                            mask_snapshot.threshold,
                            mask_snapshot.mode,
                            mask_snapshot.blur_kernel,
                        )
                    )
                )
            self.position_changed.emit(frame_index, total_frames)
            frame_index += 1

            if isinstance(self.source, str) and source_fps > 0:
                frame_interval = 1.0 / source_fps
                spent = time.perf_counter() - loop_started
                if spent < frame_interval:
                    time.sleep(frame_interval - spent)

        with self.cap_lock:
            if self.cap is cap:
                self.cap = None
        cap.release()
        self.status.emit("capture stopped")


class InferenceWorker(QThread):
    metrics = Signal(object)
    status = Signal(str)

    def __init__(
        self,
        segmenter: PromptSegmenter,
        frame_buffer: LatestFrameBuffer,
        mask_buffer: LatestMaskBuffer,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.frame_buffer = frame_buffer
        self.mask_buffer = mask_buffer
        self.settings = RuntimeSettings()
        self.settings_lock = threading.Lock()
        self.running = False

    def configure(self, settings: RuntimeSettings) -> None:
        with self.settings_lock:
            self.settings = settings

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.running = True
        processed = 0
        skipped = 0
        last_emit = time.perf_counter()
        last_sequence = 0
        self.status.emit("inference started")

        while self.running:
            captured = self.frame_buffer.get_latest()
            if captured is None or captured.sequence == last_sequence:
                time.sleep(0.005)
                continue
            last_sequence = captured.sequence

            with self.settings_lock:
                settings = RuntimeSettings(**self.settings.__dict__)

            process_interval = max(1, settings.skip_frames + 1)
            should_process = captured.frame_index % process_interval == 0
            if should_process and settings.prompt.strip():
                started = time.perf_counter()
                try:
                    mask, model_latency_ms = self.segmenter.predict_mask(
                        captured.frame,
                        settings.prompt.strip(),
                        settings.infer_scale,
                    )
                    self.mask_buffer.update(mask, settings.threshold, settings.mode, settings.blur_kernel)
                    processed += 1
                    now = time.perf_counter()
                    fps = 1.0 / max(1e-6, now - last_emit)
                    last_emit = now
                    self.metrics.emit(
                        FrameMetrics(
                            fps=fps,
                            latency_ms=(now - captured.captured_at) * 1000.0,
                            model_latency_ms=model_latency_ms,
                            mask_coverage=float((mask >= settings.threshold).mean()),
                            processed_frames=processed,
                            skipped_frames=skipped,
                            infer_scale=settings.infer_scale,
                        )
                    )
                except Exception as exc:
                    self.status.emit(str(exc))
                    self.running = False
            else:
                skipped += 1

        self.status.emit("inference stopped")


class SweepWorker(QThread):
    finished_with_results = Signal(list)
    status = Signal(str)

    def __init__(
        self,
        segmenter: PromptSegmenter,
        frame: np.ndarray,
        settings: RuntimeSettings,
        sweep_config: SweepConfig,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.frame = frame
        self.settings = settings
        self.sweep_config = sweep_config

    def run(self) -> None:
        try:
            results = []
            original_model = self.segmenter.model_name
            device_name = str(self.segmenter.device)
            try:
                for model_id in self.sweep_config.model_ids:
                    if self.segmenter.model_name != model_id:
                        self.segmenter.set_model(model_id, device_name)
                    for prompt in self.sweep_config.prompts:
                        for skip_frames in self.sweep_config.skip_frames:
                            for infer_scale in self.sweep_config.scales:
                                for threshold in self.sweep_config.thresholds:
                                    started = time.perf_counter()
                                    mask, model_latency_ms = self.segmenter.predict_mask(
                                        self.frame,
                                        prompt,
                                        infer_scale,
                                    )
                                    _ = apply_effect(
                                        self.frame,
                                        mask,
                                        threshold,
                                        self.settings.mode,
                                        self.settings.blur_kernel,
                                    )
                                    total_ms = (time.perf_counter() - started) * 1000.0
                                    results.append(
                                        {
                                            "model_id": model_id,
                                            "prompt": prompt,
                                            "skip_frames": skip_frames,
                                            "infer_scale": infer_scale,
                                            "threshold": threshold,
                                            "latency_ms": total_ms,
                                            "model_latency_ms": model_latency_ms,
                                            "mask_coverage": float((mask >= threshold).mean()),
                                        }
                                    )
            finally:
                if self.segmenter.model_name != original_model:
                    self.segmenter.set_model(original_model, device_name)
            self.finished_with_results.emit(results)
        except Exception as exc:
            self.status.emit(str(exc))


class BenchmarkWorker(QThread):
    finished_with_results = Signal(list)
    row_ready = Signal(dict)
    progress = Signal(int, int, str)
    preview = Signal(QImage, QImage)
    status = Signal(str)

    def __init__(
        self,
        segmenter: PromptSegmenter,
        settings: RuntimeSettings,
        max_frames: int,
        sweep_config: SweepConfig,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.settings = settings
        self.max_frames = max_frames
        self.sweep_config = sweep_config

    def run(self) -> None:
        try:
            if not CAMVID_VIDEO.exists() or not CAMVID_GT_DIR.exists():
                raise RuntimeError("CamVid benchmark files not found. Run benchmark/prepare_camvid_benchmark.py first.")

            frames: list[np.ndarray] = []
            gt_masks: list[np.ndarray] = []
            cap = cv2.VideoCapture(str(CAMVID_VIDEO))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {CAMVID_VIDEO}")

            frame_index = 0
            while self.max_frames <= 0 or len(frames) < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                gt_path = CAMVID_GT_DIR / f"frame_{frame_index:06d}_mask.png"
                if gt_path.exists():
                    frames.append(frame)
                    gt_masks.append(read_gt_mask(gt_path, width, height))
                frame_index += 1
            cap.release()

            if not frames:
                raise RuntimeError("No benchmark frames with GT masks found")

            results = []
            combinations = [
                (model_id, prompt, skip_frames, scale, threshold)
                for model_id in self.sweep_config.model_ids
                for prompt in self.sweep_config.prompts
                for skip_frames in self.sweep_config.skip_frames
                for scale in self.sweep_config.scales
                for threshold in self.sweep_config.thresholds
            ]
            total_combinations = len(combinations)
            original_model = self.segmenter.model_name
            device_name = str(self.segmenter.device)
            try:
                for combo_index, (model_id, prompt, skip_frames, infer_scale, threshold) in enumerate(
                    combinations,
                    start=1,
                ):
                    if self.segmenter.model_name != model_id:
                        self.segmenter.set_model(model_id, device_name)
                    self.progress.emit(
                        combo_index,
                        total_combinations,
                        (
                            f"backend={model_id}, prompt={prompt}, skip={skip_frames}, "
                            f"scale={infer_scale:.2f}, threshold={threshold:.2f}"
                        ),
                    )
                    tp_total = fp_total = tn_total = fn_total = 0
                    latencies = []
                    model_calls = 0
                    latest_mask: np.ndarray | None = None
                    process_interval = max(1, skip_frames + 1)
                    started = time.perf_counter()
                    for frame_offset, (frame, gt_mask) in enumerate(zip(frames, gt_masks)):
                        should_process = frame_offset % process_interval == 0 or latest_mask is None
                        if should_process:
                            latest_mask, model_latency_ms = self.segmenter.predict_mask(
                                frame,
                                prompt,
                                infer_scale,
                            )
                            latencies.append(model_latency_ms)
                            model_calls += 1
                        assert latest_mask is not None
                        output = apply_effect(
                            frame,
                            latest_mask,
                            threshold,
                            self.settings.mode,
                            self.settings.blur_kernel,
                        )
                        self.preview.emit(bgr_to_qimage(frame), bgr_to_qimage(output))
                        tp, fp, tn, fn = binary_confusion_counts(latest_mask >= threshold, gt_mask)
                        tp_total += tp
                        fp_total += fp
                        tn_total += tn
                        fn_total += fn
                    elapsed = time.perf_counter() - started
                    leakage, damage = metric_rates(tp_total, fp_total, tn_total, fn_total)
                    row = {
                        "model_id": model_id,
                        "prompt": prompt,
                        "skip_frames": skip_frames,
                        "infer_scale": infer_scale,
                        "threshold": threshold,
                        "non_target_leakage": leakage,
                        "target_damage": damage,
                        "fps": float(len(frames) / elapsed) if elapsed > 0 else 0.0,
                        "latency_ms": float((elapsed / len(frames)) * 1000.0) if frames else 0.0,
                        "model_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                        "model_calls": model_calls,
                        "tp": tp_total,
                        "fp": fp_total,
                        "tn": tn_total,
                        "fn": fn_total,
                    }
                    results.append(row)
                    self.row_ready.emit(row)
            finally:
                if self.segmenter.model_name != original_model:
                    self.segmenter.set_model(original_model, device_name)
            self.finished_with_results.emit(results)
        except Exception as exc:
            self.status.emit(str(exc))


class ImagePane(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #050608; color: #aab2c0; border: 1px solid #303541; }"
        )
        self.pixmap_source: QPixmap | None = None

    @Slot(QImage)
    def set_image(self, image: QImage) -> None:
        self.pixmap_source = QPixmap.fromImage(image)
        self.update_scaled()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.update_scaled()

    def update_scaled(self) -> None:
        if self.pixmap_source is None:
            return
        self.setPixmap(
            self.pixmap_source.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    def __init__(self, segmenter: PromptSegmenter) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.frame_buffer: LatestFrameBuffer | None = None
        self.mask_buffer: LatestMaskBuffer | None = None
        self.capture_worker: CaptureWorker | None = None
        self.inference_worker: InferenceWorker | None = None
        self.active_source: int | str | None = None
        self.sweep_worker: SweepWorker | None = None
        self.benchmark_worker: BenchmarkWorker | None = None
        self.retiring_threads: list[QThread] = []
        self.latest_benchmark_results: list[dict] = []

        self.setWindowTitle("Focus On You - Real-time Prompt Segmentation MVP")
        self.resize(1360, 860)
        self.setMinimumSize(1040, 680)
        self._build_ui()
        self._connect_ui()
        self.apply_dark_theme()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        video_layout = QVBoxLayout()
        video_layout.setSpacing(8)
        panes = QHBoxLayout()
        panes.setSpacing(8)
        self.input_pane = ImagePane("Input")
        self.output_pane = ImagePane("Processed")
        panes.addWidget(self.input_pane, 1)
        panes.addWidget(self.output_pane, 1)
        video_layout.addLayout(panes, 1)

        metrics = QGridLayout()
        self.fps_label = QLabel("0.00")
        self.latency_label = QLabel("0.0 ms")
        self.model_latency_label = QLabel("0.0 ms")
        self.coverage_label = QLabel("0.000")
        self.frames_label = QLabel("0 / 0")
        for i, (name, label) in enumerate(
            [
                ("FPS", self.fps_label),
                ("Latency", self.latency_label),
                ("Model latency", self.model_latency_label),
                ("Mask coverage", self.coverage_label),
                ("Processed / skipped", self.frames_label),
            ]
        ):
            box = QGroupBox(name)
            layout = QVBoxLayout(box)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 22px; font-weight: 700;")
            layout.addWidget(label)
            metrics.addWidget(box, i // 3, i % 3)
        video_layout.addLayout(metrics)
        main.addLayout(video_layout, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(390)
        controls_scroll.setMaximumWidth(470)
        controls_container = QWidget()
        controls = QVBoxLayout()
        controls.setContentsMargins(6, 6, 6, 6)
        controls.setSpacing(12)
        controls_container.setLayout(controls)
        controls_scroll.setWidget(controls_container)
        main.addWidget(controls_scroll)

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)
        self.model_name = QComboBox()
        self.model_name.setEditable(False)
        self.model_name.addItems(BACKEND_PRESETS)
        self.model_name.setCurrentText(self.segmenter.model_name)
        self.device_name = QComboBox()
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.insert(0, "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        self.device_name.addItems(devices)
        self.device_name.setCurrentText(str(self.segmenter.device))
        self.apply_model_button = QPushButton("Apply Model")
        self.current_backend_label = QLabel()
        self.current_backend_label.setWordWrap(True)
        self.current_backend_label.setStyleSheet("color: #47c2ff; font-weight: 700;")
        model_form.addRow("Backend", self.model_name)
        model_form.addRow("Device", self.device_name)
        model_form.addRow(self.apply_model_button)
        model_form.addRow("Current", self.current_backend_label)
        controls.addWidget(model_box)

        source_box = QGroupBox("Source Controls")
        source_form = QFormLayout(source_box)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        self.start_camera_button = QPushButton("Start Camera")
        self.stop_camera_button = QPushButton("Stop Camera")
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.start_camera_button)
        camera_row.addWidget(self.stop_camera_button)

        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select a video file")
        self.browse_button = QPushButton("Browse")
        path_row = QHBoxLayout()
        path_row.addWidget(self.video_path)
        path_row.addWidget(self.browse_button)
        self.play_video_button = QPushButton("Play Video")
        self.pause_video_button = QPushButton("Pause")
        self.stop_video_button = QPushButton("Stop Video")
        video_button_row = QHBoxLayout()
        video_button_row.addWidget(self.play_video_button)
        video_button_row.addWidget(self.pause_video_button)
        video_button_row.addWidget(self.stop_video_button)
        self.video_seek = QSlider(Qt.Horizontal)
        self.video_seek.setRange(0, 0)
        self.video_position = QLabel("0 / 0")
        seek_row = QHBoxLayout()
        seek_row.addWidget(self.video_seek)
        seek_row.addWidget(self.video_position)
        source_form.addRow("Camera index", self.camera_index)
        source_form.addRow(camera_row)
        source_form.addRow("Video file", path_row)
        source_form.addRow(video_button_row)
        source_form.addRow("Video position", seek_row)
        controls.addWidget(source_box)

        prompt_box = QGroupBox("Prompt & Effect")
        prompt_form = QFormLayout(prompt_box)
        self.prompt = QLineEdit(DEFAULT_SETTINGS["prompt"])
        self.mode = QComboBox()
        self.mode.addItems(["blur", "remove", "dim", "mask"])
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(5, 95)
        self.threshold.setValue(DEFAULT_SETTINGS["threshold"])
        self.threshold_label = QLabel("0.50")
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold)
        threshold_row.addWidget(self.threshold_label)
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(3, 99)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(DEFAULT_SETTINGS["blur_kernel"])
        prompt_form.addRow("Target prompt", self.prompt)
        prompt_form.addRow("Mode", self.mode)
        prompt_form.addRow("Threshold", threshold_row)
        prompt_form.addRow("Blur kernel", self.blur_kernel)
        controls.addWidget(prompt_box)

        realtime_box = QGroupBox("Realtime Controls")
        realtime_form = QFormLayout(realtime_box)
        self.infer_scale = QComboBox()
        self.infer_scale.addItems(["0.25", "0.50", "0.75", "1.00"])
        self.infer_scale.setCurrentText(DEFAULT_SETTINGS["infer_scale"])
        self.skip_frames = QSpinBox()
        self.skip_frames.setRange(0, 30)
        self.skip_frames.setValue(DEFAULT_SETTINGS["skip_frames"])
        self.reset_hyperparams_button = QPushButton("Reset Hyperparameters")
        realtime_form.addRow("Inference scale", self.infer_scale)
        realtime_form.addRow("Skip frames", self.skip_frames)
        realtime_form.addRow(self.reset_hyperparams_button)
        controls.addWidget(realtime_box)

        sweep_space_box = QGroupBox("Sweep Space")
        sweep_space_form = QFormLayout(sweep_space_box)
        self.sweep_prompts = QLineEdit(DEFAULT_SETTINGS["sweep_prompts"])
        self.sweep_models = QComboBox()
        self.sweep_models.addItems(BACKEND_PRESETS)
        self.sweep_models.setCurrentText(DEFAULT_SETTINGS["sweep_models"])
        self.sweep_scales = QLineEdit(DEFAULT_SETTINGS["sweep_scales"])
        self.sweep_thresholds = QLineEdit(DEFAULT_SETTINGS["sweep_thresholds"])
        self.sweep_skip_frames = QLineEdit(DEFAULT_SETTINGS["sweep_skip_frames"])
        self.sweep_estimate = QLabel("")
        self.sweep_estimate.setWordWrap(True)
        sweep_space_form.addRow("Target prompts", self.sweep_prompts)
        sweep_space_form.addRow("Backend", self.sweep_models)
        sweep_space_form.addRow("Scales", self.sweep_scales)
        sweep_space_form.addRow("Thresholds", self.sweep_thresholds)
        sweep_space_form.addRow("Skip frames", self.sweep_skip_frames)
        sweep_space_form.addRow("Combinations", self.sweep_estimate)
        controls.addWidget(sweep_space_box)

        sweep_box = QGroupBox("Optimization Sweep")
        sweep_layout = QVBoxLayout(sweep_box)
        self.sweep_button = QPushButton("Run Sweep On Current Frame")
        self.sweep_table = QTableWidget(0, 7)
        self.sweep_table.setHorizontalHeaderLabels(["backend", "prompt", "skip", "scale", "thr", "lat ms", "coverage"])
        fit_table_to_panel(self.sweep_table)
        self.sweep_table.setMinimumHeight(120)
        self.sweep_table.setMaximumHeight(170)
        sweep_layout.addWidget(self.sweep_button)
        sweep_layout.addWidget(self.sweep_table)
        controls.addWidget(sweep_box)

        benchmark_box = QGroupBox("GT Benchmark")
        benchmark_layout = QVBoxLayout(benchmark_box)
        self.benchmark_frames = QSpinBox()
        self.benchmark_frames.setRange(0, 100000)
        self.benchmark_frames.setValue(DEFAULT_SETTINGS["benchmark_frames"])
        self.benchmark_frames.setSpecialValueText("All")
        self.benchmark_button = QPushButton("Run CamVid Road Benchmark")
        self.save_benchmark_button = QPushButton("Save result")
        self.save_benchmark_button.setEnabled(False)
        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setRange(0, 12)
        self.benchmark_progress.setValue(0)
        self.benchmark_progress.setTextVisible(True)
        self.benchmark_progress_label = QLabel("idle")
        self.benchmark_progress_label.setWordWrap(True)
        self.benchmark_progress_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.benchmark_table = QTableWidget(0, 9)
        self.benchmark_table.setHorizontalHeaderLabels(
            ["backend", "prompt", "skip", "scale", "thr", "leak", "damage", "FPS", "lat ms"]
        )
        fit_table_to_panel(self.benchmark_table, default_section_size=68)
        self.benchmark_table.setMinimumHeight(150)
        self.benchmark_table.setMaximumHeight(210)
        benchmark_form = QFormLayout()
        benchmark_form.addRow("Frames", self.benchmark_frames)
        benchmark_button_row = QHBoxLayout()
        benchmark_button_row.addWidget(self.benchmark_button)
        benchmark_button_row.addWidget(self.save_benchmark_button)
        benchmark_layout.addLayout(benchmark_form)
        benchmark_layout.addLayout(benchmark_button_row)
        benchmark_layout.addWidget(self.benchmark_progress)
        benchmark_layout.addWidget(self.benchmark_progress_label)
        benchmark_layout.addWidget(self.benchmark_table)
        controls.addWidget(benchmark_box)

        self.hyperparameter_controls = [
            self.model_name,
            self.device_name,
            self.apply_model_button,
            self.prompt,
            self.mode,
            self.threshold,
            self.blur_kernel,
            self.infer_scale,
            self.skip_frames,
            self.reset_hyperparams_button,
            self.sweep_prompts,
            self.sweep_models,
            self.sweep_scales,
            self.sweep_thresholds,
            self.sweep_skip_frames,
            self.sweep_button,
            self.benchmark_frames,
        ]

        self.status_label = QLabel(f"Device: {self.segmenter.device}. Backend loads on first inference.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)
        controls.addStretch(1)
        self.update_sweep_estimate()
        self.update_current_backend_label()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    def _connect_ui(self) -> None:
        self.threshold.valueChanged.connect(self.on_threshold_changed)
        self.apply_model_button.clicked.connect(self.apply_model)
        self.browse_button.clicked.connect(self.browse_video)
        self.start_camera_button.clicked.connect(self.start_camera)
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.play_video_button.clicked.connect(self.play_video)
        self.pause_video_button.clicked.connect(self.pause_video)
        self.stop_video_button.clicked.connect(self.stop_video)
        self.video_seek.sliderReleased.connect(self.seek_video)
        self.reset_hyperparams_button.clicked.connect(self.reset_hyperparameters)
        self.sweep_button.clicked.connect(self.run_sweep)
        self.benchmark_button.clicked.connect(self.run_benchmark)
        self.save_benchmark_button.clicked.connect(self.save_benchmark_results)
        self.sweep_prompts.textChanged.connect(self.update_sweep_estimate)
        self.sweep_models.currentTextChanged.connect(self.update_sweep_estimate)
        self.sweep_scales.textChanged.connect(self.update_sweep_estimate)
        self.sweep_thresholds.textChanged.connect(self.update_sweep_estimate)
        self.sweep_skip_frames.textChanged.connect(self.update_sweep_estimate)
        self.benchmark_frames.valueChanged.connect(self.update_sweep_estimate)
        self.model_name.currentTextChanged.connect(self.on_backend_selection_changed)
        self.device_name.currentTextChanged.connect(lambda: self.update_current_backend_label(pending=True))
        for widget in [
            self.prompt,
            self.mode,
            self.infer_scale,
            self.skip_frames,
            self.blur_kernel,
        ]:
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.push_settings)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.push_settings)
            else:
                widget.valueChanged.connect(self.push_settings)

    def apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111318; color: #f2f3f5; }
            QGroupBox { border: 1px solid #303541; border-radius: 6px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aab2c0; }
            QLineEdit, QComboBox, QSpinBox, QTableWidget {
                background: #1b1f29; color: #f2f3f5; border: 1px solid #303541; border-radius: 4px; padding: 4px;
            }
            QPushButton { background: #263142; color: #f2f3f5; border: 1px solid #3b475a; border-radius: 5px; padding: 7px; font-weight: 700; }
            QPushButton:hover { background: #314059; }
            QSlider::groove:horizontal { height: 4px; background: #303541; }
            QSlider::handle:horizontal { background: #47c2ff; width: 14px; margin: -5px 0; border-radius: 7px; }
            QHeaderView::section { background: #202532; color: #aab2c0; border: 0; padding: 4px; }
            """
        )

    @Slot(str)
    def on_backend_selection_changed(self, backend_name: str) -> None:
        self.update_sweep_estimate()
        if backend_name == BACKEND_YOLO11N_SEG and self.prompt.text().strip().lower() == "road":
            self.prompt.setText("person")
        self.update_current_backend_label(pending=True)

    def update_current_backend_label(self, *, pending: bool = False) -> None:
        if pending:
            self.current_backend_label.setText(
                f"Pending: {self.model_name.currentText()} / {self.device_name.currentText()}"
            )
            return
        self.current_backend_label.setText(f"{self.segmenter.model_name} / {self.segmenter.device}")

    def current_settings(self) -> RuntimeSettings:
        return RuntimeSettings(
            prompt=self.prompt.text().strip() or "target",
            mode=self.mode.currentText(),
            threshold=self.threshold.value() / 100.0,
            infer_scale=float(self.infer_scale.currentText()),
            blur_kernel=self.blur_kernel.value(),
            skip_frames=self.skip_frames.value(),
        )

    def set_hyperparameter_controls_enabled(self, enabled: bool) -> None:
        for widget in self.hyperparameter_controls:
            widget.setEnabled(enabled)

    def current_sweep_config(self) -> SweepConfig:
        return SweepConfig(
            prompts=parse_text_csv(self.sweep_prompts.text(), "target prompts"),
            model_ids=[self.sweep_models.currentText()],
            scales=parse_float_csv(self.sweep_scales.text(), 0.05, 1.0, "scales"),
            thresholds=parse_float_csv(self.sweep_thresholds.text(), 0.01, 0.99, "thresholds"),
            skip_frames=parse_int_csv(self.sweep_skip_frames.text(), 0, 30, "skip frames"),
        )

    @Slot()
    def update_sweep_estimate(self, *args) -> None:  # type: ignore[no-untyped-def]
        try:
            config = self.current_sweep_config()
        except ValueError as exc:
            self.sweep_estimate.setText(str(exc))
            return
        frame_text = "all frames" if self.benchmark_frames.value() == 0 else f"{self.benchmark_frames.value()} frames"
        self.sweep_estimate.setText(
            f"{config.combination_count} combinations "
            f"({len(config.model_ids)} backends x {len(config.prompts)} prompts x "
            f"{len(config.skip_frames)} skip values x {len(config.scales)} scales x "
            f"{len(config.thresholds)} thresholds), benchmark: {frame_text}"
        )

    @Slot()
    def reset_hyperparameters(self) -> None:
        self.prompt.setText(DEFAULT_SETTINGS["prompt"])
        self.mode.setCurrentText(DEFAULT_SETTINGS["mode"])
        self.threshold.setValue(DEFAULT_SETTINGS["threshold"])
        self.infer_scale.setCurrentText(DEFAULT_SETTINGS["infer_scale"])
        self.sweep_prompts.setText(DEFAULT_SETTINGS["sweep_prompts"])
        self.sweep_models.setCurrentText(DEFAULT_SETTINGS["sweep_models"])
        self.sweep_scales.setText(DEFAULT_SETTINGS["sweep_scales"])
        self.sweep_thresholds.setText(DEFAULT_SETTINGS["sweep_thresholds"])
        self.sweep_skip_frames.setText(DEFAULT_SETTINGS["sweep_skip_frames"])
        self.blur_kernel.setValue(DEFAULT_SETTINGS["blur_kernel"])
        self.skip_frames.setValue(DEFAULT_SETTINGS["skip_frames"])
        self.benchmark_frames.setValue(DEFAULT_SETTINGS["benchmark_frames"])
        self.update_sweep_estimate()
        self.push_settings()
        self.set_status("Hyperparameters reset to defaults")

    @Slot()
    def push_settings(self, *args) -> None:  # type: ignore[no-untyped-def]
        settings = self.current_settings()
        if self.inference_worker is not None:
            self.inference_worker.configure(settings)
        if self.mask_buffer is not None:
            self.mask_buffer.set_visualization(settings.threshold, settings.mode, settings.blur_kernel)

    @Slot(int)
    def on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value / 100.0:.2f}")
        self.push_settings()

    @Slot()
    def apply_model(self) -> None:
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            QMessageBox.warning(self, "Sweep running", "Wait for the current sweep before switching backends.")
            return
        if self.benchmark_worker is not None and self.benchmark_worker.isRunning():
            QMessageBox.warning(self, "Benchmark running", "Wait for the current benchmark before switching backends.")
            return
        if self.retiring_threads:
            QMessageBox.warning(self, "Worker stopping", "Wait for the previous source to finish stopping.")
            self.model_name.setCurrentText(self.segmenter.model_name)
            self.update_current_backend_label()
            return
        requested_backend = self.model_name.currentText()
        unavailable_message = backend_unavailable_message(requested_backend)
        if unavailable_message is not None:
            QMessageBox.warning(self, "Backend unavailable", unavailable_message)
            self.model_name.setCurrentText(self.segmenter.model_name)
            self.update_current_backend_label()
            return
        self.stop_worker(wait=True)
        try:
            self.segmenter.set_model(requested_backend, self.device_name.currentText())
        except Exception as exc:
            QMessageBox.warning(self, "Backend switch failed", str(exc))
            self.model_name.setCurrentText(self.segmenter.model_name)
            self.update_current_backend_label()
            return
        self.set_status(
            f"Backend set to {self.segmenter.model_name} on {self.segmenter.device}. "
            "It will load on the next inference."
        )
        self.update_current_backend_label()

    @Slot()
    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", str(Path.cwd()), "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path.setText(path)

    def start_source(self, source: int | str) -> None:
        self.stop_worker(wait=True)
        self.active_source = source
        self.frame_buffer = LatestFrameBuffer()
        self.mask_buffer = LatestMaskBuffer()
        self.capture_worker = CaptureWorker(self.frame_buffer, self.mask_buffer)
        self.inference_worker = InferenceWorker(self.segmenter, self.frame_buffer, self.mask_buffer)
        self.capture_worker.set_source(source)
        self.inference_worker.configure(self.current_settings())
        self.capture_worker.input_frame.connect(self.input_pane.set_image)
        self.capture_worker.output_frame.connect(self.output_pane.set_image)
        self.capture_worker.status.connect(self.set_status)
        self.capture_worker.video_info.connect(self.update_video_info)
        self.capture_worker.position_changed.connect(self.update_video_position)
        self.capture_worker.finished.connect(self.handle_capture_finished)
        self.inference_worker.metrics.connect(self.update_metrics)
        self.inference_worker.status.connect(self.set_status)
        self.capture_worker.start()
        self.inference_worker.start()

    @Slot()
    def start_camera(self) -> None:
        self.video_seek.setRange(0, 0)
        self.video_position.setText("camera")
        self.start_source(self.camera_index.value())

    @Slot()
    def stop_camera(self) -> None:
        self.stop_worker()
        self.video_position.setText("0 / 0")

    @Slot()
    def play_video(self) -> None:
        path = self.video_path.text().strip()
        if not path:
            self.browse_video()
            path = self.video_path.text().strip()
        if not path:
            return
        if self.capture_worker is not None and self.active_source == path:
            self.capture_worker.resume()
            self.set_status("video resumed")
            return
        self.start_source(path)

    @Slot()
    def pause_video(self) -> None:
        if self.capture_worker is not None and isinstance(self.active_source, str):
            self.capture_worker.pause()
            self.set_status("video paused")

    @Slot()
    def stop_video(self) -> None:
        self.stop_worker()
        self.video_seek.blockSignals(True)
        self.video_seek.setValue(0)
        self.video_seek.blockSignals(False)
        maximum = self.video_seek.maximum()
        self.video_position.setText(f"0 / {maximum}")

    @Slot()
    def seek_video(self) -> None:
        if self.capture_worker is not None and isinstance(self.active_source, str):
            self.capture_worker.seek(self.video_seek.value())

    @Slot(int, float)
    def update_video_info(self, total_frames: int, fps: float) -> None:
        if total_frames > 0:
            self.video_seek.setRange(0, max(0, total_frames - 1))
            self.video_position.setText(f"0 / {total_frames - 1} ({fps:.1f} fps)")
        else:
            self.video_seek.setRange(0, 0)
            self.video_position.setText("camera")

    @Slot(int, int)
    def update_video_position(self, frame_index: int, total_frames: int) -> None:
        if total_frames <= 0:
            return
        self.video_seek.blockSignals(True)
        self.video_seek.setValue(min(frame_index, self.video_seek.maximum()))
        self.video_seek.blockSignals(False)
        self.video_position.setText(f"{frame_index} / {max(0, total_frames - 1)}")

    @Slot()
    def stop_worker(self, *, wait: bool = False) -> None:
        capture_worker = self.capture_worker
        inference_worker = self.inference_worker
        self.capture_worker = None
        self.inference_worker = None
        self.active_source = None
        self.frame_buffer = None
        self.mask_buffer = None

        if capture_worker is not None:
            capture_worker.stop()
        if inference_worker is not None:
            inference_worker.stop()
        for worker in [capture_worker, inference_worker]:
            if worker is None:
                continue
            if wait and worker.isRunning() and worker.wait(5000):
                worker.deleteLater()
            elif worker.isRunning():
                self.retire_thread(worker)
            else:
                worker.deleteLater()

    @Slot()
    def handle_capture_finished(self) -> None:
        capture_worker = self.sender()
        if capture_worker is not self.capture_worker:
            return
        inference_worker = self.inference_worker
        finished_source = self.active_source
        self.capture_worker = None
        self.inference_worker = None
        self.active_source = None
        self.frame_buffer = None
        self.mask_buffer = None
        if inference_worker is not None:
            inference_worker.stop()
            if inference_worker.isRunning():
                self.retire_thread(inference_worker)
            else:
                inference_worker.deleteLater()
        if isinstance(capture_worker, QThread):
            capture_worker.deleteLater()
        self.set_status("video finished" if isinstance(finished_source, str) else "capture stopped")

    def retire_thread(self, thread: QThread) -> None:
        if thread in self.retiring_threads:
            return
        self.retiring_threads.append(thread)
        thread.finished.connect(lambda thread=thread: self.release_retired_thread(thread))
        if not thread.isRunning():
            self.release_retired_thread(thread)

    def release_retired_thread(self, thread: QThread) -> None:
        if thread not in self.retiring_threads:
            return
        self.retiring_threads.remove(thread)
        thread.deleteLater()

    @Slot(object)
    def update_metrics(self, metrics: FrameMetrics) -> None:
        self.fps_label.setText(f"{metrics.fps:.2f}")
        self.latency_label.setText(f"{metrics.latency_ms:.1f} ms")
        self.model_latency_label.setText(f"{metrics.model_latency_ms:.1f} ms")
        self.coverage_label.setText(f"{metrics.mask_coverage:.3f}")
        self.frames_label.setText(f"{metrics.processed_frames} / {metrics.skipped_frames}")

    @Slot(str)
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot()
    def run_sweep(self) -> None:
        latest = self.frame_buffer.get_latest() if self.frame_buffer is not None else None
        frame = latest.frame if latest is not None else None
        if frame is None:
            QMessageBox.warning(self, "No frame", "Start a camera or video source first.")
            return
        try:
            sweep_config = self.current_sweep_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid sweep space", str(exc))
            return
        for backend_name in sweep_config.model_ids:
            unavailable_message = backend_unavailable_message(backend_name)
            if unavailable_message is not None:
                QMessageBox.warning(self, "Backend unavailable", unavailable_message)
                return
        if any(model_id != self.segmenter.model_name for model_id in sweep_config.model_ids):
            self.stop_worker(wait=True)
        self.sweep_button.setEnabled(False)
        self.sweep_table.setRowCount(0)
        self.sweep_worker = SweepWorker(self.segmenter, frame, self.current_settings(), sweep_config)
        self.sweep_worker.finished_with_results.connect(self.populate_sweep)
        self.sweep_worker.status.connect(self.set_status)
        self.sweep_worker.finished.connect(lambda: self.sweep_button.setEnabled(True))
        self.sweep_worker.start()

    @Slot(list)
    def populate_sweep(self, rows: list[dict]) -> None:
        self.sweep_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row["model_id"]),
                str(row["prompt"]),
                str(row["skip_frames"]),
                f'{row["infer_scale"]:.2f}',
                f'{row["threshold"]:.2f}',
                f'{row["latency_ms"]:.1f}',
                f'{row["mask_coverage"]:.3f}',
            ]
            for col, value in enumerate(values):
                self.sweep_table.setItem(row_index, col, QTableWidgetItem(value))
        self.set_status("Sweep complete")

    @Slot()
    def run_benchmark(self) -> None:
        try:
            sweep_config = self.current_sweep_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid sweep space", str(exc))
            return
        for backend_name in sweep_config.model_ids:
            unavailable_message = backend_unavailable_message(backend_name)
            if unavailable_message is not None:
                QMessageBox.warning(self, "Backend unavailable", unavailable_message)
                return
        self.stop_worker(wait=True)
        self.video_position.setText("benchmark")
        self.set_hyperparameter_controls_enabled(False)
        self.benchmark_button.setEnabled(False)
        self.save_benchmark_button.setEnabled(False)
        self.latest_benchmark_results = []
        self.benchmark_table.setRowCount(0)
        self.benchmark_progress.setRange(0, sweep_config.combination_count)
        self.benchmark_progress.setValue(0)
        self.benchmark_progress_label.setText("loading benchmark frames...")
        self.benchmark_worker = BenchmarkWorker(
            self.segmenter,
            self.current_settings(),
            self.benchmark_frames.value(),
            sweep_config,
        )
        self.benchmark_worker.row_ready.connect(self.append_benchmark_row)
        self.benchmark_worker.progress.connect(self.update_benchmark_progress)
        self.benchmark_worker.preview.connect(self.update_benchmark_preview)
        self.benchmark_worker.finished_with_results.connect(self.populate_benchmark)
        self.benchmark_worker.status.connect(self.set_status)
        self.benchmark_worker.finished.connect(self.finish_benchmark_run)
        self.benchmark_worker.start()

    @Slot()
    def finish_benchmark_run(self) -> None:
        self.set_hyperparameter_controls_enabled(True)
        self.benchmark_button.setEnabled(True)
        self.save_benchmark_button.setEnabled(bool(self.latest_benchmark_results))

    @Slot(list)
    def populate_benchmark(self, rows: list[dict]) -> None:
        self.latest_benchmark_results = rows
        self.benchmark_progress.setValue(self.benchmark_progress.maximum())
        self.benchmark_progress_label.setText(f"complete: {len(rows)} combinations")
        self.save_benchmark_button.setEnabled(bool(rows))
        self.set_status("Benchmark complete")

    @Slot()
    def save_benchmark_results(self) -> None:
        if not self.latest_benchmark_results:
            QMessageBox.warning(self, "No benchmark results", "Run a benchmark before saving results.")
            self.save_benchmark_button.setEnabled(False)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save benchmark result",
            str(Path.cwd() / "benchmark_result.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        output_path = Path(path)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        fieldnames = [
            "backend",
            "prompt",
            "skip_frames",
            "infer_scale",
            "threshold",
            "non_target_leakage",
            "target_damage",
            "fps",
            "latency_ms",
            "model_latency_ms",
            "model_calls",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
        try:
            with output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.latest_benchmark_results:
                    output_row = {field: row.get(field, "") for field in fieldnames}
                    output_row["backend"] = row.get("model_id", "")
                    writer.writerow(output_row)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return

        self.set_status(f"Benchmark result saved: {output_path}")

    @Slot(dict)
    def append_benchmark_row(self, row: dict) -> None:
        row_index = self.benchmark_table.rowCount()
        self.benchmark_table.insertRow(row_index)
        values = [
            str(row["model_id"]),
            str(row["prompt"]),
            str(row["skip_frames"]),
            f'{row["infer_scale"]:.2f}',
            f'{row["threshold"]:.2f}',
            f'{row["non_target_leakage"]:.4f}',
            f'{row["target_damage"]:.4f}',
            f'{row["fps"]:.2f}',
            f'{row["latency_ms"]:.1f}',
        ]
        for col, value in enumerate(values):
            self.benchmark_table.setItem(row_index, col, QTableWidgetItem(value))
        self.benchmark_table.scrollToBottom()
        self.benchmark_progress.setValue(row_index + 1)

    @Slot(int, int, str)
    def update_benchmark_progress(self, current: int, total: int, label: str) -> None:
        self.benchmark_progress.setRange(0, total)
        self.benchmark_progress.setValue(current - 1)
        compact_label = label
        if len(compact_label) > 96:
            compact_label = f"{compact_label[:93]}..."
        self.benchmark_progress_label.setText(f"running {current}/{total}\n{compact_label}")

    @Slot(QImage, QImage)
    def update_benchmark_preview(self, input_image: QImage, output_image: QImage) -> None:
        self.input_pane.set_image(input_image)
        self.output_pane.set_image(output_image)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.stop_worker(wait=True)
        for thread in list(self.retiring_threads):
            if thread.isRunning():
                thread.wait()
            self.release_retired_thread(thread)
        for thread in [self.sweep_worker, self.benchmark_worker]:
            if thread is not None and thread.isRunning():
                thread.wait()
        self.segmenter.close()
        event.accept()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide6 real-time prompt segmentation MVP.")
    parser.add_argument("--backend", default=BACKEND_CLIPSEG, choices=BACKEND_PRESETS)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv)
    segmenter = PromptSegmenter(args.backend, args.device)
    window = MainWindow(segmenter)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
