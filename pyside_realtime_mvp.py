#!/usr/bin/env python3
"""PySide6 single-file MVP for real-time prompt-based video filtering.

Install:
    pip install -r requirements.txt

Run:
    python pyside_realtime_mvp.py

The MVP keeps all application code in this file. It supports camera/video input,
text-prompted segmentation, non-target blur/removal, live FPS/latency metrics,
parameter sweeps, and a CamVid road GT benchmark when benchmark_data/ and
benchmark_gt/ are present.
"""

from __future__ import annotations

import argparse
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
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
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


MODEL_NAME = "CIDAS/clipseg-rd64-refined"
CAMVID_VIDEO = Path("benchmark_data/camvid_road_0001TP.mp4")
CAMVID_GT_DIR = Path("benchmark_gt/camvid_road_0001TP")


@dataclass
class RuntimeSettings:
    prompt: str = "road"
    mode: str = "blur"
    threshold: float = 0.5
    infer_width: int = 192
    blur_kernel: int = 35
    frame_stride: int = 1
    fps_cap: int = 15


@dataclass
class FrameMetrics:
    fps: float
    latency_ms: float
    model_latency_ms: float
    mask_coverage: float
    processed_frames: int
    skipped_frames: int
    infer_width: int


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


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


class PromptSegmenter:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor: CLIPSegProcessor | None = None
        self.model: CLIPSegForImageSegmentation | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        with self.lock:
            if self.model is not None and self.processor is not None:
                return
            self.processor = CLIPSegProcessor.from_pretrained(self.model_name)
            self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True

    def set_model(self, model_name: str, device: str) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Model name cannot be empty")
        with self.lock:
            self.processor = None
            self.model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.model_name = model_name
            self.device = torch.device(device)

    @torch.inference_mode()
    def predict_mask(self, frame: np.ndarray, prompt: str, infer_width: int) -> tuple[np.ndarray, float]:
        self.load()
        assert self.processor is not None
        assert self.model is not None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(1.0, infer_width / max(1, width))
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


class VideoWorker(QThread):
    input_frame = Signal(QImage)
    output_frame = Signal(QImage)
    metrics = Signal(object)
    status = Signal(str)
    position_changed = Signal(int, int)
    video_info = Signal(int, float)

    def __init__(self, segmenter: PromptSegmenter) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.settings = RuntimeSettings()
        self.settings_lock = threading.Lock()
        self.source: int | str = 0
        self.running = False
        self.paused = False
        self.seek_frame: int | None = None
        self.control_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_lock = threading.Lock()

    def configure(self, settings: RuntimeSettings) -> None:
        with self.settings_lock:
            self.settings = settings

    def set_source(self, source: int | str) -> None:
        self.source = source
        with self.control_lock:
            self.paused = False
            self.seek_frame = None

    def stop(self) -> None:
        self.running = False

    def pause(self) -> None:
        with self.control_lock:
            self.paused = True

    def resume(self) -> None:
        with self.control_lock:
            self.paused = False

    def seek(self, frame_index: int) -> None:
        with self.control_lock:
            self.seek_frame = max(0, frame_index)

    def snapshot(self) -> np.ndarray | None:
        with self.latest_frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def run(self) -> None:
        self.running = True
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status.emit(f"Could not open source: {self.source}")
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

            with self.latest_frame_lock:
                self.latest_frame = frame.copy()
            self.input_frame.emit(bgr_to_qimage(frame))
            self.position_changed.emit(frame_index, total_frames)

            with self.settings_lock:
                settings = RuntimeSettings(**self.settings.__dict__)

            should_process = frame_index % max(1, settings.frame_stride) == 0
            if should_process and settings.prompt.strip():
                started = time.perf_counter()
                try:
                    mask, model_latency_ms = self.segmenter.predict_mask(
                        frame,
                        settings.prompt.strip(),
                        settings.infer_width,
                    )
                    output = apply_effect(frame, mask, settings.threshold, settings.mode, settings.blur_kernel)
                    processed += 1
                    self.output_frame.emit(bgr_to_qimage(output))
                    elapsed = time.perf_counter() - started
                    now = time.perf_counter()
                    fps = 1.0 / max(1e-6, now - last_emit)
                    last_emit = now
                    self.metrics.emit(
                        FrameMetrics(
                            fps=fps,
                            latency_ms=elapsed * 1000.0,
                            model_latency_ms=model_latency_ms,
                            mask_coverage=float((mask >= settings.threshold).mean()),
                            processed_frames=processed,
                            skipped_frames=skipped,
                            infer_width=settings.infer_width,
                        )
                    )
                except Exception as exc:
                    self.status.emit(str(exc))
            else:
                skipped += 1

            frame_index += 1
            if paused:
                continue
            fps_cap = max(1, settings.fps_cap)
            target_dt = 1.0 / fps_cap
            spent = time.perf_counter() - loop_started
            if spent < target_dt:
                time.sleep(target_dt - spent)

        cap.release()
        self.status.emit("stopped")


class SweepWorker(QThread):
    finished_with_results = Signal(list)
    status = Signal(str)

    def __init__(self, segmenter: PromptSegmenter, frame: np.ndarray, settings: RuntimeSettings) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.frame = frame
        self.settings = settings

    def run(self) -> None:
        try:
            results = []
            for infer_width in [128, 192, 256, 320]:
                for threshold in [0.35, 0.50, 0.65]:
                    started = time.perf_counter()
                    mask, model_latency_ms = self.segmenter.predict_mask(
                        self.frame,
                        self.settings.prompt,
                        infer_width,
                    )
                    _ = apply_effect(self.frame, mask, threshold, self.settings.mode, self.settings.blur_kernel)
                    total_ms = (time.perf_counter() - started) * 1000.0
                    results.append(
                        {
                            "infer_width": infer_width,
                            "threshold": threshold,
                            "latency_ms": total_ms,
                            "model_latency_ms": model_latency_ms,
                            "mask_coverage": float((mask >= threshold).mean()),
                        }
                    )
            self.finished_with_results.emit(results)
        except Exception as exc:
            self.status.emit(str(exc))


class BenchmarkWorker(QThread):
    finished_with_results = Signal(list)
    row_ready = Signal(dict)
    progress = Signal(int, int, str)
    preview = Signal(QImage, QImage)
    status = Signal(str)

    def __init__(self, segmenter: PromptSegmenter, settings: RuntimeSettings, max_frames: int) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.settings = settings
        self.max_frames = max_frames

    def run(self) -> None:
        try:
            if not CAMVID_VIDEO.exists() or not CAMVID_GT_DIR.exists():
                raise RuntimeError("CamVid benchmark files not found. Run prepare_camvid_benchmark.py first.")

            frames: list[np.ndarray] = []
            gt_masks: list[np.ndarray] = []
            cap = cv2.VideoCapture(str(CAMVID_VIDEO))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {CAMVID_VIDEO}")

            frame_index = 0
            while len(frames) < self.max_frames:
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
            combinations = [(width, threshold) for width in [128, 192, 256] for threshold in [0.35, 0.50, 0.65]]
            total_combinations = len(combinations)
            for combo_index, (infer_width, threshold) in enumerate(combinations, start=1):
                self.progress.emit(
                    combo_index,
                    total_combinations,
                    f"width={infer_width}, threshold={threshold:.2f}",
                )
                tp_total = fp_total = tn_total = fn_total = 0
                latencies = []
                started = time.perf_counter()
                for frame_offset, (frame, gt_mask) in enumerate(zip(frames, gt_masks)):
                    mask, model_latency_ms = self.segmenter.predict_mask(
                        frame,
                        self.settings.prompt,
                        infer_width,
                    )
                    output = apply_effect(frame, mask, threshold, self.settings.mode, self.settings.blur_kernel)
                    self.preview.emit(bgr_to_qimage(frame), bgr_to_qimage(output))
                    latencies.append(model_latency_ms)
                    tp, fp, tn, fn = binary_confusion_counts(mask >= threshold, gt_mask)
                    tp_total += tp
                    fp_total += fp
                    tn_total += tn
                    fn_total += fn
                elapsed = time.perf_counter() - started
                leakage, damage = metric_rates(tp_total, fp_total, tn_total, fn_total)
                row = {
                    "infer_width": infer_width,
                    "threshold": threshold,
                    "non_target_leakage": leakage,
                    "target_damage": damage,
                    "fps": float(len(frames) / elapsed) if elapsed > 0 else 0.0,
                    "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                    "tp": tp_total,
                    "fp": fp_total,
                    "tn": tn_total,
                    "fn": fn_total,
                }
                results.append(row)
                self.row_ready.emit(row)
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
        self.worker: VideoWorker | None = None
        self.sweep_worker: SweepWorker | None = None
        self.benchmark_worker: BenchmarkWorker | None = None

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
        self.model_name.setEditable(True)
        self.model_name.addItems(
            [
                "CIDAS/clipseg-rd64-refined",
                "CIDAS/clipseg-rd64",
                "CIDAS/clipseg-rd16",
            ]
        )
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
        model_form.addRow("Model ID", self.model_name)
        model_form.addRow("Device", self.device_name)
        model_form.addRow(self.apply_model_button)
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
        self.prompt = QLineEdit("road")
        self.mode = QComboBox()
        self.mode.addItems(["blur", "remove", "dim", "mask"])
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(5, 95)
        self.threshold.setValue(50)
        self.threshold_label = QLabel("0.50")
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold)
        threshold_row.addWidget(self.threshold_label)
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(3, 99)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(35)
        prompt_form.addRow("Target prompt", self.prompt)
        prompt_form.addRow("Mode", self.mode)
        prompt_form.addRow("Threshold", threshold_row)
        prompt_form.addRow("Blur kernel", self.blur_kernel)
        controls.addWidget(prompt_box)

        realtime_box = QGroupBox("Realtime Controls")
        realtime_form = QFormLayout(realtime_box)
        self.infer_width = QComboBox()
        self.infer_width.addItems(["128", "192", "256", "320", "480"])
        self.infer_width.setCurrentText("192")
        self.frame_stride = QSpinBox()
        self.frame_stride.setRange(1, 30)
        self.frame_stride.setValue(1)
        self.fps_cap = QSpinBox()
        self.fps_cap.setRange(1, 60)
        self.fps_cap.setValue(15)
        realtime_form.addRow("Inference width", self.infer_width)
        realtime_form.addRow("Process every N frames", self.frame_stride)
        realtime_form.addRow("Target FPS cap", self.fps_cap)
        controls.addWidget(realtime_box)

        sweep_box = QGroupBox("Optimization Sweep")
        sweep_layout = QVBoxLayout(sweep_box)
        self.sweep_button = QPushButton("Run Sweep On Current Frame")
        self.sweep_table = QTableWidget(0, 4)
        self.sweep_table.setHorizontalHeaderLabels(["width", "thr", "lat ms", "coverage"])
        self.sweep_table.setMinimumHeight(120)
        self.sweep_table.setMaximumHeight(170)
        sweep_layout.addWidget(self.sweep_button)
        sweep_layout.addWidget(self.sweep_table)
        controls.addWidget(sweep_box)

        benchmark_box = QGroupBox("GT Benchmark")
        benchmark_layout = QVBoxLayout(benchmark_box)
        self.benchmark_frames = QSpinBox()
        self.benchmark_frames.setRange(1, 30)
        self.benchmark_frames.setValue(10)
        self.benchmark_button = QPushButton("Run CamVid Road Benchmark")
        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setRange(0, 9)
        self.benchmark_progress.setValue(0)
        self.benchmark_progress.setTextVisible(True)
        self.benchmark_progress_label = QLabel("idle")
        self.benchmark_table = QTableWidget(0, 6)
        self.benchmark_table.setHorizontalHeaderLabels(["width", "thr", "leak", "damage", "FPS", "lat ms"])
        self.benchmark_table.setMinimumHeight(150)
        self.benchmark_table.setMaximumHeight(210)
        benchmark_form = QFormLayout()
        benchmark_form.addRow("Frames", self.benchmark_frames)
        benchmark_layout.addLayout(benchmark_form)
        benchmark_layout.addWidget(self.benchmark_button)
        benchmark_layout.addWidget(self.benchmark_progress)
        benchmark_layout.addWidget(self.benchmark_progress_label)
        benchmark_layout.addWidget(self.benchmark_table)
        controls.addWidget(benchmark_box)

        self.status_label = QLabel(f"Device: {self.segmenter.device}. Model loads on first inference.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)
        controls.addStretch(1)

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
        self.sweep_button.clicked.connect(self.run_sweep)
        self.benchmark_button.clicked.connect(self.run_benchmark)
        for widget in [
            self.prompt,
            self.mode,
            self.infer_width,
            self.frame_stride,
            self.fps_cap,
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

    def current_settings(self) -> RuntimeSettings:
        return RuntimeSettings(
            prompt=self.prompt.text().strip() or "target",
            mode=self.mode.currentText(),
            threshold=self.threshold.value() / 100.0,
            infer_width=int(self.infer_width.currentText()),
            blur_kernel=self.blur_kernel.value(),
            frame_stride=self.frame_stride.value(),
            fps_cap=self.fps_cap.value(),
        )

    @Slot()
    def push_settings(self, *args) -> None:  # type: ignore[no-untyped-def]
        if self.worker is not None:
            self.worker.configure(self.current_settings())

    @Slot(int)
    def on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value / 100.0:.2f}")
        self.push_settings()

    @Slot()
    def apply_model(self) -> None:
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            QMessageBox.warning(self, "Sweep running", "Wait for the current sweep before switching models.")
            return
        if self.benchmark_worker is not None and self.benchmark_worker.isRunning():
            QMessageBox.warning(self, "Benchmark running", "Wait for the current benchmark before switching models.")
            return
        self.stop_worker()
        try:
            self.segmenter.set_model(self.model_name.currentText(), self.device_name.currentText())
        except Exception as exc:
            QMessageBox.warning(self, "Model switch failed", str(exc))
            return
        self.set_status(
            f"Model set to {self.segmenter.model_name} on {self.segmenter.device}. "
            "It will load on the next inference."
        )

    @Slot()
    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", str(Path.cwd()), "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path.setText(path)

    def start_source(self, source: int | str) -> None:
        self.stop_worker()
        self.worker = VideoWorker(self.segmenter)
        self.worker.set_source(source)
        self.worker.configure(self.current_settings())
        self.worker.input_frame.connect(self.input_pane.set_image)
        self.worker.output_frame.connect(self.output_pane.set_image)
        self.worker.metrics.connect(self.update_metrics)
        self.worker.status.connect(self.set_status)
        self.worker.video_info.connect(self.update_video_info)
        self.worker.position_changed.connect(self.update_video_position)
        self.worker.start()

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
        if self.worker is not None and self.worker.source == path:
            self.worker.resume()
            self.set_status("video resumed")
            return
        self.start_source(path)

    @Slot()
    def pause_video(self) -> None:
        if self.worker is not None and isinstance(self.worker.source, str):
            self.worker.pause()
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
        if self.worker is not None and isinstance(self.worker.source, str):
            self.worker.seek(self.video_seek.value())

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
    def stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(1500)
        self.worker = None

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
        frame = self.worker.snapshot() if self.worker is not None else None
        if frame is None:
            QMessageBox.warning(self, "No frame", "Start a camera or video source first.")
            return
        self.sweep_button.setEnabled(False)
        self.sweep_table.setRowCount(0)
        self.sweep_worker = SweepWorker(self.segmenter, frame, self.current_settings())
        self.sweep_worker.finished_with_results.connect(self.populate_sweep)
        self.sweep_worker.status.connect(self.set_status)
        self.sweep_worker.finished.connect(lambda: self.sweep_button.setEnabled(True))
        self.sweep_worker.start()

    @Slot(list)
    def populate_sweep(self, rows: list[dict]) -> None:
        self.sweep_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row["infer_width"]),
                f'{row["threshold"]:.2f}',
                f'{row["latency_ms"]:.1f}',
                f'{row["mask_coverage"]:.3f}',
            ]
            for col, value in enumerate(values):
                self.sweep_table.setItem(row_index, col, QTableWidgetItem(value))
        self.set_status("Sweep complete")

    @Slot()
    def run_benchmark(self) -> None:
        self.benchmark_button.setEnabled(False)
        self.benchmark_table.setRowCount(0)
        self.benchmark_progress.setRange(0, 9)
        self.benchmark_progress.setValue(0)
        self.benchmark_progress_label.setText("loading benchmark frames...")
        self.benchmark_worker = BenchmarkWorker(
            self.segmenter,
            self.current_settings(),
            self.benchmark_frames.value(),
        )
        self.benchmark_worker.row_ready.connect(self.append_benchmark_row)
        self.benchmark_worker.progress.connect(self.update_benchmark_progress)
        self.benchmark_worker.preview.connect(self.update_benchmark_preview)
        self.benchmark_worker.finished_with_results.connect(self.populate_benchmark)
        self.benchmark_worker.status.connect(self.set_status)
        self.benchmark_worker.finished.connect(lambda: self.benchmark_button.setEnabled(True))
        self.benchmark_worker.start()

    @Slot(list)
    def populate_benchmark(self, rows: list[dict]) -> None:
        self.benchmark_progress.setValue(self.benchmark_progress.maximum())
        self.benchmark_progress_label.setText(f"complete: {len(rows)} combinations")
        self.set_status("Benchmark complete")

    @Slot(dict)
    def append_benchmark_row(self, row: dict) -> None:
        row_index = self.benchmark_table.rowCount()
        self.benchmark_table.insertRow(row_index)
        values = [
            str(row["infer_width"]),
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
        self.benchmark_progress_label.setText(f"running {current}/{total}: {label}")

    @Slot(QImage, QImage)
    def update_benchmark_preview(self, input_image: QImage, output_image: QImage) -> None:
        self.input_pane.set_image(input_image)
        self.output_pane.set_image(output_image)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.stop_worker()
        for thread in [self.sweep_worker, self.benchmark_worker]:
            if thread is not None and thread.isRunning():
                thread.wait(1000)
        event.accept()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide6 real-time prompt segmentation MVP.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv)
    segmenter = PromptSegmenter(args.model, args.device)
    window = MainWindow(segmenter)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
