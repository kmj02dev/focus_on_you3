# Focus On You

Real-time prompt-based video filtering MVP.

The app lets you enter a text prompt while using a camera or watching a video, then preserves the prompted target and blurs, removes, dims, or previews the non-target pixels.

Core metrics:

- `non_target_leakage = FP / (FP + TN)`: range `0.0` to `1.0`; lower is better.
- `target_damage = FN / (TP + FN)`: range `0.0` to `1.0`; lower is better.
- `FPS`: range `0.0` to `+inf`; higher is better.
- `latency_ms`: range `0.0` to `+inf`; lower is better.

Metric definitions:

- `TP`: target pixels correctly preserved.
- `FP`: non-target pixels incorrectly preserved.
- `TN`: non-target pixels correctly removed, blurred, or suppressed.
- `FN`: target pixels incorrectly removed, blurred, or suppressed.

`non_target_leakage` measures how much background/non-target content leaks through the filter. A high value means pixels that should have been hidden are still visible, which is risky for privacy or distraction removal.

`target_damage` measures how much of the prompted target is damaged by the filter. A high value means the object or region the user wanted to keep is being removed or blurred too aggressively.

For this project, `non_target_leakage`, `target_damage`, and `latency_ms` should be as low as possible, while `FPS` should be as high as possible. The practical trade-off is that lowering `non_target_leakage` often increases `target_damage`, while protecting the target more generously can increase leakage.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Desktop MVP

The PySide6 app is implemented in a single file:

```text
main.py
```

It supports:

- camera start/stop
- video play/pause/stop/seek
- editable backend selector
- prompt-based target preservation
- live effect preview on the same frame as the input feed
- non-target blur/remove/dim/mask preview in sweeps and GT benchmark runs
- live FPS and latency metrics
- optimization sweep on the current frame
- CamVid GT benchmark sweep with visible progress and every-frame preview

Live playback uses a split pipeline: one worker captures camera/video frames and another worker runs inference on the latest available frame. This keeps input playback responsive even when model inference is slower than the capture rate. During live playback, `Processed` shows the same video frame as `Input` with the latest model mask rendered as the selected blur/remove/dim/mask effect preview.

In the GT benchmark controls, `Frames = All` uses every available benchmark frame.

The main resolution hyperparameter is `scale`, not absolute width. `scale` controls the inference resolution relative to the original frame:

- `scale = 0.25`: infer at 25% of the original frame width and height.
- `scale = 0.50`: infer at 50% of the original frame width and height.
- `scale = 1.00`: infer at the original frame size.

Lower `scale` usually improves FPS and latency, but can increase `non_target_leakage` or `target_damage`. Higher `scale` usually improves mask quality, but costs more latency.

`Skip frames` controls how many live camera/video frames are skipped between model calls. `0` runs inference on every frame, `1` processes every other frame, and `4` processes one frame after skipping four. Larger values should improve playback responsiveness, but per-call `model_latency_ms` can still vary because the GPU may idle, clocks may change, and UI/video decode work continues between inference calls.

The sweep space is editable in the UI. Backend is selected from a dropdown; the other fields accept comma-separated values:

- `Target prompts`: default `road`
- `Backend`: default `CLIPSeg`
- `Scales`: default `0.25,0.50,0.75,1.00`
- `Thresholds`: default `0.35,0.50,0.65`
- `Skip frames`: default `0`

The app shows the resulting combination count before running the current-frame sweep or CamVid benchmark. During CamVid benchmark runs, hyperparameter controls are disabled so the run uses a fixed snapshot of the sweep space.

For benchmark `Skip frames`, the app still evaluates every benchmark frame. Skipped frames reuse the most recent mask, matching the live playback path where inference can run slower than video display. Benchmark `latency_ms` is reported per evaluated frame, while saved CSV files also include `model_latency_ms` for model-call latency.

The Backend dropdown exposes these user-facing pipelines:

- `CLIPSeg`: text-to-mask baseline using `CIDAS/clipseg-rd64-refined`.
- `YOLO-World small box-only`: open-vocabulary detection boxes converted to a binary mask.
- `YOLO11n-seg`: COCO-class segmentation using `weights/yolo11n-seg.pt`. Useful prompts include `person`, `car`, `vehicle`, `dog`, `cat`, `animal`, `사람`, `자동차`, and `개`.
- `YOLO-World small + SAM2 tiny tracking`: visible in the dropdown, but blocked until SAM2 tiny runtime/checkpoint is installed.
- `SAM 3`: visible in the dropdown, but blocked until SAM 3 runtime/checkpoint is installed.
- `Grounding DINO tiny + SAM2 tiny`: visible in the dropdown, but blocked until Grounding DINO and SAM2 tiny runtimes/checkpoints are installed.

The Model panel always shows the currently applied backend/device. If a different backend is selected but `Apply Model` has not been pressed yet, it is shown as pending.

Use `Reset Hyperparameters` to restore the default prompt/effect/realtime/benchmark settings.

## CamVid Benchmark

The benchmark helper lives under `benchmark/`:

```bash
git clone --depth 1 https://github.com/lih627/CamVid.git datasets/CamVid
python benchmark/prepare_camvid_benchmark.py --class-name road --frames 30 --width 480
python main.py
```

The helper generates:

- `benchmark_data/camvid_road_0001TP.mp4`
- `benchmark_gt/camvid_road_0001TP/`

These files are used by the app's `Run CamVid Road Benchmark` button.
