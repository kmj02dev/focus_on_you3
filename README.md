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
- editable model selector
- prompt-based target preservation
- non-target blur/remove/dim/mask preview
- live FPS and latency metrics
- optimization sweep on the current frame
- CamVid GT benchmark sweep with visible progress and every-frame preview

In the GT benchmark controls, `Frames = All` uses every available benchmark frame.

The main resolution hyperparameter is `scale`, not absolute width. `scale` controls the inference resolution relative to the original frame:

- `scale = 0.25`: infer at 25% of the original frame width and height.
- `scale = 0.50`: infer at 50% of the original frame width and height.
- `scale = 1.00`: infer at the original frame size.

Lower `scale` usually improves FPS and latency, but can increase `non_target_leakage` or `target_damage`. Higher `scale` usually improves mask quality, but costs more latency.

The sweep space is editable in the UI as comma-separated values:

- `Scales CSV`: default `0.25,0.50,0.75,1.00`
- `Thresholds CSV`: default `0.35,0.50,0.65`

The app shows the resulting combination count before running the current-frame sweep or CamVid benchmark.

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
