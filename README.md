# Focus On You

Real-time prompt-based video filtering MVP.

The app lets you enter a text prompt while using a camera or watching a video, then preserves the prompted target and blurs, removes, dims, or previews the non-target pixels.

Core metrics:

- `non_target_leakage = FP / (FP + TN)`
- `target_damage = FN / (TP + FN)`
- `FPS`
- `latency_ms`

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
