# Text-Prompted Segmentation

This project segments every supported image and video under `data/` using a text prompt, then writes masks and visualizations to `outputs/`.

## Usage

```bash
python segment_prompt.py "person"
```

Useful options:

```bash
python segment_prompt.py "car" --threshold 0.45 --frame-stride 2
```

To calculate the core GT-based metrics, provide ground-truth masks:

```bash
python segment_prompt.py "person" --gt-dir gt
```

- Image masks can be named like `gt/sample_input.png` or `gt/sample_input_mask.png`.
- Video frame masks can be named like `gt/opencv_vtest/frame_000000_mask.png`.

- Image outputs are saved under `outputs/images/<file_stem>/`.
- Video outputs are saved under `outputs/videos/<file_stem>/`.
- `outputs/summary.json` records processed files, prompt, non-target leakage, target damage, FPS, latency, mIoU, confusion counts, mask scores, and output paths.

Core metrics:

- `non_target_leakage = FP / (FP + TN)`
- `target_damage = FN / (TP + FN)`
- `FPS`
- `latency_ms`

The default model is `CIDAS/clipseg-rd64-refined`, which performs text-conditioned segmentation with CLIPSeg.

## Real-time Desktop MVP

Run the PySide6 single-file MVP:

```bash
python pyside_realtime_mvp.py
```

The app supports camera/video input, prompt-based target preservation, non-target blur/removal, live FPS/latency, per-frame optimization sweep, and CamVid GT benchmark metrics.
The model selector is editable, so you can switch between the bundled CLIPSeg presets or type another compatible Hugging Face model ID.

## CamVid Benchmark

CamVid can be prepared as a small video-style benchmark with binary GT masks:

```bash
git clone --depth 1 https://github.com/lih627/CamVid.git datasets/CamVid
python prepare_camvid_benchmark.py --class-name road --frames 30
python segment_prompt.py "road" --data-dir benchmark_data --gt-dir benchmark_gt --output-dir outputs/camvid_benchmark
```
