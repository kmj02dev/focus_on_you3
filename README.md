# Focus On You

실시간 프롬프트 기반 비디오 필터링 데스크톱 MVP입니다.

카메라 또는 비디오 입력에서 사용자가 보존하고 싶은 대상을 텍스트 프롬프트로 입력하면, 모델이 해당 대상의 마스크를 예측하고 나머지 영역을 흐림, 제거, 어둡게 처리, 또는 마스크 미리보기 형태로 시각화합니다.

예를 들어 `person`, `car`, `road` 같은 프롬프트를 입력하면 앱은 해당 대상은 유지하고 비대상 영역은 선택한 효과로 억제합니다.

## 주요 기능

- 카메라 입력 시작/정지
- 비디오 파일 재생/일시정지/정지/시크
- 텍스트 프롬프트 기반 target preservation
- 원본 화면과 처리 결과 화면 동시 표시
- 비대상 영역 처리 모드 지원: `blur`, `remove`, `dim`, `mask`
- 실시간 FPS, latency, model latency, mask coverage 표시
- 추론 해상도, threshold, skip frame 조절
- 현재 프레임 기준 하이퍼파라미터 sweep
- CamVid road ground truth 기반 벤치마크
- 벤치마크 결과 CSV 저장

## 기술 스택

- Python
- PySide6
- OpenCV
- NumPy
- PyTorch
- Transformers
- Ultralytics
- CLIPSeg
- YOLO-World
- YOLO11n-seg

## 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

실행 옵션:

```bash
python main.py --backend CLIPSeg --device cpu
```

사용 가능한 backend 값은 다음과 같습니다.

- `CLIPSeg`
- `YOLO-World small box-only`
- `YOLO11n-seg`
- `YOLO-World small + SAM2 tiny tracking`
- `SAM 3`
- `Grounding DINO tiny + SAM2 tiny`

단, 현재 실제 실행 가능한 백엔드는 `CLIPSeg`, `YOLO-World small box-only`, `YOLO11n-seg`입니다. SAM2, SAM3, Grounding DINO 조합은 UI에 표시되지만 런타임과 체크포인트가 아직 연결되어 있지 않아 실행이 차단됩니다.

## 프로젝트 구조

```text
.
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── opencv_vtest.avi
├── weights/
│   ├── yolo11n-seg.pt
│   ├── yolov8n.pt
│   └── FastSAM-s.pt
├── benchmark/
│   └── prepare_camvid_benchmark.py
├── benchmark_data/
│   └── camvid_road_0001TP.mp4
└── benchmark_gt/
    └── camvid_road_0001TP/
```

앱의 핵심 구현은 `main.py` 한 파일에 들어 있습니다. GUI, 모델 로딩, 프레임 캡처, 추론 worker, sweep, benchmark 로직이 모두 포함되어 있습니다.

## 사용 흐름

1. 앱을 실행합니다.
2. `Start Camera`로 카메라를 시작하거나 `Browse`로 비디오를 선택한 뒤 `Play Video`를 누릅니다.
3. `Target prompt`에 보존할 대상을 입력합니다.
4. `Mode`에서 비대상 영역 처리 방식을 선택합니다.
5. `Threshold`, `Inference scale`, `Skip frames`를 조절하며 결과를 확인합니다.
6. 필요하면 `Run Sweep On Current Frame` 또는 `Run CamVid Road Benchmark`를 실행합니다.

## 모델 백엔드

### CLIPSeg

`CIDAS/clipseg-rd64-refined`를 사용하는 text-to-mask baseline입니다. 텍스트 프롬프트와 이미지를 함께 입력받아 프롬프트에 대응하는 영역의 segmentation mask를 생성합니다.

`road`처럼 일반 객체 detection class가 아닌 개념도 시도할 수 있기 때문에 CamVid road benchmark의 기본 backend로 사용하기 적합합니다.

### YOLO-World small box-only

Open-vocabulary detection 모델입니다. 입력 프롬프트에 해당하는 객체를 box 단위로 탐지한 뒤, 탐지 박스를 binary mask로 변환합니다.

정확한 segmentation mask가 아니라 bounding box 기반 mask이므로 빠른 detection baseline 성격이 강합니다.

### YOLO11n-seg

`weights/yolo11n-seg.pt`를 사용하는 COCO class 기반 segmentation backend입니다.

이 모델은 임의의 텍스트 프롬프트를 자유롭게 이해하는 모델이 아니라, COCO dataset에 정의된 class id를 기준으로 동작합니다. 그래서 프로젝트에서는 prompt를 COCO class id로 매핑합니다.

지원 예시:

- `person`, `people`, `human`, `사람`
- `car`, `vehicle`, `자동차`
- `dog`, `cat`, `animal`, `개`, `강아지`

주의할 점은 `road`가 COCO class가 아니라는 것입니다. 따라서 YOLO11n-seg는 road benchmark용이라기보다 person, car, dog 같은 COCO 객체 segmentation baseline에 가깝습니다.

## 핵심 하이퍼파라미터

### Threshold

모델이 예측한 mask probability를 binary mask로 바꾸는 기준값입니다.

- 낮은 threshold: target을 더 넓게 보존하지만 background leakage가 증가할 수 있음
- 높은 threshold: background 억제는 강해지지만 target 일부가 손상될 수 있음

### Inference scale

모델 추론 전에 프레임을 얼마나 축소할지 정하는 비율입니다.

예를 들어 원본 프레임이 `1280 x 720`이라면:

```text
scale = 1.00 -> 1280 x 720으로 추론
scale = 0.50 -> 640 x 360으로 추론
scale = 0.25 -> 320 x 180으로 추론
```

낮은 scale은 FPS와 latency에 유리하지만 mask 품질이 떨어질 수 있습니다. 높은 scale은 mask 품질에는 유리하지만 계산량이 증가합니다.

### Skip frames

몇 프레임마다 모델 추론을 수행할지 조절합니다.

```text
skip_frames = 0 -> 모든 프레임에서 추론
skip_frames = 1 -> 한 프레임 건너 한 번 추론
skip_frames = 4 -> 네 프레임 건너 한 번 추론
```

건너뛴 프레임에서는 가장 최근 mask를 재사용합니다. 실시간 재생 반응성을 높이기 위한 옵션입니다.

## 평가지표

이 프로젝트는 target을 잘 보존하면서 non-target을 최대한 억제하는 것을 목표로 합니다.

### Confusion matrix 기준

- `TP`: target pixel을 올바르게 보존
- `FP`: non-target pixel을 잘못 보존
- `TN`: non-target pixel을 올바르게 억제
- `FN`: target pixel을 잘못 억제

### Core metrics

```text
non_target_leakage = FP / (FP + TN)
target_damage = FN / (TP + FN)
```

- `non_target_leakage`: 비대상 영역이 얼마나 새어 나왔는지 측정합니다. 낮을수록 좋습니다.
- `target_damage`: 보존해야 할 대상이 얼마나 손상되었는지 측정합니다. 낮을수록 좋습니다.
- `FPS`: 초당 처리 프레임 수입니다. 높을수록 좋습니다.
- `latency_ms`: 프레임 처리 지연 시간입니다. 낮을수록 좋습니다.

실제 trade-off는 다음과 같습니다.

- non-target leakage를 줄이려고 threshold를 높이면 target damage가 증가할 수 있습니다.
- target을 넓게 보호하려고 threshold를 낮추면 background leakage가 증가할 수 있습니다.
- inference scale을 낮추면 속도는 빨라지지만 mask 품질이 저하될 수 있습니다.

## CamVid 벤치마크

CamVid road class를 기준으로 GT mask를 만들고, 예측 mask와 비교해 정량 평가를 수행합니다.

이미 준비된 파일:

- `benchmark_data/camvid_road_0001TP.mp4`
- `benchmark_gt/camvid_road_0001TP/`

새로 생성하려면 CamVid dataset을 받은 뒤 다음 명령을 실행합니다.

```bash
git clone --depth 1 https://github.com/lih627/CamVid.git datasets/CamVid
python benchmark/prepare_camvid_benchmark.py --class-name road --frames 30 --width 480
python main.py
```

앱에서 `Run CamVid Road Benchmark` 버튼을 누르면 현재 sweep 설정에 따라 benchmark가 실행됩니다.

## Sweep

앱의 sweep 영역에서는 다음 값을 조합해 현재 프레임 또는 CamVid benchmark에서 성능을 비교할 수 있습니다.

- target prompts
- backend
- scales
- thresholds
- skip frames

기본값:

```text
Target prompts: road
Backend: CLIPSeg
Scales: 0.25,0.50,0.75,1.00
Thresholds: 0.35,0.50,0.65
Skip frames: 0
```

앱은 실행 전에 조합 수를 표시합니다. benchmark 중에는 설정이 바뀌지 않도록 관련 컨트롤을 비활성화합니다.

## 구현 특징

실시간 재생은 capture worker와 inference worker를 분리한 구조입니다.

- capture worker: 카메라 또는 비디오 프레임을 계속 읽고 UI에 표시
- inference worker: 최신 프레임을 가져와 모델 추론 수행
- mask buffer: 가장 최근 예측 mask를 저장하고 재사용

이 구조 덕분에 모델 추론이 느려져도 입력 영상 재생이 완전히 멈추지 않고, 처리 화면은 최신 mask를 기반으로 계속 갱신됩니다.

## 참고 사항

- CLIPSeg는 최초 실행 시 Hugging Face model download가 필요할 수 있습니다.
- CUDA 환경이 불안정하면 `--device cpu`로 실행할 수 있습니다.
- YOLO11n-seg는 COCO class 기반이므로 `road` 같은 class는 지원하지 않습니다.
- `weights/FastSAM-s.pt`는 포함되어 있지만 현재 UI의 SAM 계열 backend와 직접 연결되어 있지는 않습니다.
