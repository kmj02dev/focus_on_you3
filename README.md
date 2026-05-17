# Focus On You

실시간 프롬프트 기반 비디오 필터링 애플리케이션 입니다. 카메라 또는 동영상 입력을 보면서 텍스트 프롬프트로 보존할 대상을 지정하면, 대상 영역은 유지하고 나머지 영역은 흐림, 제거, 어둡게 처리, 마스크 미리보기 방식으로 표시합니다.

예를 들어 프롬프트에 `road`를 입력하면 도로 영역을 중심으로 보존하고, 비대상 영역에는 선택한 효과를 적용합니다.

## 주요 기능

- PySide6 기반 데스크톱 GUI
- 카메라 입력 시작/중지
- 동영상 파일 재생/일시정지/정지/탐색
- 텍스트 프롬프트 기반 대상 영역 추정
- 비대상 영역 효과 적용: `blur`, `remove`, `dim`, `mask`
- 실시간 FPS, 지연 시간, 모델 지연 시간, 마스크 커버리지 표시
- 현재 프레임 기준 하이퍼파라미터 스윕
- CamVid 도로 클래스 기준 GT 벤치마크
- 벤치마크 결과 CSV 저장

## 프로젝트 구조

```text
.
├── main.py                              # PySide6 애플리케이션 본체
├── requirements.txt                     # Python 의존성 목록
├── data/
│   └── opencv_vtest.avi                 # 테스트용 비디오 파일
├── benchmark/
│   └── prepare_camvid_benchmark.py      # CamVid 벤치마크 데이터 생성 스크립트
├── benchmark_data/
│   └── camvid_road_0001TP.mp4           # 준비된 벤치마크 비디오
└── benchmark_gt/
    ├── camvid_road_0001TP/              # 프레임별 GT 마스크
    └── camvid_road_0001TP_metadata.json # 벤치마크 메타데이터
```

## 실행 환경

- Python 3.10 이상 권장
- 웹캠 또는 테스트용 비디오 파일
- GPU는 선택 사항입니다. CUDA 사용 가능 환경에서는 자동으로 `cuda`를 기본 장치로 선택하고, 그렇지 않으면 `cpu`를 사용합니다.

## 설치

가상환경 사용을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경 활성화 명령이 다릅니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 기본 실행

```bash
python main.py
```

백엔드와 장치를 명시해서 실행할 수도 있습니다.

```bash
python main.py --backend CLIPSeg --device cpu
python main.py --backend "YOLO-seg" --device cuda
```

처음 실행하거나 처음 추론할 때는 모델 파일을 내려받기 때문에 시간이 걸릴 수 있습니다.

## 간단한 사용 방법

1. `python main.py`로 프로그램을 실행합니다.
2. `Model` 영역에서 사용할 백엔드와 장치를 선택합니다.
3. 카메라를 사용할 경우 `Start Camera`를 누릅니다.
4. 동영상을 사용할 경우 `Browse`로 파일을 선택한 뒤 `Play Video`를 누릅니다.
5. `Target prompt`에 보존할 대상을 입력합니다. 예: `road`, `person`, `car`
6. `Mode`에서 비대상 영역 처리 방식을 선택합니다.
7. `Threshold`, `Inference scale`, `Skip frames`를 조절하며 품질과 속도의 균형을 확인합니다.

## 주요 설정

`Target prompt`는 모델이 찾을 대상 텍스트입니다. YOLO-World 백엔드는 쉼표로 여러 클래스를 입력할 수 있습니다.

`Mode`는 출력 효과입니다.

- `blur`: 비대상 영역 흐림 처리
- `remove`: 비대상 영역을 검은색으로 제거
- `dim`: 비대상 영역을 어둡게 표시
- `mask`: 모델 마스크를 컬러맵으로 미리보기

`Threshold`는 마스크를 대상/비대상으로 나누는 기준입니다. 값을 높이면 더 확실한 영역만 대상으로 남기고, 값을 낮추면 더 넓은 영역이 보존됩니다.

`Inference scale`은 추론 해상도 비율입니다.

- `0.25`: 원본 가로/세로의 25% 크기로 추론
- `0.50`: 원본 가로/세로의 50% 크기로 추론
- `1.00`: 원본 크기로 추론

낮은 값은 보통 빠르지만 마스크 품질이 떨어질 수 있고, 높은 값은 품질이 좋아질 수 있지만 지연 시간이 늘어납니다.

`Skip frames`는 모델 추론 사이에 건너뛸 프레임 수입니다. `0`은 모든 프레임에서 추론하고, `1`은 한 프레임씩 건너뛰며, `4`는 4프레임을 건너뛴 뒤 추론합니다.

## 지원 백엔드

현재 지원하는 백엔드 모델은 다음 두 가지입니다.

- `SegCLIP`: 텍스트 프롬프트를 기반으로 대상 영역 마스크를 생성하는 세그멘테이션 모델입니다. 현재 코드에서는 `CLIPSeg` 이름으로 표시되며 `CIDAS/clipseg-rd64-refined`를 사용합니다.
- `Yolo-seg`: 텍스트 프롬프트 기반 객체 검출 결과를 마스크처럼 사용해 대상 영역을 보존합니다. 현재 코드에서는 `YOLO-World small box-only` 이름으로 표시됩니다.

UI에는 다음 백엔드도 표시되지만, 필요한 런타임과 체크포인트가 아직 포함되어 있지 않아 실행은 차단됩니다.

- `YOLO-World small + SAM2 tiny tracking`
- `SAM 3`
- `Grounding DINO tiny + SAM2 tiny`

## 최적화 스윕

`Sweep Space`에서 프롬프트, 백엔드, 스케일, 임계값, 프레임 스킵 값을 입력한 뒤 `Run Sweep On Current Frame`을 누르면 현재 프레임 기준으로 조합별 결과를 확인할 수 있습니다.

입력값은 쉼표로 구분합니다.

```text
Target prompts: road
Scales: 0.25,0.50,0.75,1.00
Thresholds: 0.35,0.50,0.65
Skip frames: 0
```

## CamVid 벤치마크

앱에는 CamVid 도로 클래스 벤치마크 기능이 포함되어 있습니다. 현재 저장소에 이미 다음 파일이 있으면 바로 `Run CamVid Road Benchmark` 버튼을 사용할 수 있습니다.

```text
benchmark_data/camvid_road_0001TP.mp4
benchmark_gt/camvid_road_0001TP/
```

벤치마크 데이터를 새로 만들려면 CamVid 데이터셋을 준비한 뒤 아래 명령을 실행합니다.

```bash
git clone --depth 1 https://github.com/lih627/CamVid.git datasets/CamVid
python benchmark/prepare_camvid_benchmark.py --class-name road --frames 30 --width 480
python main.py
```

벤치마크 결과는 UI의 `Save result` 버튼으로 CSV 파일로 저장할 수 있습니다.

## 평가 지표

벤치마크에서는 다음 지표를 사용합니다.

- `non_target_leakage = FP / (FP + TN)`: 비대상 영역이 얼마나 많이 보존되었는지 나타냅니다. 낮을수록 좋습니다.
- `target_damage = FN / (TP + FN)`: 대상 영역이 얼마나 많이 손상되었는지 나타냅니다. 낮을수록 좋습니다.
- `FPS`: 초당 처리 프레임 수입니다. 높을수록 좋습니다.
- `latency_ms`: 프레임당 평균 처리 시간입니다. 낮을수록 좋습니다.
- `model_latency_ms`: 실제 모델 호출에 걸린 평균 시간입니다. 낮을수록 좋습니다.

각 픽셀 기준의 혼동 행렬 정의는 다음과 같습니다.

- `TP`: 대상 픽셀을 올바르게 보존
- `FP`: 비대상 픽셀을 잘못 보존
- `TN`: 비대상 픽셀을 올바르게 억제
- `FN`: 대상 픽셀을 잘못 억제

실제 튜닝에서는 `non_target_leakage`, `target_damage`, `latency_ms`는 낮추고 `FPS`는 높이는 방향을 목표로 합니다. 다만 비대상 누출을 줄이면 대상 손상이 늘 수 있고, 대상을 더 넓게 보존하면 비대상 누출이 늘 수 있습니다.

## 참고 사항

- GUI 애플리케이션이므로 원격 서버나 WSL 환경에서는 디스플레이 설정이 필요할 수 있습니다.
- `torch`, `transformers`, `ultralytics` 설치와 모델 다운로드에는 시간이 걸릴 수 있습니다.
- CUDA를 사용하려면 로컬 환경에 맞는 PyTorch CUDA 빌드가 설치되어 있어야 합니다.
- `Reset Hyperparameters` 버튼을 누르면 주요 설정이 기본값으로 돌아갑니다.
