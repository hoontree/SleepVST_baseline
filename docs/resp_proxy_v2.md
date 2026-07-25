# 호흡 프록시 v2 — 산출물 · 불러오기 가이드

KVSS 비디오에서 추출한 **호흡 프록시 v2**(`data/resp_proxy_video_epochs_v2/`)의 구조와
불러오는 방법을 정리한다. 추출 알고리즘 자체는
[`src/data/preprocess/respiratory_extraction_v2.py`](../src/data/preprocess/respiratory_extraction_v2.py)에 있다.

> **TL;DR** — 학습에 쓸 거면 새 로더를 짤 필요 없다.
> 기존 `KVSS` 데이터셋에 `config/data/kvss_bwproxy_v2.yaml`을 물리면 그대로 나온다.
> ```bash
> python -m src.cli_transfer data=kvss_bwproxy_v2
> ```

---

## 1. v2가 v1과 다른 점

v1(`resp_extraction_r`)은 PBM 에너지 맵의 **최댓값 픽셀 한 점**만 LK 광학흐름으로 추적한다.
그 한 점이 잘못 잡히면 그 에포크는 통째로 망가진다.

v2는 같은 PBM 에너지 맵을 **ROI 선택에만** 쓰고, 신호는 다음 경로로 뽑는다.

```
video (T,H,W,3)
   │
   ├─► PBM(motionMag) ──► energy map (H,W)      # ROI 선택 전용
   │                          │
   │                          └─► top-K(=10) 최고 에너지 시드 픽셀
   │                                    │
   └────────────────────────────────────┴─► 시드마다 LK 광학흐름 추적 (y 변위)
                                                  │
                                    에너지 가중 평균 ──► raw (T,)
                                                  │
                        zero-phase bandpass 0.1–0.5 Hz (sosfiltfilt, 실제 fps)
                                                  │
                                              z-score ──► signal (T,)
```

v1 대비 개선점 4가지:

| # | 항목 | v1 | v2 |
|---|------|----|----|
| 1 | ROI | 단일 최대점 (취약) | top-K=10 소프트 ROI, 에너지 가중 평균 |
| 2 | 샘플링 | `make_half()`로 프레임 절반 → 실제 ≈2.5 Hz | 전 프레임 유지 → 네이티브 5 Hz |
| 3 | 필터 | 인과 `sosfilt`, `fs=5` 하드코딩(실제와 불일치) | zero-phase `sosfiltfilt`, 실제 `fps` 사용 |
| 4 | 품질관리 | 없음 | 에포크별 SNR/호흡수 사이드카 저장 |

**정확도** — v1/v2 비교 스크립트([`scripts/compare_proxy_v1_v2.py`](../scripts/compare_proxy_v1_v2.py))로
EDF 오라클 대비 측정한 에포크별 |r| 중앙값:

| 방식 | median \|r\| |
|------|--------------|
| medfilt legacy | 0.175 |
| v2 초기안 (증폭영상 휘도) | 0.249 |
| v1 (단일점 광학흐름) | 0.371 |
| **v2 (소프트 ROI 광학흐름)** | **0.549** |

> ⚠️ 위 수치는 비교 스크립트가 뽑은 **부분 레코드 subset** 기준이다.
> 전체 데이터 오라클 검증(`results/proxy_validation_v2.csv`)은 아직 돌리지 않았다 — [§6](#6-남은-일) 참조.

---

## 2. 디스크 산출물

```
data/resp_proxy_video_epochs_v2/
└── {record_id}/                          # 예: A2019-EM-01-0001
    ├── epoch_1/
    │   ├── epoch_1_movement.npy          # 호흡 프록시 신호  (samples,) float
    │   └── epoch_1_quality.npz           # 에포크 품질 지표
    ├── epoch_2/
    │   └── ...
    └── epoch_819/
```

에포크 번호는 **1-based**다. 디렉토리 이름순 정렬은 `epoch_10 < epoch_2`가 되므로
반드시 숫자로 파싱해 정렬해야 한다 (기존 로더들은 이미 그렇게 한다).

### `epoch_N_movement.npy`

30초 에포크 하나의 z-score 정규화된 호흡 프록시. 길이는 **레코드마다 고정**이고
`round(30 × video_fps)`로 정해진다.

| 길이 | 레코드 수 | 비고 |
|------|-----------|------|
| 150 | 409 | 5.00 fps — BW 오라클과 동일 |
| 149 | 48 | ≈4.97 fps |
| 148 | 42 | ≈4.93 fps |

레코드의 **마지막 에포크 1개만** 짧을 수 있다(영상이 에포크 중간에 끝나는 경우, 예: 16/20/43 샘플).
374개 레코드가 짧은 꼬리 1개를 갖고, 125개는 없다. **꼬리가 2개 이상 짧은 레코드는 0개**다.
→ 기존 로더의 "마지막 1개 드롭 + 나머지 리샘플" 로직과 정확히 맞는다.

### `epoch_N_quality.npz`

| 키 | 타입 | 의미 |
|----|------|------|
| `breathing_rate` | float | 호흡대역 우세 주파수 (breaths/min) |
| `snr` | float | 피크 ±0.03 Hz 전력 / 0.1–0.5 Hz 대역 전력 (1.0 = 완전 단일주파수) |
| `peak_freq` | float | 우세 주파수 (Hz) |
| `ok` | bool | `snr ≥ 0.5` **and** `6 ≤ BR ≤ 30` |
| `max_point` | (2,) int | PBM 에너지 최댓값 픽셀 좌표 (row, col) |

---

## 3. 불러오기 — 학습/전이학습용 (기존 로더 사용)

새 코드 필요 없다. [`src/data/datasets/KVSS.py`](../src/data/datasets/KVSS.py)가
`respiratory_signal_dir`만 v2로 바꾸면 그대로 처리하고,
그 설정이 [`config/data/kvss_bwproxy_v2.yaml`](../config/data/kvss_bwproxy_v2.yaml)에 이미 들어있다.

```bash
python -m src.cli_transfer data=kvss_bwproxy_v2
```

config에서 v2 관련 핵심 필드:

```yaml
respiratory_signal_dir: .../data/resp_proxy_video_epochs_v2
data_source: video        # 'video' 여야 epoch_*/*.npy 경로를 탄다
bw_patch_samples: 150     # v2는 네이티브 5 Hz (v1은 74였음)
model: SleepVST           # HW+BW 동시 로드 (BW만이면 SleepVST_BW)
mode: transfer            # 'finetune'이면 비디오 로딩 경로를 아예 안 탄다
```

`bw_patch_samples: 150`이 149/148 레코드를 알아서 처리한다 —
길이가 다른 에포크는 `scipy.signal.resample_poly`로 150에 맞추고, 짧은 꼬리 1개는 버린다.

파이썬에서 직접 쓸 때:

```python
from omegaconf import OmegaConf
from src.data.datasets.KVSS import KVSS

cfg = OmegaConf.load("config/data/kvss_bwproxy_v2.yaml")
kw = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
      if k not in ("_target_", "name", "split")}

ds = KVSS(split="train", **kw)
s = ds.samples[0]
# s["x_bw"]   (T, 150)  v2 호흡 프록시
# s["x_hw"]   (T, 300)  실측 심박 오라클 (data_source와 무관하게 signal_dir에서 읽음)
# s["motion"] (T, 90)   모션 피처 30종 × (past/current/future)
# s["label"]  (T,)      0=Wake 1=N1+N2 2=N3 3=REM
# 네 배열 모두 T = min(...)으로 정렬되어 길이가 같다
```

실측 확인된 출력 (3개 레코드 스모크 테스트):

```
A2019-EM-01-0001: x_bw(818,150) x_hw(818,300) motion(818,90) label(818,)
A2019-EM-01-0175: x_bw(822,150) x_hw(822,300) motion(822,90) label(822,)   # 148-modal → 150 리샘플됨
A2021-EM-01-0079: x_bw(897,150) x_hw(897,300) motion(897,90) label(897,)
```

> ℹ️ transfer 경로는 라벨을 `video_dir/{id}/{id}_annotation.json`에서 읽는다.
> `{id}_label.npy`에 의존하는 건 `mode: finetune`(regular sample) 쪽이고, 그 파일은 이 환경에 없다.

---

## 4. 불러오기 — 품질 사이드카

`quality.npz`는 기존에 읽는 코드가 없어서 리더를 하나 추가했다
([`respiratory_extraction_v2.load_quality`](../src/data/preprocess/respiratory_extraction_v2.py)).

```python
from src.data.preprocess.respiratory_extraction_v2 import load_quality

q = load_quality("data/resp_proxy_video_epochs_v2/A2019-EM-01-0001")
# q["epoch"]          (N,)   1-based 에포크 번호, 오름차순
# q["breathing_rate"] (N,)   breaths/min
# q["snr"]            (N,)
# q["peak_freq"]      (N,)   Hz
# q["ok"]             (N,)   bool
# q["max_point"]      (N, 2) (row, col)

good = q["ok"]                      # 신뢰 가능한 에포크 마스크
print(f"{good.sum()}/{len(good)} ({100*good.mean():.1f}%)")
```

에포크 인덱스가 1-based이므로, 0-based 배열(`x_bw` 등)과 맞출 땐 `q["epoch"] - 1`을 쓴다.
꼬리 에포크가 드롭됐을 수 있으니 길이를 그대로 신뢰하지 말고 인덱스로 정렬하는 편이 안전하다.

---

## 5. 검증 결과 (2026-07-25 확인)

전체 추출은 **2026-06-29 ~ 07-03 (약 4.5일, 60 workers)** 에 완료됐다.

| 항목 | 값 |
|------|-----|
| 처리 레코드 | **499 / 499** (소스의 `A*` 전체, 누락 0) |
| 생성 에포크 | **415,076** / 기대 415,192 (99.97%) |
| 출력 크기 | 4.8 GB |

**결손 분석** — 80개 레코드가 기대치보다 적지만 **전부 꼬리(tail)에서만** 1–2 에포크 부족하고,
중간 결손(interior gap)은 **0개**다. 크래시가 아니라 영상 길이가 annotation의 epoch 수보다
짧아 자연 종료된 케이스다 (총 116 에포크, 0.03%).

**파일 무결성** — 60개 레코드 × 20 에포크 = 1,200개 무작위 샘플:

| 검사 | 결과 |
|------|------|
| 읽기 실패 | 0 |
| NaN/Inf | 0 |
| 상수 신호(std≈0) | 0 |
| `quality.npz` 누락 | 0 |

**품질 분포** (동일 샘플):

| 지표 | 값 |
|------|-----|
| SNR 중앙값 | 0.452 |
| 호흡수 중앙값 | 14.0 /min (p10 10.0, p90 18.0) |
| `ok` 통과율 | 36.5% |

> ⚠️ `ok` 통과율 36.5%는 임계값 `snr ≥ 0.5`가 SNR 중앙값(0.452) 바로 위에 걸려 있어서 나온 값이다.
> 신호가 실패한 게 아니라 QC 기준이 빡빡한 쪽이다. `ok`로 필터링할 계획이면 임계값을 먼저 재검토할 것.
> 레코드별 편차도 크다 — 예: `A2019-EM-01-0001` 54.1% vs `A2021-EM-01-0079` 14.6%.

---

## 6. 남은 일

- **전체 오라클 검증 미실행.** `results/`에 v1 결과(`proxy_validation_v1.csv`)는 있지만 v2는 없다:
  ```bash
  python -m src.eval.respiratory_proxy_validation \
      --proxy-root data/resp_proxy_video_epochs_v2 \
      --ref-dir    data/resp_proxy_video \
      --records 80 \
      --out-csv    results/proxy_validation_v2.csv
  ```
- `ok` 임계값 재보정 (§5 참조).

---

## 7. 재추출

```bash
# 전체 (A* 499개, 60 workers) — config/preprocess/respiratory_v2.yaml 사용
python scripts/run_v2_extraction.py

# 일부 레코드만
python scripts/run_v2_extraction.py --records A2019-EM-01-0022,A2021-EM-01-0020

# 워커 수 조정 / 기존 결과 무시하고 재처리 / 출력 경로 변경
python scripts/run_v2_extraction.py --workers 40 --no-skip --output-dir /path/to/out
```

`skip_existing: true`가 기본이라 중단 후 재실행하면 이어서 처리한다.
`A*` 필터는 [`respiratory_pipeline_mp.py`](../src/data/preprocess/respiratory_pipeline_mp.py)의
`input_path.glob('A*/*.mp4')`에서 걸린다 — 소스 1,000개 중 A로 시작하는 499개만 대상이고
B\*/D\* 등 501개는 자동 제외된다.

> ℹ️ `scripts/run_v2_extraction.py`는 Hydra를 거치지 않고 `OmegaConf.load()`로 config를 직접 읽는다.
> `--config-name preprocess/respiratory_v2`처럼 경로 구분자가 든 config 이름을 Hydra에 넘기면
> 빈 struct config가 만들어져 `Key 'log' is not in struct`로 죽기 때문이다.

---

## 관련 파일

| 경로 | 역할 |
|------|------|
| [`src/data/preprocess/respiratory_extraction_v2.py`](../src/data/preprocess/respiratory_extraction_v2.py) | v2 추출 알고리즘 + `load_quality()` |
| [`src/data/preprocess/respiratory_pipeline_mp.py`](../src/data/preprocess/respiratory_pipeline_mp.py) | 멀티프로세싱 오케스트레이션 |
| [`src/data/datasets/KVSS.py`](../src/data/datasets/KVSS.py) | 학습용 로더 (v1/v2 공용) |
| [`config/preprocess/respiratory_v2.yaml`](../config/preprocess/respiratory_v2.yaml) | 추출 config |
| [`config/data/kvss_bwproxy_v2.yaml`](../config/data/kvss_bwproxy_v2.yaml) | 학습 데이터 config |
| [`scripts/run_v2_extraction.py`](../scripts/run_v2_extraction.py) | 추출 실행 스크립트 |
| [`scripts/compare_proxy_v1_v2.py`](../scripts/compare_proxy_v1_v2.py) | v1/v2 비교 |
| [`src/eval/respiratory_proxy_validation.py`](../src/eval/respiratory_proxy_validation.py) | EDF 오라클 대비 검증 |
