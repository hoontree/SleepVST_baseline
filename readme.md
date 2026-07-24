# SleepVST Baseline

SleepVST 논문([arXiv:2404.03831](https://arxiv.org/abs/2404.03831)) 기반 수면단계 분류 구현체.

- **입력**: 심박(HW, heart) 파형 + 호흡(BW, breath) 파형
- **모델**: 1D ResNet 인코더 → Pre-LN Transformer → 선형 분류기
- **출력**: 4-class 수면단계 (`Wake` / `N1+N2` / `N3` / `REM`)
- **확장**: KVSS 비디오에서 추출한 **호흡 프록시 + 모션 피처**로 RandomForest 헤드를 transfer 학습 (센서 없이 영상만으로 수면단계 추정하는 것이 최종 목표)

> ℹ️ **명령별 진입 config가 두 갈래입니다.** `transfer_to_video`는 기본 config(`config/defaults.yaml`)로 돌고,
> `pretrain/finetune/test`는 `config/command2/`의 전용 config로 돕니다(`--config-name command2/<명령>`).
> 예전엔 command2 쪽 배선이 깨져 있었는데, 이 문서를 만들며 함께 수정했습니다. 자세한 실행법은
> [§5 실행 방법](#5-실행-방법)에, 수정 내역과 남은 정리거리는 [§7 알려진 이슈](#7-알려진-이슈-수정-내역--남은-정리)에 있습니다.

---

## 1. 전체 파이프라인 (큰 그림)

```
  [원천 데이터]                         [전처리 CLI]                          [학습/평가]
                                                                      
  EDF (SHHS/MESA/SNUH) ──► cli_preprocess ──► data/{shhs,mesa}/  ─┐
                            (HW/BW 신호 추출)                       │
                                                                   ├─► cli_train command=pretrain
                                                                   │      (SHHS+MESA로 SleepVST 사전학습)
                                                                   │            │
                                                                   │            ▼
  KVSS 비디오 ──► cli_extract_respiratory ──► data/resp_proxy_*  ─┤     cli_train command=finetune
              │  (호흡 프록시 v1/v2 추출)                          │      (KVSS 실측 신호로 미세조정)
              │                                                    │            │
              └► cli_motionfeatures ──► data/motionfeatures/  ─────┤            ▼
                 (모션 피처 30종 추출)                              └─► cli_train command=transfer_to_video
                                                                          (인코더 freeze → 피처 추출 →
                                                                           RandomForest 헤드 학습/평가)
                                                                                 │
                                                                                 ▼
                                                                          src/eval/* (프록시 검증,
                                                                          실패모드 분석, 예측 변환)
```

핵심 아이디어: **SleepVST 인코더를 실측 신호로 학습**해 두고, 그 인코더를 얼려(frozen) 비디오 유래
호흡 프록시에 통과시켜 얻은 임베딩(128차원) + 모션 피처(90차원)를 **RandomForest**로 분류한다.

---

## 2. 디렉토리 구조

```
SleepVST_baseline/
├── src/
│   ├── cli_train.py              # [진입점] pretrain / finetune / test / transfer_to_video
│   ├── cli_preprocess.py         # [진입점] EDF 전처리 (SNUH/KISS, 호흡 EDF)
│   ├── cli_motionfeatures.py     # [진입점] 비디오 → 모션 피처
│   ├── cli_extract_respiratory.py# [진입점] 비디오 → 호흡 프록시 신호
│   ├── cli_transfer.py           # [진입점] transfer_to_video 전용 shim
│   ├── cli.py                    # [진입점] 구버전 호환 라우터
│   │
│   ├── models/
│   │   ├── SleepVST.py           # SleepVST(HW+BW) / SleepVST_BW(BW only) 모델
│   │   └── RFClassifier.py       # SleepVSTVideoRF: RandomForest 헤드 (impute+RF pipeline)
│   │
│   ├── data/
│   │   ├── base_datamodule.py    # BaseDataset 추상클래스 (_setup/_load_samples/_discover_ids)
│   │   ├── datasets/
│   │   │   ├── KVSS.py           # KVSS 로더 (mode=transfer면 모션+호흡, 아니면 실측신호)
│   │   │   ├── SHHS.py / MESA.py # 사전학습용 공개 데이터셋 로더
│   │   └── preprocess/
│   │       ├── motion_extractor.py           # 모션 피처 추출 본체
│   │       ├── respiratory_extraction.py     # 호흡 프록시 v1 (optical-flow)
│   │       ├── respiratory_extraction_v2.py  # 호흡 프록시 v2 (robust PBM + soft-ROI LK)
│   │       ├── respiratory_pipeline.py       # 단일 프로세스 파이프라인
│   │       ├── respiratory_pipeline_mp.py    # 멀티프로세스 파이프라인 (extract CLI가 사용)
│   │       ├── preprocess_SNUH.py            # SNUH/KISS EDF 전처리
│   │       ├── preprocess_respiratory_edf.py # 호흡 EDF 전처리
│   │       ├── filters/ · motion_mag/        # 스티어러블 피라미드 / 위상기반 모션확대
│   │       └── motion_shape_tracker.py 등    # 보조 유틸
│   │
│   ├── train/
│   │   ├── loop.py               # prepare_pretrain / prepare_finetune / test / sliding_window_inference
│   │   ├── trainer.py            # Trainer: epoch 루프·체크포인트·early stopping
│   │   └── transfer.py           # transfer_to_video 본체 (피처추출 + RF 학습/평가 + 혼동행렬)
│   │
│   ├── eval/
│   │   ├── metrics.py                     # MetricsTracker / AverageMeter
│   │   ├── full_oracle_predict.py         # [hydra] 풀-oracle 예측 재생성
│   │   ├── proxy_failure_analysis.py      # [argparse] G1~G4 실패모드 분석
│   │   ├── respiratory_proxy_validation.py# [argparse] 프록시 vs EDF 기준 검증
│   │   ├── make_vinuss_predictions.py     # [argparse] ViNUSS 결과 → 표준 스키마
│   │   └── convert_seq_predictions.py     # [argparse] 5-class → 4-class 매핑
│   │
│   └── utils/                    # logger, utils(setup_device 등)
│
├── config/                       # Hydra 설정 (§4 참고)
│   ├── defaults.yaml             # ★ 모든 진입점의 기본 config (transfer 지향)
│   ├── data/                     # kvss, kvss_bwproxy_v2, shhs, mesa
│   ├── model/                    # SleepVST, SleepVST_BW
│   ├── mode/                     # transfer / pretrain / finetune (일부 필드는 미사용, §7-11)
│   ├── preprocess/               # respiratory(_v2), motionfeatures, snuh, respiratory_edf
│   ├── train/                    # defaults (lr/epochs 등)
│   ├── log/                      # log_transfer
│   └── command2/                 # pretrain/finetune/test 전용 config (--config-name 으로 사용)
│
├── models/                       # (gitignored) 구버전 RF 체크포인트
├── checkpoint/                   # (gitignored) pretrain/finetune/randomforest 체크포인트
├── data/                         # (gitignored) 전처리 산출물
├── scripts/                      # compare_proxy_v1_v2.py, run_v2_extraction.{py,sh}
├── notebooks/ · archive/         # 분석/탐색 노트북 (archive는 gitignored)
└── readme.md
```

---

## 3. 모델 구조 요약

| 구성요소 | 설명 |
|---|---|
| `WaveformEncoder` | 1D ResNet. `(B, N에폭, L샘플)` → `reshape(B,1,N·L)` → conv/resblock → `AdaptiveAvgPool1d(seq_len)` → `(B, seq_len, out)`. **길이 L에 유연** (AdaptiveAvgPool 덕분에 L=74든 150이든 안 터짐) |
| `SleepVST` | HW 인코더 + BW 인코더 → 각 `d_model/2` 투영 후 concat → PreLN Transformer(6층) → `Linear(d_model, 4)` |
| `SleepVST_BW` | BW 단일 인코더(out=128) → `d_model` 투영 → 동일 Transformer/분류기 |
| `SleepVSTVideoRF` | `SimpleImputer(median)` + `RandomForestClassifier(n=300, class_weight=balanced_subsample)`. 입력 = 신호임베딩 128 + 모션 90 = **218차원**. `joblib`로 저장/로드 |

`transfer_to_video`는 `model.classifier = nn.Identity()`로 바꿔 인코더를 **피처 추출기**로 쓴다.

---

## 4. 설정(Config) 체계

Hydra 기반. **모든 학습/평가 진입점은 `config/defaults.yaml`을 기본으로 로드**한다
(`@hydra.main(config_name="defaults")`). `defaults.yaml`은 다음을 조합한다:

```
data: kvss_bwproxy_v2 · model: SleepVST · mode: transfer · log: log_transfer
preprocess: respiratory · train: defaults · (+ _self_)
command: transfer_to_video   # 기본 명령
```

CLI에서 Hydra 문법으로 덮어쓸 수 있다:

```bash
python -m src.cli_train command=pretrain           # 명령 변경
python -m src.cli_train data=kvss_bwproxy_v2        # data 그룹 교체
python -m src.cli_train system.gpu_ids='0,1'        # GPU 지정
python -m src.cli_train --config-name preprocess/respiratory_v2   # 루트 config 자체 교체
```

> **`config/command2/`는 pretrain/finetune/test 전용 config 묶음이다.** 진입점이
> `config_name="defaults"`로 고정돼 있으므로 `--config-name command2/<명령>`으로 로드한다.
> (예전엔 이 config들이 깨져 있었으나 [§7](#7-알려진-이슈-수정-내역--남은-정리)에서 수정됨.)

---

## 5. 실행 방법

> 아래 명령들은 **Hydra config 조합·임포트를 실제로 검증**했다(데이터/체크포인트 존재 여부와는 별개).

### 5.1 설치

```bash
# uv 사용 시
uv sync
# 또는 pip
pip install torch torchvision hydra-core omegaconf numpy scipy scikit-learn tqdm wandb joblib opencv-python
```

### 5.2 Transfer to video (RandomForest 헤드) — ✅ 실제 실행 검증됨

이 저장소의 **주 파이프라인**. 인코더를 얼려 피처를 뽑고 RF 헤드를 학습·평가한다.
기본 config가 이제 `data: kvss_bwproxy_v2`(mode=transfer, SleepVST dual, v2 프록시)라 bare 명령이 바로 동작한다.

```bash
python -m src.cli_train                       # command 기본값 transfer_to_video, data=kvss_bwproxy_v2
python -m src.cli_train command=transfer_to_video   # 동일

# 전용 shim (이제 setup_logging/GPU 설정 포함 — 기능상 cli_train 경로와 동일)
python -m src.cli_transfer
```

> ⚠️ **`data=kvss`로는 transfer가 동작하지 않는다.** kvss.yaml은 `mode: finetune`/`SleepVST_BW`라
> transfer가 빈 regular 샘플을 로드한다([§7-10](#7-알려진-이슈-수정-내역--남은-정리)). 그래서 기본값을 `kvss_bwproxy_v2`로 바꿨다.

동작 모드는 `transfer.classifier.mode`로 제어: `fit`(학습·저장) / `fit_test`(학습·저장·테스트, 기본) / `test`(저장된 RF 로드 후 테스트).
저장·로드 경로가 이제 일치하므로 `mode=test` 단독 실행도 방금 저장한 RF(`_{data_source}` 접미사 포함)를 찾는다.

```bash
# 저장한 RF를 다시 로드해 테스트만
python -m src.cli_train transfer.classifier.mode=test system.gpu_ids=0
```

> ✅ **실제 실행 확인 (2026-07-24)**: `transfer.classifier.mode=test`를 GPU에서 완주 —
> 체크포인트 로드 → KVSS test 70개 로드 → 피처 추출 `(57838, 218)` → `SleepVST_rf_model_video.pkl`
> 로드(수정된 경로) → 예측·지표 산출 → CSV/혼동행렬 PNG 저장까지 exit 0. (지표 자체는 이 RF가
> v2 프록시와 다른 데이터로 학습돼 낮게 나오지만, 그건 모델/데이터 정합성 문제로 파이프라인 배선과 무관.)

### 5.3 사전학습 (SHHS + MESA) — ✅ dispatch 검증됨

```bash
# 권장: 전용 config (checkpoint/pretrain/, 로그명 sleepvst_pretrain, epochs 50)
python -m src.cli_train --config-name command2/pretrain

# 기본 config로도 동작 (checkpoint/sleepvst_train.pth, epochs 30)
python -m src.cli_train command=pretrain
```
SHHS/MESA config는 `config/data/*.yaml`에서 코드가 직접 로드한다.
> ✅ 실행 확인: dispatch → `prepare_pretrain` → SHHS 데이터셋 빌드까지 도달. 완주하려면 SHHS 스플릿
> 정의 파일(`data/shhs/<split>/A-<split>_set.txt`)이 있어야 한다(이 환경엔 npy만 있고 스플릿 txt는 없음).

### 5.4 미세조정 (KVSS) — ✅ dispatch 검증됨

```bash
python -m src.cli_train --config-name command2/finetune
```
- 사전학습 가중치 `checkpoint/pretrain/pretrained_sleepvst_bw.pth`를 로드해 KVSS 실측 신호로 미세조정.
- 체크포인트 저장: `checkpoint/finetune/sleepvst_finetune_kvss_finetuned.pth`.
- 기본 config의 `command=finetune`은 전용 config를 쓰라는 안내 후 종료된다.
> ✅ 실행 확인: dispatch → `prepare_finetune` → KVSS train(333)/valid(69)/test(70) 빌드까지 도달.
> **완주하려면 KVSS `{id}_label.npy`가 필요**하다(이 환경엔 없음 — 라벨이 annotation JSON에만 있어
> transfer 경로만 소비. finetune/test의 regular-sample 경로는 label.npy를 요구, [§7-13](#7-알려진-이슈-수정-내역--남은-정리)).

### 5.5 평가 (test) — ✅ dispatch 검증됨

```bash
# ⚠️ command2/test.yaml의 기본 test.checkpoint 파일명(sleepvst_finetune_kvss_finetuned.pth)은
#    이 환경에 없다. 존재하는 체크포인트로 오버라이드:
python -m src.cli_train --config-name command2/test \
    test.checkpoint=checkpoint/finetune/kvss_finetuned_bw.pth
```
- `test.checkpoint`의 모델을 로드해 `test.datasets`(기본 `kvss`)에 대해 sliding-window 추론.
- 기본 config의 `command=test` 역시 전용 config 안내 후 종료된다.
> ✅ 실행 확인: dispatch → `test()` → 체크포인트 로드 → KVSS test 빌드까지 도달. 완주 조건은 §5.4와 동일(label.npy).

### 5.6 전처리 — ✅ (config 주의)

```bash
# 비디오 → 모션 피처
python -m src.cli_motionfeatures                # config_name=preprocess/motionfeatures 고정

# 비디오 → 호흡 프록시 (v1 기본)
python -m src.cli_extract_respiratory           # config_name=preprocess/respiratory 고정
# 호흡 프록시 v2
python -m src.cli_extract_respiratory --config-name preprocess/respiratory_v2
# (scripts/run_v2_extraction.sh 가 위 v2 명령을 래핑)

# EDF 전처리 — ⚠ 기본 config엔 dataset 지정이 없어 그대로는 exit(1).
#   해당 preprocess config를 함께 지정해야 한다 (예시, 사용 전 config 확인 권장):
python -m src.cli_preprocess command=preprocess preprocess=snuh
python -m src.cli_preprocess command=preprocess_respiratory_edf preprocess=respiratory_edf
```

### 5.7 분석/평가 스크립트 — ✅ (argparse, 독립 실행)

```bash
# v1 vs v2 호흡 프록시 비교
python scripts/compare_proxy_v1_v2.py --records 10 --epochs 20

# 프록시 vs EDF 기준 검증
python -m src.eval.respiratory_proxy_validation \
    --proxy-root data/resp_proxy_video_epochs \
    --ref-dir    data/resp_proxy_video \
    --out-csv    results/proxy_validation_v1.csv

# G1~G4 실패모드 분석 (--vinuss 필수)
python -m src.eval.proxy_failure_analysis \
    --vinuss results/predictions/vinuss_predictions_YYYY-MM-DD.csv \
    --proxy  results/predictions/sleepvst_dual_predictions_YYYY-MM-DD.csv

# 예측 스키마 변환
python -m src.eval.make_vinuss_predictions --src <movinet_csv> --out <out_csv>
python -m src.eval.convert_seq_predictions --files <csv...> --reference <ref_csv>
```

---

## 6. 데이터셋

| 이름 | 용도 | 비고 |
|---|---|---|
| **SHHS** | 사전학습 | 공개. `data/shhs/` |
| **MESA** | 사전학습 | 공개. `data/mesa/` |
| **KVSS** | 미세조정·transfer | 비공개. `data/kvss/` + 비디오 유래 `data/resp_proxy_video_epochs(_v2)/`, `data/motionfeatures/` |

KVSS 스플릿은 `data/kvss/A-{train,valid,test}_set.txt`로 정의되며, `config/data/kvss*.yaml`의
`exceptions:` 목록에 있는 레코드는 로딩에서 제외된다.

---

## 7. 알려진 이슈 (수정 내역 / 남은 정리)

코드·config를 대조하고 **실제로 실행**하며 발견한 문제들. **#1~#10은 이 브랜치에서 수정 완료**, #11~ 는 남은 정리거리 / 환경 의존성.

### ✅ 수정 완료

| # | 위치 | 증상 | 수정 |
|---|---|---|---|
| **1** | `config/command2/finetune.yaml` | `--config-name command2/finetune` 실행 시 모든 키가 `command2:` 아래로 중첩돼 `cfg.command`조차 못 읽음 → "Unknown training command" | 파일 맨 위에 **`# @package _global_` 추가** |
| **2** | `config/command2/{pretrain,test}.yaml` | `cli_train.run()`이 dispatch 전에 `cfg.system.device`/`gpu_ids`([cli_train.py:42-43])를 읽는데 두 config에 `system:` 블록이 없어 크래시 | **`system:` 블록 추가** (+ `_self_` 명시로 경고 제거) |
| **3** | `command=test` (기본 config) | `test()`가 `cfg.test.checkpoint`([loop.py:409]) 등을 읽는데 기본 config엔 `test:`가 없어 크래시 | `run()`에 **가드 추가** — 전용 config(`command2/test`) 안내 후 정상 종료 |
| **4** | `src/train/transfer.py` 저장/로드 경로 | 저장은 `..._rf_model_{data_source}.pkl`(접미사 O), 로드는 접미사 X → `mode=test`가 방금 저장한 모델을 못 찾음 | **로드 경로에도 `_{cfg.data.data_source}` 추가**해 저장·로드 일치 (실행으로 확인) |
| **5** | `command=finetune` (기본 config) | `prepare_finetune`이 `cfg.train.pretrained_checkpoint`([loop.py:379])를 읽는데 없어 크래시 | `run()`에 **가드 추가** — 전용 config(`command2/finetune`) 안내 후 정상 종료 |
| **6** | `src/cli_transfer.py` | `cli_train` 경로와 달리 파일 로깅/GPU 선택이 적용 안 됨 | shim에 **`setup_logging()`·`setup_device()` 추가** |
| **7** | `config/command2/{pretrain,test}.yaml` | Hydra "Defaults list is missing `_self_`" 경고 | defaults에 **`_self_` 명시** |
| **8** | `src/models/registry.py`, `src/data/registry.py` | 어디서도 import 안 되는 죽은 코드 | **삭제** |
| **9** | `src/utils/utils.py` `setup_device` | `system.gpu_ids=0`(int)로 오버라이드하면 `os.environ[...]=int` → `TypeError`. **실제 실행에서 발견** | `str(gpu_ids)`로 강제 변환 |
| **10** | `config/defaults.yaml` `data: kvss` | transfer 기본 실행이 빈 샘플 로드로 실패. kvss.yaml은 `mode=finetune`/`SleepVST_BW`라 transfer가 regular 샘플을 로드(라벨 없음→0개). **실제 실행에서 발견** | 기본 data를 **`kvss_bwproxy_v2`**(mode=transfer, SleepVST, v2 프록시)로 변경 — model·command과 정합 |

> #3·#5는 "기본 config로도 finetune/test가 돌게" 만드는 대신 의도된 전용 config(`command2/`)로
> 안내하는 방식으로 처리했다. **학습/평가 정식 실행은 `--config-name command2/<명령>`.**

### 🔧 남은 정리거리 / 환경 의존성

| # | 위치 | 내용 |
|---|---|---|
| **11** | `config/mode/pretrain.yaml`, `mode/finetune.yaml`, `mode/extract_respiratory.yaml` | `mode.lr`/`max_epochs`/`freeze`는 코드에서 안 읽힘, `mode/extract_respiratory.yaml`도 미사용. 크래시는 없으나 헷갈리는 죽은 config |
| **12** | `config/model/*.yaml` `input_length: 74` | 실제로는 레이어 크기를 결정하지 않는 값(`WaveformEncoder`는 AdaptiveAvgPool). no-op이라 문서로만 남김 |
| **13** | KVSS regular-sample 경로 (`load_regular_samples`) | finetune/test는 `{id}_label.npy`를 요구하는데(없으면 레코드 skip) 이 환경엔 label.npy가 0개(라벨이 annotation JSON에만 존재). 그래서 finetune/test는 **dispatch까지만 검증**됨. 완주하려면 label.npy 생성 또는 로더가 JSON/CSV 라벨을 쓰도록 수정 필요 |
| **14** | `config/command2/test.yaml` `test.checkpoint` | 기본값 `sleepvst_finetune_kvss_finetuned.pth`가 이 환경에 없음(있는 건 `kvss_finetuned{,_bw}.pth`). 실행 시 오버라이드하거나 실제 finetune 산출물 경로로 갱신 필요 |

### 실제 실행 검증 (2026-07-24, GPU)

| 명령 | 어디까지 확인 | 결과 |
|---|---|---|
| `cli_train transfer.classifier.mode=test` (data=kvss_bwproxy_v2) | **완주** | exit 0 — 피처 `(57838,218)` 추출 → RF 로드(수정경로) → 예측·지표·CSV·PNG |
| `--config-name command2/pretrain` | dispatch → SHHS 빌드 | ✅ 배선 정상 (SHHS 스플릿 txt 없어 그 지점서 멈춤) |
| `--config-name command2/finetune` | dispatch → KVSS train/val/test 빌드 | ✅ 배선 정상 (label.npy 없어 그 지점서 멈춤, #13) |
| `--config-name command2/test` | dispatch → 체크포인트 로드 → 데이터 빌드 | ✅ 배선 정상 (label.npy 없어 그 지점서 멈춤, #13) |
| `command=finetune`/`command=test` (기본 config) | 가드 | ✅ 안내 메시지 후 정상 종료 |

> 즉 **모든 명령의 config·dispatch 배선은 실행으로 확인**됐다. finetune/test/pretrain의 *완주*는
> 이 환경에 없는 데이터(KVSS label.npy, SHHS 스플릿 txt)에만 걸려 있고, 배선 문제는 아니다.

---

## 8. 인용

```bibtex
@article{sleepvst2024,
  title={SleepVST: Sleep Stage Classification using Video and Sensor Transformer},
  url={https://arxiv.org/abs/2404.03831},
  year={2024}
}
```
