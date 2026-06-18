# Results Index

Proxy failure-mode analysis(G1–G4)용 결과 파일 정리. 어느 파일이 어느 실험 역할인지 확정 매핑.

## 모델/세팅 ↔ 파일 (확정)

KVSS 정식 분류기 = SleepVST `forward_features`(128d) **+ 비디오 motion feature(90d) → RandomForest**
(`checkpoint/randomforest/SleepVST_rf_model.pkl`, n_features_in_=218). 라벨 0=Wake,1=N1/N2,2=N3,3=REM.

| 역할 | cardiac | respiratory | per-epoch 파일 | acc | kappa | f1_macro |
|------|---------|-------------|----------------|-----|-------|----------|
| **full-oracle** | real ECG | real belt | `predictions/full_oracle_predictions_2025-10-29.csv` | 0.764 | 0.607 | 0.698 |
| **partial-oracle** | real ECG | **proxy resp** | `predictions/partial_oracle_predictions_2025-11-09.csv` | 0.612 | 0.221 | 0.358 |
| **ViNUSS** | end-to-end video | | `predictions/vinuss_predictions_2026-06-18.csv` | 0.804 | 0.683 | — |

- full→partial 은 호흡벨트를 video proxy 로 교체한 차이(같은 dual 모델/RF). N3·REM acc 가 크게
  떨어지는 게 proxy 호흡의 열화를 보여줌 (full REM 0.81 → partial REM 0.03).
- ViNUSS = `movinet_mamba_test/stage2_test_results/test_results_all_4class.csv` 를 표준 스키마로
  변환한 것 (per-subject acc 가 `metrics/vinuss_subject_stats.csv` 와 1e-16 일치).
- 테스트 셋 KVSS 70 subjects / 57,901 epochs, subject 당 recording 1개.

predictions 스키마: `subject_id, recording_idx, epoch_idx, ground_truth_label, ground_truth_name,
prediction_label, prediction_name`. (recording_idx 스킴은 파일마다 달라, 분석 merge 는
`(subject_id, epoch_idx)` 키로 함.)

## metrics (per-subject)
- `metrics/full_oracle_subject_metrics_2025-10-29.csv`
- `metrics/partial_oracle_subject_metrics_2025-11-09.csv`
- `metrics/vinuss_subject_stats.csv`  (= ViNUSS subject stats)

## G1–G4 분석 — ViNUSS vs partial-oracle

스크립트: [src/eval/proxy_failure_analysis.py](../src/eval/proxy_failure_analysis.py)

```bash
python -m src.eval.proxy_failure_analysis \
    --vinuss results/predictions/vinuss_predictions_2026-06-18.csv \
    --proxy  results/predictions/partial_oracle_predictions_2025-11-09.csv
# 결과 → results/analysis/proxy_failure/
# --proxy 를 full_oracle 로 바꾸면 ViNUSS vs full-oracle 도 동일하게 가능
```

**결과 (`analysis/proxy_failure/`, matched 57,901 epochs):**

| group | 의미 | count | pct |
|-------|------|------:|----:|
| G1 | ViNUSS✓ / Proxy✗ | 15,815 | 27.3% |
| G2 | ViNUSS✗ / Proxy✓ |  4,676 |  8.1% |
| G3 | both ✓ | 30,747 | 53.1% |
| G4 | both ✗ |  6,663 | 11.5% |

- McNemar χ² ≈ 6054 (p≪0.05).
- stage별: ViNUSS 가 **REM(G1 5692 : G2 12)·N3(G1 3107 : G2 34)** 에서 압도 → partial-oracle 의
  proxy 호흡이 N3/REM 에서 무너지는 구간을 ViNUSS 가 복구. partial-oracle 은 N1/N2(G2 4035)에서만 우세.
- 산출물: `group_summary.csv`, `group_by_stage.csv`, `group_by_subject.csv`(vinuss_gain↓),
  `G1_proxy_confusion.csv`, `G2_vinuss_confusion.csv`, `epoch_groups.csv`, `report.md`.

## 코드 (재현)
- [src/eval/make_vinuss_predictions.py](../src/eval/make_vinuss_predictions.py) — movinet stage2 →
  ViNUSS 표준 스키마 변환 (`vinuss_predictions_*.csv` 생성).
- [src/eval/proxy_failure_analysis.py](../src/eval/proxy_failure_analysis.py) — G1–G4 분석.
- `output/diagnostics/epochs_analysis_*.csv` — 전처리 length 진단(결과 아님).
- 참고: partial-oracle 을 다른 proxy(예: `kiss_respiratory`)로 재생성하려면 RF 파이프라인
  (transfer/motion 경로, `respiratory_signal_dir`/`data_source`)로 hw=real·bw=proxy 특징 추출 후
  `checkpoint/randomforest/SleepVST_rf_model.pkl` 적용.

## 남은 공백 (TODO)
- ViNUSS vs full-oracle G1–G4 (상한선 대비) — `--proxy full_oracle...` 로 바로 가능.
- partial-oracle proxy 호흡 소스: 현재 본은 기존 RF run. 더 강한 proxy(`kiss_respiratory`, real과
  0.906)로 재생성하려면 RF 파이프라인으로 hw=real, bw=kiss 추출 후 `SleepVST_rf_model.pkl` 적용.
- ModernTCN(multivariate timeseries) 미구현.
