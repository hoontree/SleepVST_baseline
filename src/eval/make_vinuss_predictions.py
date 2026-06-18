"""ViNUSS per-epoch 예측을 표준 스키마로 변환.

ViNUSS(=MoViNet+Mamba) 추론 결과는 `results/movinet_mamba_test/stage2_test_results/
test_results_all_4class.csv` 에 `subject, epoch, preds, labels` 형태로 저장돼 있다
(stage2 = finetuned; per-subject acc 가 metrics/vinuss_subject_stats.csv 와 일치하는 동일물).

proxy_failure_analysis.py 와 호환되는 스키마로 바꿔 저장한다:
    subject_id, recording_idx, epoch_idx, ground_truth_label, ground_truth_name,
    prediction_label, prediction_name

사용:
    python -m src.eval.make_vinuss_predictions
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os

import pandas as pd

STAGE_NAMES = {0: "Wake", 1: "N1+N2", 2: "N3", 3: "REM"}
DEFAULT_SRC = "results/movinet_mamba_test/stage2_test_results/test_results_all_4class.csv"


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert ViNUSS (movinet stage2) preds to standard schema")
    p.add_argument("--src", default=DEFAULT_SRC, help="movinet stage2 4class CSV")
    p.add_argument("--out", default=f"results/predictions/vinuss_predictions_{_dt.date.today().isoformat()}.csv")
    args = p.parse_args(argv)

    if not os.path.exists(args.src):
        raise SystemExit(f"[error] source 없음: {args.src}")

    v = pd.read_csv(args.src)
    out = pd.DataFrame({
        "subject_id": v["subject"],
        "recording_idx": 0,  # KVSS 는 subject 당 recording 1개; 분석 merge 는 (subject_id, epoch_idx)
        "epoch_idx": v["epoch"].astype(int),
        "ground_truth_label": v["labels"].astype(int),
        "ground_truth_name": v["labels"].astype(int).map(STAGE_NAMES),
        "prediction_label": v["preds"].astype(int),
        "prediction_name": v["preds"].astype(int).map(STAGE_NAMES),
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    acc = (out.prediction_label == out.ground_truth_label).mean()
    print(f"[done] {len(out):,} epochs, {out.subject_id.nunique()} subjects, acc={acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
