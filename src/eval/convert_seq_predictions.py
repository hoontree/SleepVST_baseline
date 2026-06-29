"""Convert sequence-model predictions (MambaSL / ModernTCN ±LongMamba SDM) to the
standard 4-class prediction schema so they can feed proxy_failure_analysis.py (G1-G4).

입력 파일 스키마: `case_id, epoch_idx, true, pred` (5-class: 0=Wake,1=N1,2=N2,3=N3,4=REM),
epoch_idx 는 1-based.

표준 스키마(기존 full_oracle/ViNUSS 와 동일)로 변환:
    subject_id, recording_idx, epoch_idx, ground_truth_label, ground_truth_name,
    prediction_label, prediction_name
- case_id -> subject_id, recording_idx = 0
- epoch_idx -1 (1-based -> ViNUSS/oracle 의 0-based 와 정렬; --reference 로 GT 정합성 검증)
- 5-class -> 4-class: {0:0, 1:1, 2:1, 3:2, 4:3} (N1,N2 -> N1/N2)

사용 예:
    python -m src.eval.convert_seq_predictions \
        --reference results/predictions/vinuss_predictions_2026-06-18.csv
변환 결과는 같은 폴더에 `<stem>_std4.csv` 로 저장.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# 5-class(0=W,1=N1,2=N2,3=N3,4=REM) -> 4-class(0=W,1=N1/N2,2=N3,3=REM)
MAP_5TO4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
STAGE_NAMES = {0: "Wake", 1: "N1/N2", 2: "N3", 3: "REM"}
EPOCH_SHIFT = -1  # 1-based -> 0-based

DEFAULT_FILES = [
    "results/predictions/mambasl_predictions.csv",
    "results/predictions/mambasl_mamba.csv",
    "results/predictions/modernTCN_predictions.csv",
    "results/predictions/modernTCN_mamba_predictions.csv",
]


def convert(path: str, reference: pd.DataFrame | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"case_id", "epoch_idx", "true", "pred"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"[error] {path}: 필수 컬럼 누락 {missing} (있는 컬럼: {list(df.columns)})")

    out = pd.DataFrame({
        "subject_id": df["case_id"].astype(str),
        "recording_idx": 0,
        "epoch_idx": df["epoch_idx"].astype(int) + EPOCH_SHIFT,
        "ground_truth_label": df["true"].astype(int).map(MAP_5TO4),
        "prediction_label": df["pred"].astype(int).map(MAP_5TO4),
    })
    out["ground_truth_name"] = out["ground_truth_label"].map(STAGE_NAMES)
    out["prediction_name"] = out["prediction_label"].map(STAGE_NAMES)
    out = out[["subject_id", "recording_idx", "epoch_idx", "ground_truth_label",
               "ground_truth_name", "prediction_label", "prediction_name"]]

    if reference is not None:
        chk = reference.merge(out, on=["subject_id", "epoch_idx"],
                              suffixes=("_ref", ""), how="inner")
        n_mis = int((chk["ground_truth_label_ref"] != chk["ground_truth_label"]).sum())
        print(f"    GT check vs reference: matched={len(chk):,}, mismatch={n_mis:,}"
              + ("  <-- 경고: 정렬 의심" if n_mis else "  OK"))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="seq-model preds -> 표준 4-class 스키마")
    p.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--reference", default=None,
                   help="GT 정합성 검증용 표준 스키마 CSV(예: ViNUSS). (subject_id, epoch_idx) 로 join")
    p.add_argument("--suffix", default="_std4")
    args = p.parse_args(argv)

    ref = None
    if args.reference:
        ref = pd.read_csv(args.reference)[["subject_id", "epoch_idx", "ground_truth_label"]]
        ref["subject_id"] = ref["subject_id"].astype(str)

    for path in args.files:
        print(f"[convert] {path}")
        out = convert(path, ref)
        stem, ext = os.path.splitext(path)
        dst = f"{stem}{args.suffix}{ext}"
        out.to_csv(dst, index=False)
        print(f"    -> {dst}  ({len(out):,} epochs)")


if __name__ == "__main__":
    main()
