"""Proxy failure-mode analysis (G1-G4).

ViNUSS(주 모델) 와 proxy-first method(SleepVST dual) 의 per-epoch 예측을 정렬해
각 테스트 epoch 을 네 그룹으로 분류하고, 그룹별 통계/혼동행렬을 뽑는다.

    G1: ViNUSS correct, Proxy wrong   (ViNUSS 가 이기는 구간)
    G2: ViNUSS wrong,   Proxy correct (Proxy 가 이기는 구간)
    G3: Both correct
    G4: Both wrong

ViNUSS per-epoch 추론은 오래 걸리므로, 결과 CSV 가 도착하면 바로 돌릴 수 있게 만들어 둔 것.
ViNUSS CSV 는 기존 예측 CSV 와 같은 스키마라고 가정한다:
    subject_id, recording_idx, epoch_idx, ground_truth_label, prediction_label
(이름 컬럼 ground_truth_name / prediction_name 은 있으면 쓰고 없으면 무시)

사용 예:
    python -m src.eval.proxy_failure_analysis \
        --vinuss results/predictions/vinuss_predictions_2026-06-18.csv \
        --proxy  results/predictions/sleepvst_dual_predictions_2025-11-09.csv

같은 스크립트로 ViNUSS vs oracle / ViNUSS vs ModernTCN 도 가능(--proxy 를 바꾸면 됨).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# 4-class sleep stage 라벨 (src/eval/metrics.py 와 동일)
LABELS = [0, 1, 2, 3]
STAGE_NAMES = ["Wake", "N1/N2", "N3", "REM"]
# subject 당 recording 1개이고 파일마다 recording_idx 스킴이 달라, 정렬 키는
# (subject_id, epoch_idx) 로 둔다. recording_idx 는 있으면 참고용으로만 둔다.
KEY = ["subject_id", "epoch_idx"]
REQUIRED = KEY + ["ground_truth_label", "prediction_label"]


def _load(path: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[error] {model_name} 예측 파일이 없습니다: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(
            f"[error] {model_name} CSV ({path}) 에 필수 컬럼이 없습니다: {missing}\n"
            f"        존재하는 컬럼: {list(df.columns)}\n"
            f"        기대 스키마: {REQUIRED}"
        )
    df = df[REQUIRED].copy()
    df["epoch_idx"] = df["epoch_idx"].astype(int)
    df["ground_truth_label"] = df["ground_truth_label"].astype(int)
    df["prediction_label"] = df["prediction_label"].astype(int)
    return df


def _model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS),
    }


def align(vinuss: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """(subject, recording, epoch) 기준 inner join + 정합성 검사."""
    merged = vinuss.merge(
        proxy, on=KEY, how="inner", suffixes=("_vinuss", "_proxy")
    )

    n_v, n_p, n_m = len(vinuss), len(proxy), len(merged)
    print(f"[align] ViNUSS epochs={n_v:,} | Proxy epochs={n_p:,} | matched={n_m:,}")
    if n_m == 0:
        sys.exit("[error] 정렬된 epoch 이 0개입니다. key 컬럼 값이 두 파일에서 일치하는지 확인하세요.")
    if n_m < min(n_v, n_p):
        print(f"[align][warn] 일부 epoch 이 한쪽에만 존재합니다 "
              f"(ViNUSS 단독 {n_v - n_m:,}, Proxy 단독 {n_p - n_m:,}). "
              f"교집합 {n_m:,} 개로만 분석합니다.")

    # ground truth 정합성: 같은 epoch 의 GT 가 두 파일에서 다르면 라벨 정렬이 깨진 것
    gt_mismatch = (merged["ground_truth_label_vinuss"]
                   != merged["ground_truth_label_proxy"])
    if gt_mismatch.any():
        print(f"[align][warn] ground_truth 가 두 파일에서 다른 epoch {gt_mismatch.sum():,}개 발견 — "
              f"epoch 정렬/라벨 매핑을 점검하세요. ViNUSS 쪽 GT 를 기준으로 진행합니다.")

    merged = merged.rename(columns={
        "ground_truth_label_vinuss": "gt",
        "prediction_label_vinuss": "vinuss_pred",
        "prediction_label_proxy": "proxy_pred",
    })[KEY + ["gt", "vinuss_pred", "proxy_pred"]]
    return merged


def assign_groups(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["vinuss_correct"] = m["vinuss_pred"] == m["gt"]
    m["proxy_correct"] = m["proxy_pred"] == m["gt"]

    conditions = [
        m["vinuss_correct"] & ~m["proxy_correct"],   # G1
        ~m["vinuss_correct"] & m["proxy_correct"],   # G2
        m["vinuss_correct"] & m["proxy_correct"],    # G3
        ~m["vinuss_correct"] & ~m["proxy_correct"],  # G4
    ]
    m["group"] = np.select(conditions, ["G1", "G2", "G3", "G4"], default="?")
    return m


def write_outputs(m: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    total = len(m)

    # 0) 모델 전체 성능 + McNemar
    vm = _model_metrics(m["gt"].values, m["vinuss_pred"].values)
    pm = _model_metrics(m["gt"].values, m["proxy_pred"].values)

    # 1) 그룹 요약
    grp = (m["group"].value_counts()
           .reindex(["G1", "G2", "G3", "G4"], fill_value=0))
    summary = pd.DataFrame({
        "group": grp.index,
        "description": [
            "ViNUSS correct, Proxy wrong",
            "ViNUSS wrong, Proxy correct",
            "Both correct",
            "Both wrong",
        ],
        "count": grp.values,
        "pct": (grp.values / total * 100).round(2),
    })
    summary.to_csv(os.path.join(outdir, "group_summary.csv"), index=False)

    # 2) ground-truth stage 별 그룹 분포
    by_stage = (m.groupby(["gt", "group"]).size()
                .unstack(fill_value=0)
                .reindex(columns=["G1", "G2", "G3", "G4"], fill_value=0))
    by_stage.index = [STAGE_NAMES[i] for i in by_stage.index]
    by_stage["total"] = by_stage.sum(axis=1)
    by_stage.to_csv(os.path.join(outdir, "group_by_stage.csv"))

    # 3) subject 별 그룹 분포 (+ 각 모델 accuracy)
    by_subj = (m.groupby(["subject_id", "group"]).size()
               .unstack(fill_value=0)
               .reindex(columns=["G1", "G2", "G3", "G4"], fill_value=0))
    by_subj["n_epochs"] = by_subj.sum(axis=1)
    subj_acc = m.groupby("subject_id").agg(
        vinuss_acc=("vinuss_correct", "mean"),
        proxy_acc=("proxy_correct", "mean"),
    )
    by_subj = by_subj.join(subj_acc)
    # ViNUSS 가 가장 많이 이긴(=G1-G2 큰) subject 순
    by_subj["vinuss_gain"] = by_subj["G1"] - by_subj["G2"]
    by_subj = by_subj.sort_values("vinuss_gain", ascending=False)
    by_subj.to_csv(os.path.join(outdir, "group_by_subject.csv"))

    # 4) 실패 모드 혼동행렬
    #    G1 = ViNUSS 가 맞고 Proxy 가 틀린 구간에서 Proxy 가 무엇으로 틀렸나 (gt x proxy_pred)
    #    G2 = Proxy 가 맞고 ViNUSS 가 틀린 구간에서 ViNUSS 가 무엇으로 틀렸나 (gt x vinuss_pred)
    def _cm(sub: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        cm = confusion_matrix(sub["gt"], sub[pred_col], labels=LABELS)
        return pd.DataFrame(cm, index=STAGE_NAMES, columns=STAGE_NAMES)

    g1 = m[m["group"] == "G1"]
    g2 = m[m["group"] == "G2"]
    if len(g1):
        _cm(g1, "proxy_pred").to_csv(
            os.path.join(outdir, "G1_proxy_confusion.csv"))
    if len(g2):
        _cm(g2, "vinuss_pred").to_csv(
            os.path.join(outdir, "G2_vinuss_confusion.csv"))

    # 5) per-epoch 전체 덤프 (drill-down 용)
    m.to_csv(os.path.join(outdir, "epoch_groups.csv"), index=False)

    # 6) 사람이 읽는 리포트
    n1, n2 = int(grp["G1"]), int(grp["G2"])
    # McNemar (불일치쌍 b=G1, c=G2 에 대한 연속성 보정 카이제곱)
    if (n1 + n2) > 0:
        mcnemar_chi2 = (abs(n1 - n2) - 1) ** 2 / (n1 + n2)
    else:
        mcnemar_chi2 = float("nan")

    lines = []
    lines.append("# Proxy failure-mode analysis (G1-G4)\n")
    lines.append(f"- matched epochs: **{total:,}**")
    lines.append(f"- ViNUSS : acc={vm['accuracy']:.4f}, kappa={vm['kappa']:.4f}")
    lines.append(f"- Proxy  : acc={pm['accuracy']:.4f}, kappa={pm['kappa']:.4f}\n")
    lines.append("## Group summary\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append(f"- G1 (ViNUSS 단독 정답) = {n1:,} / G2 (Proxy 단독 정답) = {n2:,}")
    lines.append(f"- McNemar χ²(1, cc) = {mcnemar_chi2:.3f} "
                 f"(>3.84 이면 p<0.05 로 두 모델 차이 유의)\n")
    lines.append("## Group distribution by ground-truth stage\n")
    lines.append(by_stage.to_markdown())
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `group_summary.csv` — 그룹별 count/pct")
    lines.append("- `group_by_stage.csv` — stage x group")
    lines.append("- `group_by_subject.csv` — subject x group (+모델 acc, vinuss_gain 내림차순)")
    lines.append("- `G1_proxy_confusion.csv` — G1 구간 Proxy 오분류 (gt x proxy_pred)")
    lines.append("- `G2_vinuss_confusion.csv` — G2 구간 ViNUSS 오분류 (gt x vinuss_pred)")
    lines.append("- `epoch_groups.csv` — per-epoch 전체 그룹 라벨")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"[done] 결과 저장: {outdir}/")


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Proxy failure-mode analysis (G1-G4)")
    p.add_argument("--vinuss", required=True,
                   help="ViNUSS per-epoch 예측 CSV")
    p.add_argument("--proxy",
                   default="results/predictions/sleepvst_dual_predictions_2025-11-09.csv",
                   help="Proxy(SleepVST dual) per-epoch 예측 CSV "
                        "(oracle/ModernTCN 비교 시 이 경로를 바꾸면 됨)")
    p.add_argument("--outdir", default="results/analysis/proxy_failure",
                   help="결과 출력 디렉터리")
    args = p.parse_args(argv)

    vinuss = _load(args.vinuss, "ViNUSS")
    proxy = _load(args.proxy, "Proxy")
    merged = align(vinuss, proxy)
    grouped = assign_groups(merged)
    write_outputs(grouped, args.outdir)


if __name__ == "__main__":
    main()
