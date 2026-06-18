# Proxy failure-mode analysis (G1-G4)

- matched epochs: **57,901**
- ViNUSS : acc=0.8042, kappa=0.6835
- Proxy  : acc=0.6118, kappa=0.2363

## Group summary

| group   | description                 |   count |   pct |
|:--------|:----------------------------|--------:|------:|
| G1      | ViNUSS correct, Proxy wrong |   15815 | 27.31 |
| G2      | ViNUSS wrong, Proxy correct |    4676 |  8.08 |
| G3      | Both correct                |   30747 | 53.1  |
| G4      | Both wrong                  |    6663 | 11.51 |

- G1 (ViNUSS 단독 정답) = 15,815 / G2 (Proxy 단독 정답) = 4,676
- McNemar χ²(1, cc) = 6054.123 (>3.84 이면 p<0.05 로 두 모델 차이 유의)

## Group distribution by ground-truth stage

|       |   G1 |   G2 |    G3 |   G4 |   total |
|:------|-----:|-----:|------:|-----:|--------:|
| Wake  | 5476 |  595 |  4832 | 3254 |   14157 |
| N1/N2 | 1540 | 4035 | 25123 |  503 |   31201 |
| N3    | 3107 |   34 |   616 | 1693 |    5450 |
| REM   | 5692 |   12 |   176 | 1213 |    7093 |

## Files

- `group_summary.csv` — 그룹별 count/pct
- `group_by_stage.csv` — stage x group
- `group_by_subject.csv` — subject x group (+모델 acc, vinuss_gain 내림차순)
- `G1_proxy_confusion.csv` — G1 구간 Proxy 오분류 (gt x proxy_pred)
- `G2_vinuss_confusion.csv` — G2 구간 ViNUSS 오분류 (gt x vinuss_pred)
- `epoch_groups.csv` — per-epoch 전체 그룹 라벨
