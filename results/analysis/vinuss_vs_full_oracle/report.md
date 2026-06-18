# Proxy failure-mode analysis (G1-G4)

- matched epochs: **57,915**
- ViNUSS : acc=0.8042, kappa=0.6836
- Proxy  : acc=0.7642, kappa=0.6074

## Group summary

| group   | description                 |   count |   pct |
|:--------|:----------------------------|--------:|------:|
| G1      | ViNUSS correct, Proxy wrong |    7763 | 13.4  |
| G2      | ViNUSS wrong, Proxy correct |    5447 |  9.41 |
| G3      | Both correct                |   38812 | 67.02 |
| G4      | Both wrong                  |    5893 | 10.18 |

- G1 (ViNUSS 단독 정답) = 7,763 / G2 (Proxy 단독 정답) = 5,447
- McNemar χ²(1, cc) = 405.695 (>3.84 이면 p<0.05 로 두 모델 차이 유의)

## Group distribution by ground-truth stage

|       |   G1 |   G2 |    G3 |   G4 |   total |
|:------|-----:|-----:|------:|-----:|--------:|
| Wake  | 2404 | 1446 |  7915 | 2404 |   14169 |
| N1/N2 | 2787 | 3161 | 23877 | 1377 |   31202 |
| N3    | 1861 |  172 |  1862 | 1555 |    5450 |
| REM   |  711 |  668 |  5158 |  557 |    7094 |

## Files

- `group_summary.csv` — 그룹별 count/pct
- `group_by_stage.csv` — stage x group
- `group_by_subject.csv` — subject x group (+모델 acc, vinuss_gain 내림차순)
- `G1_proxy_confusion.csv` — G1 구간 Proxy 오분류 (gt x proxy_pred)
- `G2_vinuss_confusion.csv` — G2 구간 ViNUSS 오분류 (gt x vinuss_pred)
- `epoch_groups.csv` — per-epoch 전체 그룹 라벨
