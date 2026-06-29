# SleepVST: Sleep Stage Classification using Video and Sensor Transformer

Implementation of SleepVST paper: https://arxiv.org/abs/2404.03831

## Project Structure

```
SleepVST_baseline/
├── src/                          # Main source code (use this for development)
│   ├── cli_train.py              # Entry point: training/eval (pretrain, finetune, test, transfer_to_video)
│   ├── cli_preprocess.py         # Entry point: data preprocessing (edf/motion/respiratory)
│   ├── cli_motionfeatures.py     # Entry point: motion feature extraction from videos
│   ├── cli_extract_respiratory.py# Entry point: respiratory signal extraction from videos
│   ├── cli_transfer.py           # Entry point: transfer_to_video (dedicated shim)
│   ├── cli.py                    # Backward-compatible shim routing to the two above
│   ├── utils/                    # Common utilities (logger, utils)
│   ├── data/                     # Datasets, registry, preprocessing pipelines
│   │   ├── datasets/             # KVSS / MESA / SHHS loaders
│   │   └── preprocess/           # motion + respiratory extraction, filters
│   │       ├── respiratory_extraction.py    # v1 optical-flow proxy
│   │       ├── respiratory_extraction_v2.py # v2 robust PBM proxy
│   │       ├── respiratory_pipeline.py      # single-process pipeline
│   │       ├── respiratory_pipeline_mp.py   # multiprocess pipeline
│   │       ├── motion_extractor.py
│   │       └── motion_mag/
│   ├── models/                   # SleepVST, RFClassifier, registry
│   ├── train/                    # Training loops (loop.py, transfer.py, trainer.py)
│   └── eval/                     # Evaluation metrics and analysis scripts
│       ├── metrics.py
│       ├── proxy_failure_analysis.py        # G1-G4 failure-mode breakdown
│       ├── respiratory_proxy_validation.py  # proxy vs EDF reference scoring
│       ├── full_oracle_predict.py           # regenerate full-oracle predictions
│       ├── make_vinuss_predictions.py       # convert ViNUSS raw results to schema
│       └── convert_seq_predictions.py       # 5-class → 4-class prediction mapping
├── scripts/                      # One-shot analysis / comparison scripts
│   └── compare_proxy_v1_v2.py   # Side-by-side v1 vs v2 respiratory proxy scoring
├── config/                       # Hydra configurations (data/model/mode/preprocess/train)
│   ├── defaults.yaml
│   ├── extraction.yaml
│   ├── command2/                 # Per-command config overrides
│   ├── mode/                     # pretrain / finetune / test / transfer
│   ├── preprocess/               # respiratory / motionfeatures / EDF configs
│   └── train/
├── notebooks/                    # Analysis / exploration notebooks (kept for reference)
├── results/                      # Generated experiment outputs (CSV/TXT)
│   ├── RESULTS_INDEX.md          # Model comparison table + file→role mapping
│   ├── predictions/              # Per-epoch test predictions (date-suffixed per run)
│   ├── metrics/                  # Per-subject metrics + subject_stats
│   ├── analysis/                 # Fairness, intensity, proxy failure (G1-G4), test metadata
│   ├── label_comparison/         # Per-subject CSV-vs-JSON label comparison dumps
│   ├── proxy_validation_v1.csv   # v1 proxy vs EDF reference scores
│   └── movinet_mamba_test/       # MoViNet/Mamba ablation results
├── figures/                      # All plots/figures (gitignored — PNGs)
├── models/                       # Trained RF model pickles (gitignored — large)
├── archive/                      # Disposable debug scripts / one-off notebooks (gitignored, kept on disk)
│   ├── notebooks/
│   └── scripts/
├── data/                         # Datasets, .npy features (gitignored)
├── test_sample/                  # Per-subject test samples (gitignored)
├── checkpoint/                   # Model checkpoints (gitignored)
├── logs/  output/  outputs/  wandb/   # Runtime logs & training outputs (gitignored)
└── readme.md
```

> **Note on duplicated outputs:** `results/predictions/` and `results/metrics/` keep more than one
> dated copy of the same file (e.g. `..._2025-11-08.csv` vs `..._2025-11-09.csv`). These come from
> different evaluation runs and are intentionally all preserved — the date suffix marks the run.
> See `results/RESULTS_INDEX.md` for the authoritative model↔file mapping and metric summary.

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install hydra-core omegaconf
pip install numpy scipy scikit-learn
pip install tqdm wandb
```

## Usage

The project uses Hydra for configuration management. Commands are split across
focused entry points by task:

- **Training / evaluation** (GPU): `python -m src.cli_train`
- **EDF preprocessing** (CPU): `python -m src.cli_preprocess`
- **Motion feature extraction** (CPU / multiprocessing): `python -m src.cli_motionfeatures`
- **Respiratory video extraction** (CPU / multiprocessing): `python -m src.cli_extract_respiratory`

> `python -m src.cli command=<command>` is kept for older training and EDF preprocessing
> commands, but the dedicated entry points are preferred.

### Available Commands

#### 1. Pretraining on SHHS + MESA
```bash
python -m src.cli_train command=pretrain
```

#### 2. Fine-tuning on KVSS
```bash
python -m src.cli_train command=finetune
```

#### 3. Evaluation
```bash
python -m src.cli_train command=test
```

#### 4. Data Preprocessing
```bash
# Extract signals from EDF files (SHHS/MESA/SNUH datasets)
python -m src.cli_preprocess command=preprocess

# Extract respiratory signals from EDF files
python -m src.cli_preprocess command=preprocess_respiratory_edf

# Extract motion features from videos (KVSS dataset)
python -m src.cli_motionfeatures

# Extract respiratory signals from videos (KVSS dataset)
python -m src.cli_extract_respiratory
```

For detailed information on respiratory signal extraction, see [RESPIRATORY_EXTRACTION_GUIDE.md](RESPIRATORY_EXTRACTION_GUIDE.md).

#### 5. Transfer Learning to Video Domain
```bash
# Via the main training CLI
python -m src.cli_train command=transfer_to_video

# Or via the dedicated shim
python -m src.cli_transfer
```

#### 6. Proxy Validation & Failure Analysis
```bash
# Compare v1 (optical-flow) vs v2 (robust PBM) respiratory proxy
python scripts/compare_proxy_v1_v2.py [--records 10] [--epochs 20]

# Validate proxy against EDF reference (per-epoch r, br_mae, etc.)
python -m src.eval.respiratory_proxy_validation \
    --proxy-root data/resp_proxy_video_epochs \
    --ref-dir    data/resp_proxy_video \
    --out-csv results/proxy_validation_v1.csv

# G1-G4 failure-mode breakdown (ViNUSS vs proxy model)
python -m src.eval.proxy_failure_analysis \
    --vinuss results/predictions/vinuss_predictions_2026-06-18.csv \
    --proxy  results/predictions/sleepvst_dual_predictions_2025-11-09.csv
```

### Configuration

Override any configuration using Hydra syntax:

```bash
# Change batch size and learning rate
python -m src.cli_train command=pretrain train.batch_size=256 train.lr=0.0005

# Use different GPU
python -m src.cli_train command=pretrain system.gpu_ids='0,1'

# Change dataset path
python -m src.cli_train command=pretrain data.shhs.root=/path/to/shhs
```

## Model Architecture

- **Input**: ECG waveform (240×300) + Respiratory waveform (240×150)
- **Encoder**: ResNet-based CNN → Transformer
- **Output**: 4-class sleep stage classification (Wake, N1/N2, N3, REM)

## Datasets

- **SHHS**: Sleep Heart Health Study (public)
- **MESA**: Multi-Ethnic Study of Atherosclerosis (public)
- **KVSS**: Korean Video Sleep Study (proprietary)

## Citation

If you use this code, please cite:
```
@article{sleepvst2024,
  title={SleepVST: Sleep Stage Classification using Video and Sensor Transformer},
  url={https://arxiv.org/abs/2404.03831},
  year={2024}
}
```
