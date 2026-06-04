# SleepVST: Sleep Stage Classification using Video and Sensor Transformer

Implementation of SleepVST paper: https://arxiv.org/abs/2404.03831

## Project Structure

```
SleepVST_baseline/
├── src/                          # Main source code (use this for development)
│   ├── cli.py                    # CLI entry point with Hydra
│   ├── common/                   # Common utilities (logger, utils)
│   ├── data/                     # Datasets, registry, preprocessing pipelines
│   │   ├── datasets/             # KVSS / MESA / SHHS loaders
│   │   └── preprocess/           # motion + respiratory extraction, filters
│   ├── models/                   # SleepVST, RFClassifier, registry
│   ├── train/                    # Training loops (loop.py, transfer.py)
│   └── eval/                     # Evaluation metrics
├── config/                       # Hydra configurations (data/model/mode/preprocess/train)
├── notebooks/                    # Analysis / exploration notebooks (kept for reference)
├── results/                      # Generated experiment outputs (CSV/TXT)
│   ├── predictions/              # Per-epoch test predictions (date-suffixed per run)
│   ├── metrics/                  # Per-subject metrics + subject_stats
│   ├── analysis/                 # Fairness, intensity, test-sample metadata, label-comparison logs
│   ├── label_comparison/         # Per-subject CSV-vs-JSON label comparison dumps
│   └── movinet_mamba_test/       # MoViNet/Mamba ablation results
├── figures/                      # All plots/figures (gitignored — PNGs)
├── models/                       # Trained RF model pickles (gitignored — large)
├── legacy/                       # Pre-pretrain preprocessing code (reference only — do not modify)
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

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install hydra-core omegaconf
pip install numpy scipy scikit-learn
pip install tqdm wandb
```

## Usage

The project uses Hydra for configuration management. All commands are executed through the CLI:

```bash
python -m src.cli [OPTIONS]
```

### Available Commands

#### 1. Pretraining on SHHS + MESA
```bash
python -m src.cli command=pretrain
```

#### 2. Fine-tuning on KVSS
```bash
python -m src.cli command=finetune
```

#### 3. Evaluation
```bash
python -m src.cli command=eval
```

#### 4. Data Preprocessing
```bash
# Extract signals from EDF files (SHHS/MESA/SNUH datasets)
python -m src.cli command=preprocess

# Extract motion features from videos (KVSS dataset)
python -m src.cli command=motionfeatures

# Extract respiratory signals from videos (KVSS dataset)
python -m src.cli mode=extract_respiratory
```

For detailed information on respiratory signal extraction, see [RESPIRATORY_EXTRACTION_GUIDE.md](RESPIRATORY_EXTRACTION_GUIDE.md).

#### 5. Transfer Learning to Video Domain
```bash
python -m src.cli command=transfer_to_video
```

### Configuration

Override any configuration using Hydra syntax:

```bash
# Change batch size and learning rate
python -m src.cli command=pretrain train.batch_size=256 train.lr=0.0005

# Use different GPU
python -m src.cli command=pretrain system.gpu_ids='0,1'

# Change dataset path
python -m src.cli command=pretrain data.shhs.root=/path/to/shhs
```

## Model Architecture

- **Input**: ECG waveform (240×300) + Respiratory waveform (240×150)
- **Encoder**: ResNet-based CNN → Transformer
- **Output**: 4-class sleep stage classification (Wake, N1/N2, N3, REM)

## Datasets

- **SHHS**: Sleep Heart Health Study (public)
- **MESA**: Multi-Ethnic Study of Atherosclerosis (public)
- **KVSS**: Korean Video Sleep Study (proprietary)

## Legacy Code

The `legacy/` directory contains deprecated code that has been refactored. Do not modify files in this directory. See [legacy/README.md](legacy/README.md) for details.

## Citation

If you use this code, please cite:
```
@article{sleepvst2024,
  title={SleepVST: Sleep Stage Classification using Video and Sensor Transformer},
  url={https://arxiv.org/abs/2404.03831},
  year={2024}
}
```