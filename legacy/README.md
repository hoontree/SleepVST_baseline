# Legacy Code

This directory contains legacy code that has been replaced by the refactored version in the `src/` directory.

## Directory Structure

- `data/` - Legacy dataset implementations (replaced by `src/data/datasets/`)
- `models/` - Legacy model implementations (replaced by `src/models/`)
- `utils/` - Legacy utility functions (replaced by `src/common/`)
- `processing/` - Legacy preprocessing code (replaced by `src/data/preprocess/`)
- `scripts/` - Legacy standalone scripts (replaced by `src/cli.py` commands)

## Legacy Files

- `main.py` - Legacy training script (replaced by `src/cli.py`)
- `config.py` - Legacy argument parser (replaced by Hydra config in `config/`)
- `transfer_learning.py` - Legacy transfer learning script (replaced by `src/train/transfer.py`)
- `preprocess_SNUH.py` - Legacy SNUH preprocessing
- `respiratory.py` - Legacy respiratory signal processing

## Important

**Do not modify files in this directory.** This code is kept for reference only.

For active development, use the code in the `src/` directory with the CLI:
```bash
python -m src.cli --config-path=config --config-name=defaults command=<command>
```

Available commands: `pretrain`, `finetune`, `eval`, `preprocess`, `motionfeatures`, `extract_signal`, `transfer_to_video`
