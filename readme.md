# SleepVST - Sleep Staging Model

SleepVST is a deep learning model designed to classify sleep stages using electrocardiogram (ECG) and respiratory (THOR) signals. It enables lightweight sleep stage analysis that can be deployed on wearable devices by utilizing ECG and respiratory waveform data.

## Project Structure
```bash
SleepVST_baseline\
├── config.py # Configuration file 
├── main.py # Main execution script 
├── checkpoint/ # Model checkpoint storage folder 
├── data/ # Dataset folder 
│ ├── MESA.py # MESA data loader 
│ └── SHHS.py # SHHS data loader 
├── models/ # Model definitions 
│ └── SleepVST.py # SleepVST model implementation 
├── processing/ # Data preprocessing scripts 
│ └── preprocess.py # Preprocessing pipeline 
├── utils/ # Utility functions 
│ ├── io.py # File I/O functions 
│ ├── logger.py # Logging configuration 
│ ├── util.py # General utility functions 
│ └── utils_data.py # Data processing utilities 
└── output/ # Output and log folder
```

## Installation and Setup

### Required Packages

```bash
pip install torch numpy scipy scikit-learn tqdm wandb mne psutil
```

## Data Preparation
Prepare original SHHS and MESA datasets
Run data preprocessing:
```bash
python processing/preprocess.py
```

## Model Training and Evaluation
### Training
```bash
python main.py --mode train --if_scratch --batch_size 128 --end_epoch 30
```
### Testing
```bash
python main.py --mode test
```
### Training and Testing Together
```bash
python main.py --mode train_and_test --if_scratch
```
Checkpoints are stored in `checkpoint/{run_name}.pth`. The `run_name` is based
on the `--log_name` argument (and `--mode` when fine-tuning) so multiple runs do
not overwrite each other.

## Key Parameters
```
--mode: Execution mode (train, test, train_and_test)
--if_scratch: Whether to train from scratch
--batch_size: Batch size
--end_epoch: Maximum training epochs
--lr: Learning rate
--seq_len: Sequence length (default: 240)
--d_model: Model dimension (default: 128)
--num_layers: Number of transformer encoder layers (default: 2)
--num_heads: Number of multi-head attention heads (default: 8)
```
## Datasets
This project uses the following datasets:

 SHHS (Sleep Heart Health Study) - Dataset for cardiovascular disease and sleep disorder research
### MESA (Multi-Ethnic Study of Atherosclerosis) - Multi-ethnic dataset for atherosclerosis research
## Model Architecture
SleepVST consists of the following components:

### Waveform Encoder - ResNet-based encoder for ECG and respiratory waveforms
### Transformer Encoder - Pre-LN transformer encoder for processing sequential data
### Classification Head - Linear layer for sleep stage classification
## Results and Performance
SHHS Dataset
```
[[85.1 12.4  0.3  2.2]
 [ 3.9 84.   7.7  4.4]
 [ 0.5 45.9 53.1  0.5]
 [ 1.5 10.1  0.3 88.2]]
```

MESA Dataset
```
 [[87.8 10.1  0.1  2.1]
 [ 4.8 88.5  3.   3.6]
 [ 0.3 71.  28.4  0.3]
 [ 2.  12.6  0.1 85.3]]
```