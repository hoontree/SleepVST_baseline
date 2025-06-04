import argparse
import sys

def get_parser():
    parser = argparse.ArgumentParser(description="Process EDF files to extract ECG and THOR RES signals.")
    
    # Preprocessing parameters
    parser.add_argument('--edf_dir_shhs', type=str, default='/tf/00_data/#_SHHS/polysomnography/edfs/', 
                        help='Directory containing EDF files')
    parser.add_argument('--edf_dir_mesa', type=str, default='/tf/01_code/mesa/polysomnography/edfs/',
                        help='Directory containing EDF files')
    parser.add_argument('--xml_dir_shhs', type=str, 
                        default='/tf/00_data/#_SHHS/polysomnography/annotations-events-nsrr/',
                        help='Directory containing XML files')
    parser.add_argument('--xml_dir_mesa', type=str, 
                        default='/tf/01_code/mesa/polysomnography/annotations-events-nsrr/',
                        help='Directory containing XML files')
    parser.add_argument('--shhs_npy_dir', type=str, default='/tf/01_code/mylittlecodes/SleepVST_baseline/data/shhs', 
                        help='Directory to save SHHS .npy files')
    parser.add_argument('--mesa_npy_dir', type=str, default='/tf/01_code/mylittlecodes/SleepVST_baseline/data/mesa',
                        help='Directory to save MESA .npy files')
    parser.add_argument('--kvss_npy_dir', type=str, default='/tf/01_code/mylittlecodes/SleepVST_baseline/data/kvss',
                        help='Directory to save KVSS .npy files')
    parser.add_argument('--csv_dir_kvss', type=str, default='/tf/00_AIoT2/video_signal/#_2021_Sleep_Video/30sec_labels',
                        help='Directory containing KVSS CSV files')
    
    parser.add_argument('--dataset', type=str, default='all',
                        help='shhs_mesa, kvss, all')
    parser.add_argument('--save_dir', type=str, 
                        default='/tf/01_code/mylittlecodes/SleepVST_baseline/data',
                        help='Directory to save .npy files')
    parser.add_argument('--patch_size', type=int, default=300,
                        help='Size of patches to extract from signals')
    parser.add_argument('--step', type=int, default=300,
                        help='Step size for patch extraction')
    parser.add_argument('--channels', type=str, nargs='+', default=['ECG', 'THOR RES'],
                        help='Channels to extract from EDF files')
    parser.add_argument('--lowcut', type=float, default=0.66,
                        help='Low cut frequency for bandpass filter')
    parser.add_argument('--highcut', type=float, default=2.8,
                        help='High cut frequency for bandpass filter')
    parser.add_argument('--fs', type=float, default=1000.0,
                        help='Sampling frequency of the signals')
    parser.add_argument('--normalize', action='store_true',
                        help='Whether to normalize the patches')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # 모델 파라미터
    parser.add_argument('--seq_len', type=int, default=240, help='Sequence length')
    parser.add_argument('--patch_hw', type=int, default=300, help='Heart waveform patch size')
    parser.add_argument('--patch_bw', type=int, default=150, help='Breath waveform patch size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    
    # 학습 파라미터
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--if_scratch', action='store_true', help='If training from scratch')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=3, help='Early stopping patience')
    
    # 데이터셋 파라미터
    parser.add_argument('--train_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads')
    
    # 저장 경로
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory to save output files')
    parser.add_argument('--log_name', type=str, default='train', help='Log file name')
    parser.add_argument('--gpu_ids', nargs='+', default=[0], help='GPU IDs to use')
    parser.add_argument('--preprocess_log_dir', type=str, default='processing/preprocess_log', help='Directory to save preprocessing logs')
    
    parser.add_argument('--mode', type=str, default='train_and_test', 
                        choices=['train', 'test', 'train_and_test', 'finetune'],
                        help='실행 모드 (train, test, train_and_test, finetune)')
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='Fine-tuning에 사용할 학습률 (기본: 기존 lr의 1/10)')
    parser.add_argument('--pretrained_checkpoint_dir', type=str, default='output',
                        help='Fine-tuning에 사용할 사전학습된 모델 체크포인트 경로')
    parser.add_argument('--kvss', action='store_true', help='Use KVSS dataset')
    # Jupyter Notebook 환경에서 빈 리스트 전달
    if 'ipykernel' in sys.modules:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

def parse_args():
    return get_parser()
