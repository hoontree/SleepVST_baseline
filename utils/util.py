import numpy as np
from utils.customlogger import Logger
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, classification_report
import csv
from config import parse_args

args = parse_args()
logger = Logger(dir='output/log', name='SleepVST.mesa_loader', run_name=args.run_name)

class MetricsTracker:
    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.append(preds.detach().cpu().numpy().reshape(-1))
        self.targets.append(targets.detach().cpu().numpy().reshape(-1))

    def compute(self, labels: list = None):
        y_pred = np.concatenate(self.preds)
        y_true = np.concatenate(self.targets)

        # Acc_T
        acc_T = accuracy_score(y_true, y_pred)

        # Acc_mu = macro recall
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        recalls = cm.diagonal() / cm.sum(axis=1)
        acc_mu = np.mean(recalls)

        # κ_T
        kappa_T = cohen_kappa_score(y_true, y_pred)

        # κ_mu (macro kappa): 클래스별 kappa 평균
        total = cm.sum()
        kappa_list = []
        for i in range(len(cm)):
            po = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            pe = (cm[i].sum() * cm[:, i].sum()) / (total ** 2)
            kappa_i = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
            kappa_list.append(kappa_i)
        kappa_mu = np.mean(kappa_list)

        cm_norm = (confusion_matrix(y_true, y_pred, labels=labels, normalize='true') * 100).round(1)

        cr = classification_report(
            y_true, y_pred, labels=labels,
            target_names=['Wake', 'N1/N2', 'N3', 'REM'], zero_division=0
        )

        return {
            'acc_T': acc_T,
            'acc_mu': acc_mu,
            'kappa_T': kappa_T,
            'kappa_mu': kappa_mu,
            'cm': cm,
            'cm_norm': cm_norm,
            'report': cr
        }

    def reset(self):
        self.preds = []
        self.targets = []


class AverageMeter(object):
	"""Computes and stores the average and current value, similar to timer"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0
  
def parse_csv(csv_path):
    """
    Args:
        csv_path (str)
    Returns:
        list: 수면 단계 정보가 포함된 딕셔너리 리스트
    """
    
    label_map = {
            "Wake": 0,
            "N1": 1,
            "N2": 1,
            "N3": 2,
            "REM": 3,
        }
    
    try:
        sleep_epochs = []
        with open(csv_path, 'r') as f:
            ann = [row for row in csv.DictReader(f)]
            if len(ann) == 0:
                logger.error(f"CSV 파일이 비어있습니다: {csv_path}")
                return []
            for row in ann:
                sleep_epochs.append({
                    "start": (float(row['Start_Epoch'])-1) * 30,
                    "duration": 30,
                    "label": label_map[row['Event_Label']]
                })
        return sleep_epochs
    except Exception as e:
        logger.error(f"CSV 파싱 오류 {csv_path}: {e}")
        return []
  
def parse_xml(xml_path):
    """
    Args:
        xml_path (str)
    Returns:
        list: 수면 단계 정보가 포함된 딕셔너리 리스트
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        label_map = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 1,
            "Stage 3 sleep": 2,
            "Stage 4 sleep": 2,
            "REM sleep": 3,
            "Unscored": 0,
            "Movement": 0,
        }

        sleep_epochs = []

        for event in root.iter('ScoredEvent'):
            event_type = event.find('EventType').text
            if 'Stages' not in str(event_type):
                continue

            concept = event.find('EventConcept').text
            start = float(event.find('Start').text)
            duration = float(event.find('Duration').text)
            for label in label_map.keys():
                if label in concept:
                    concept = label
                    break
            if concept == None:
                continue
            stage_label = label_map[concept]

            n_epochs = int(duration // 30)
            for i in range(n_epochs):
                sleep_epochs.append({
                    "start": start + i * 30,
                    "duration": 30,
                    "label": stage_label
                })

        return sleep_epochs
    except Exception as e:
        logger.error(f"XML 파싱 오류 {xml_path}: {e}")
        return []
    
