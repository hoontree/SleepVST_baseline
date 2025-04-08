import numpy as np
from utils.logger import Logger
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

logger = Logger(dir='output/log', name='SleepVST.mesa_loader')

class MetricsTracker:
    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.append(preds.detach().cpu().numpy().reshape(-1))
        self.targets.append(targets.detach().cpu().numpy().reshape(-1))

    def compute(self, labels:list=None):
        y_pred = np.concatenate(self.preds)
        y_true = np.concatenate(self.targets)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = (confusion_matrix(y_true, y_pred, labels=labels, normalize='true') * 100).round(1)

        return acc, f1, kappa, cm, cm_norm

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
    
