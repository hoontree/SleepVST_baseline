from src.data.datasets.SHHS import SHHS
from src.data.datasets.MESA import MESA
from src.data.datasets.KVSS import KVSS


def get_dataset(name, split='train', data_dir=None, exceptions=None):
    if name not in DATA_REGISTRY:
        raise ValueError(f"Dataset {name} is not registered.")
    
    dataset_class = DATA_REGISTRY[name]
    return dataset_class(split=split, data_dir=data_dir, exceptions=exceptions)