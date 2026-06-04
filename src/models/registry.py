from src.models.SleepVST import SleepVST, SleepVST_BW

MODEL_REGISTRY = {
    "SleepVST": SleepVST,
    "SleepVST_BW": SleepVST_BW,
}

def get_model(name, *args, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} is not registered.")
    model_class = MODEL_REGISTRY[name]
    return model_class(*args, **kwargs)