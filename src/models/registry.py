"""Hydra-based model instantiation helpers."""

from hydra.utils import instantiate


def get_model(cfg):
    """Instantiate a model from a Hydra config node with ``_target_``.

    The old name-based registry has been replaced by Hydra's target resolution.
    Pass ``cfg.model`` or another config object that defines ``_target_``.
    """
    if isinstance(cfg, str):
        raise ValueError(
            "Name-based model registry is no longer supported. "
            "Pass a Hydra config node with `_target_`, e.g. cfg.model."
        )
    return instantiate(cfg)
