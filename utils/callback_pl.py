
from lightning.pytorch.callbacks import ModelSummary


def configure_callbacks():
    """callback factory

    Returns:
        List[pl.callbacks]: callbacks
    """

    callbacks = [
        ModelSummary(max_depth=4)
    ]

    return callbacks
