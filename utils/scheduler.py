import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    LRScheduler,
    StepLR,
)


def configure_scheduler(
    optimizer: Optimizer,
    use_scheduler: bool,
) -> LRScheduler:
    """scheduler factory

    Args:
        optimizer (Optimizer): optimizer
        use_scheduler (bool): flag if scheduler is used.
            Use StepLR if True, or dummy_scheduler (no lr scheduling) if False.

    Returns:
        LRScheduler: learning rate scheduler
    """
    if use_scheduler:
        return StepLR(
            optimizer,
            step_size=10,  # every 10 epoch
            gamma=0.1,  # lr = lr * 0.1
        )

    return dummy_scheduler(optimizer)


def dummy_scheduler(optimizer) -> LRScheduler:
    """A dummy scheduler that doesn't change lr because of factor=1.0"""
    return ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=65535,  # dummy max
    )


def configure_warmup_cosine_decay_lambda(total_steps, warmup_steps):
    def warmup_cosine_decay_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return warmup_cosine_decay_lambda


def configure_constant_lambda(warmup_steps=0):
    def constant_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return constant_lambda
