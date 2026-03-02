from collections.abc import Iterable
from typing import Any, TypeAlias

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer, RAdam

from args import ArgLiteral

ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


def get_grouped_params(
    model,
    no_decay=["bias", "LayerNorm", "fc_norm", "layernorm", "layer_norm"],
    base_lr=1e-5,
    weight_decay=1e-2,
):
    base_params_with_wd, base_params_without_wd = [], []

    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            base_params_without_wd.append(p)
        else:
            base_params_with_wd.append(p)

    return [
        {"params": base_params_with_wd, "weight_decay": weight_decay, "lr": base_lr},
        {"params": base_params_without_wd, "weight_decay": 0.0, "lr": base_lr},
    ]


def configure_optimizer(
    optimizer_name: ArgLiteral.OptimizerName,
    model_params: ParamsT,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Optimizer:
    """optimizer factory

    Args:
        optimizer_name (OptimizerName)
        model_params (ParamsT,): model parameters.
            Typically "model.parameters()"
        lr (float): learning rate.
        weight_decay (float): weight decay
        momentum (float, optional): momentum. Defaults to 0.9.

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        Optimizer: optimizer
    """

    if optimizer_name == ArgLiteral.OptimizerName.SGD:
        return SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name == ArgLiteral.OptimizerName.ADAM:
        return Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == ArgLiteral.OptimizerName.ADAMW:
        return AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == ArgLiteral.OptimizerName.RADAM:
        return RAdam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError("invalid optimizer_name")
