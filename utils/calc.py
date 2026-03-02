import torch


def stack_mean(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    if tensor_list[0] is None:
        return None
    return torch.stack(tensor_list, 0).mean(0)


def mean_list(list: list[float] | list[int]) -> float:
    if not list:
        return 0.0
    elif list[0] is None:
        return 0.0
    return sum(list) / len(list)
