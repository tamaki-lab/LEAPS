import torch


def load_weight_from_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    pretrained = state_dict
    model_dict = model.state_dict()
    filtered = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
    unused = {k: v for k, v in pretrained.items() if k not in model_dict or v.shape != model_dict[k].shape}

    print()
    for k in unused.keys():
        print(f"Skip loading parameter: {k}, shape: {unused[k].shape}")
    print(f"Load {len(filtered)}/{len(pretrained)} parameters.\n")

    model_dict.update(filtered)
    model.load_state_dict(model_dict)


def load_weight_from_pth(model: torch.nn.Module, pth_path: str) -> None:
    pretrained = torch.load(pth_path, map_location="cpu")
    load_weight_from_state_dict(model, pretrained)
