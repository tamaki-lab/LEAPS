import torch


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1,),
    return_topk_correct_index: bool = False,
) -> tuple[float, ...] | tuple[tuple[float, ...], torch.Tensor]:
    """Computes the accuracy over top-k predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411

    Args:
        logits (torch.Tensor): model logits of the batch.
            The shape is (B, L) for batchsize B and number of labels L
        labels (torch.Tensor): labels of the batch
            The shape is (B, )
        topk (tuple of int, optional):
            k for computing top-k accuracy. Defaults to (1,).
                topk=(1,) returns (top1,)
                topk=(1,5) returns (top1, top5)

    Returns:
        Tuple[float]: top1 accuracy, or list of top-k accuracy values
    """
    assert topk[0] == 1, "topk[0] should be top1"
    assert len(topk) >= 1

    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if batch_size == 0:
                res.append(0.0)
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() * 100.0 / batch_size)

        # --- optionally build per-sample mask -----------------------------------
        if return_topk_correct_index:
            if batch_size == 0:
                per_sample_correct = torch.empty(0, len(topk), dtype=torch.bool, device=logits.device)
            else:
                # each column j: correct within topk[j]
                per_sample_correct = torch.stack([correct[:k].any(dim=0) for k in topk], dim=1)  # (B, len(topk))
            return tuple(res), per_sample_correct

        return tuple(res)
