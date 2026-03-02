# LEAPS + base model で推論

# base_model

import argparse
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning.pytorch.plugins import TorchSyncBatchNorm

from args import ArgLiteral, ArgParse
from dataloader import TrainValDataModule
from model import config_factory, configure_model
from utils import configure_callbacks
from utils.accuracy import compute_topk_accuracy


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        command_line_args: argparse.Namespace,
        exp_name: str,
        n_classes: int,
    ):
        super().__init__()

        self.exp_name = exp_name
        self.args = command_line_args

        self.model_name = ArgLiteral.SupportedModels(self.args.model_name)

        model_config = config_factory(
            model_name=self.model_name,
            n_classes=n_classes,
            args=self.args,
        )
        self.model = configure_model(model_config)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        log_suffix = f"_{batch['dataset_name']}"
        self.log_suffix = batch["dataset_name"]

        video = batch["video"]

        label_np = batch["label"]
        labels = torch.tensor(label_np).to(video.device)

        log_params = {}

        outputs = self.model.multiview_forward(video, labels=labels)  # type: ignore
        logits = outputs.logits
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(logits, labels, topk=(1, 5), return_topk_correct_index=False)

        log_params.update(
            {
                f"val_top1{log_suffix}": top1,
                f"val_top5{log_suffix}": top5,
                f"val_loss{log_suffix}": loss.item(),  # type: ignore
            }
        )

        self.log_dict(log_params, on_step=False, on_epoch=True, prog_bar=True)

        return video


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    exp_name = datetime.now().strftime("%m%d%H%M")

    data_module = TrainValDataModule(
        command_line_args=args,
    )

    model_lightning = LightningModel(
        command_line_args=args,
        n_classes=data_module.n_classes,
        exp_name=exp_name,
    )

    callbacks = configure_callbacks()

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        # strategy="ddp",
        strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=args.log_interval_steps,
        precision="bf16-true",  # for BF16 training, use with caution for nan/inf
        limit_val_batches=args.max_val_step,
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
    )

    trainer.validate(
        model=model_lightning,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
