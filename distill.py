import argparse
import os
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins import TorchSyncBatchNorm
from torch.optim.lr_scheduler import LambdaLR

from args import ArgLiteral, ArgParse
from dataloader import TrainValDataModule
from model import config_factory, configure_model
from utils import (
    configure_callbacks,
    configure_constant_lambda,
    configure_optimizer,
    configure_warmup_cosine_decay_lambda,
    get_grouped_params,
)


class AttnDistillLightningModel(pl.LightningModule):
    def __init__(
        self,
        command_line_args: argparse.Namespace,
        exp_name: str,
        max_steps: int,
    ):
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name
        self.scheduler_name = ArgLiteral.SchedulerName(self.args.scheduler_name)
        self.warmup_epoch = self.args.warmup_epoch
        self.max_steps = max_steps

        self.model_name = ArgLiteral.SupportedModels("attn_distill")

        model_config = config_factory(
            model_name=self.model_name,
            n_classes=100,
            args=self.args,
        )
        self.model = configure_model(model_config)

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer_name = ArgLiteral.OptimizerName(self.args.optimizer_name)

        grouped_params = get_grouped_params(
            self.model,
            base_lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        optimizer = configure_optimizer(
            optimizer_name=optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            model_params=grouped_params,
        )

        warmup_steps = int(self.max_steps / self.args.num_epochs * self.warmup_epoch)

        if self.scheduler_name == ArgLiteral.SchedulerName.CONSTANT:
            base_scheduler = configure_constant_lambda(warmup_steps=warmup_steps)
        elif self.scheduler_name == ArgLiteral.SchedulerName.COSINE_DECAY:
            base_scheduler = configure_warmup_cosine_decay_lambda(self.max_steps, warmup_steps)

        scheduler = LambdaLR(
            optimizer,
            [base_scheduler for _ in range(len(optimizer.param_groups))],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_callbacks(self):

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_acc",
                mode="max",
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_acc:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def training_step(self, batch, batch_idx):

        pixel_values = batch["video"]
        labels = batch["label"]

        labels = torch.tensor(labels).to(pixel_values.device)

        outputs = self.model(pixel_values)
        loss = outputs.loss

        return loss

    def on_validation_start(self) -> None:
        super().on_validation_start()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        video = batch["video"]

        assert video.shape[1] == 1, "don't use multiview. set --clips_per_video to 1."
        video = video.squeeze(1)

        outputs = self.model(video)
        loss = outputs.loss
        log_dict = {
            "val_loss": loss.item(),
            "val_acc": outputs.acc.item() if hasattr(outputs, "acc") else None,
            "val_ent_st_mse": outputs.ent_st_mse.item() if hasattr(outputs, "ent_st_mse") else None,
            "val_ent_s_mse": outputs.ent_s_mse.item() if hasattr(outputs, "ent_s_mse") else None,
        }
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    exp_name = datetime.now().strftime("%m%d%H%M")

    data_module = TrainValDataModule(
        command_line_args=args,
    )
    max_steps = len(data_module.train_dataloader()) * args.num_epochs // args.grad_accum

    model_lightning = AttnDistillLightningModel(
        command_line_args=args,
        exp_name=exp_name,
        max_steps=max_steps,
    )

    callbacks = configure_callbacks()

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        # strategy="ddp",
        strategy="ddp_find_unused_parameters_true",
        max_epochs=args.num_epochs,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        precision="bf16-true",  # for BF16 training, use with caution for nan/inf
        limit_val_batches=10,
        limit_train_batches=10,
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=args.checkpoint_to_resume,
    )


if __name__ == "__main__":
    main()
