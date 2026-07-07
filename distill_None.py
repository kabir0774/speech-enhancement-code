from typing import Any
import numpy as np
import pandas as pd
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from asteroid.models import DCCRNet
from DCCRNet_mini import DCCRNet_mini
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.metrics import MockWERTracker

import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
from dataloader import create_dataloader, find_latest_checkpoint, VoiceBankDataset
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import get_metrics

from framework import MultiResolutionSTFTLoss
import config as cfg

COMPUTE_METRICS = ["si_sdr", "stoi", "pesq"]


class KnowledgeDistillation(pl.LightningModule):
    """
    Undistilled baseline (Table 1's "None" row): student trained with only
    the backbone MRSTFT reconstruction loss, no teacher signal at all.
    """

    def __init__(self, teacher, student, sftf_loss, cfg):
        super().__init__()
        self.automatic_optimization = True

        # teacher is unused (kept only for interface symmetry with the
        # distillation variants); no gradient ever touches it
        self.teacher = teacher
        for paras in self.teacher.parameters():
            paras.requires_grad = False

        self.student = student
        self.stft_loss = sftf_loss(fft_sizes=[512], win_lengths=[400], hop_sizes=[100])
        self.sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        student_preds = self.student(X)
        base_loss = self.stft_loss(student_preds.squeeze(1), y.squeeze(1))[1]
        loss = base_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        wer_tracker = (MockWERTracker())
        model_device = next(self.student.parameters()).device
        series_list = []

        for idx in range(len(x)):
            mix = x[idx]
            sources = y[idx]
            mix, sources = tensors_to_device([mix, sources], device=model_device)
            est_sources = self.student(mix.unsqueeze(0))
            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=16000,
                metrics_list=COMPUTE_METRICS)
            utt_metrics["mix_path"] = getattr(self.val_dataset, "mixture_path", None)
            est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
            utt_metrics.update(
                **wer_tracker(
                    mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np_normalized,
                    sample_rate=16000,
                )
            )
            series_list.append(pd.Series(utt_metrics))

        all_metrics_df = pd.DataFrame(series_list)
        final_results = {}
        for metric_name in COMPUTE_METRICS:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()

        self.log_dict(final_results, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
        return optimizer

    def train_dataloader(self):
        train_dataset = VoiceBankDataset(
            csv_dir='./data/wav16k/max/train-360',
            task='enh_single',
            sample_rate=16000,
            n_src=1,
            segment=3,
        )
        train_loader = create_dataloader(mode='train', dataset=train_dataset)
        return train_loader

    def val_dataloader(self):
        val_dataset = VoiceBankDataset(
            csv_dir='./data/wav16k/max/dev',
            task='enh_single',
            sample_rate=16000,
            n_src=1,
            segment=3,
        )
        self.val_dataset = val_dataset
        val_loader = create_dataloader(mode='valid', dataset=val_dataset)
        return val_loader


torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    with open("./conf.yml") as f:
        def_conf = yaml.safe_load(f)
        parser = prepare_parser_from_dict(def_conf, parser=parser)
    conf, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(conf)

    teacher = DCCRNet.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    student = DCCRNet_mini(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"])

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoint_None',
        filename='model-{epoch:02d}{stoi:.4f}',
        save_top_k=1,
        monitor='stoi',
        mode='max',
        verbose=True)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        default_root_dir='.',
                        callbacks=[checkpoint_callback]
                        )

    kd_module = KnowledgeDistillation(teacher,
                                    student,
                                    sftf_loss=MultiResolutionSTFTLoss,
                                    cfg=cfg)

    resume_ckpt = find_latest_checkpoint('./checkpoint_None')
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.fit(kd_module, ckpt_path=resume_ckpt)

    state_dict = torch.load(checkpoint_callback.best_model_path)
    only_student_state_dict = {}
    for key, value in state_dict['state_dict'].items():
        if key.startswith('student.'):
            only_student_state_dict[key.replace('student.', '')] = value
        else:
            continue
    state_dict['state_dict'] = only_student_state_dict

    student.load_state_dict(state_dict=state_dict["state_dict"])
    student.cpu()

    to_save = student.serialize()
    torch.save(to_save, os.path.join('./checkpoint_None', "None_best_model.pth"))
