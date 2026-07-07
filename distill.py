from typing import Any
import numpy as np
import pandas as pd
import os
import argparse
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pytorch_lightning as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from asteroid.models import DCCRNet
from DCCRNet_mini import DCCRNet_mini
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.metrics import MockWERTracker

import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
from asteroid.metrics import get_metrics
from dataloader import create_dataloader, find_latest_checkpoint, VoiceBankDataset
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from tools_for_model import cal_pesq, cal_stoi
from torch_stoi import NegSTOILoss


from framework import MultiResolutionSTFTLoss, SPKDLoss, build_review_kd
import feature_extraction
import config as cfg

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"
COMPUTE_METRICS = ["si_sdr","stoi","pesq"]


class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, sftf_loss, spkd_loss, cfg):
        super().__init__()
        self.automatic_optimization = True

        #load teacher (pre-trained)
        self.teacher = teacher
        #self.teacher_checkpoint = torch.load(cfg.teacher_weight_path)
        #self.teacher.load_state_dict(self.teacher_checkpoint['model'])

        #freeze teacher
        for paras in self.teacher.parameters():
            paras.requires_grad = False

        #load student model
        self.student = student

        #SPKD loss
        self.spkd_loss = spkd_loss
    
        #base loss - MRSFTF loss
        self.stft_loss = sftf_loss(fft_sizes=[512], win_lengths=[400],hop_sizes=[100])
        #self.stft_loss = sftf_loss(fft_sizes=[cfg.fft_len], win_lengths=[cfg.win_len],hop_sizes=[cfg.win_inc])

        # ReviewKD fusion modules (encoder/decoder). Built ONCE here (not per
        # training step) so their conv/batchnorm weights persist and actually
        # train across steps, and so Lightning moves them to the right device
        # automatically. ReviewKD.forward() ignores its `x` argument and reads
        # from `self.feature_maps` instead, so each step just needs to update
        # that attribute before calling the module - see training_step.
        self.review_kd_encoder = build_review_kd([], 'encoder')
        self.review_kd_decoder = build_review_kd([], 'decoder')

        # Feature-extraction hook wrappers, built ONCE (not per training
        # step). extract_feature_maps() now clears its own accumulator
        # lists on every call (see feature_extraction.py), so the same
        # wrapper - and its registered hooks - can safely be reused across
        # the whole training run instead of registering/removing hooks on
        # every single step.
        self.teacher_extraction = feature_extraction.DCCRNet(self.teacher)
        self.student_extraction = feature_extraction.DCCRNet(self.student)

        #
        self.mixture_path = None

        #val_loss function
        # self.sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        # self.stoi = PITLossWrapper(NegSTOILoss(sample_rate=16000), pit_from='pw_pt')

    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
       
        # getting teacher features (reusing the wrappers built once in
        # __init__, instead of registering/removing hooks every step)
        teacher_extraction = self.teacher_extraction
        teacher_features = teacher_extraction.extract_feature_maps(X)
        teacher_encoder, teacher_decoder, teacher_clstm_real, teacher_clstm_img = (teacher_features["encoder"],
                                                                                   teacher_features["decoder"],
                                                                                   teacher_features["clstm_real"][0],
                                                                                   teacher_features["clstm_img"][0])

        # getting student features
        student_extraction = self.student_extraction
        student_features = student_extraction.extract_feature_maps(X)
        student_encoder,student_decoder, student_clstm_real,student_clstm_img = (student_features["encoder"],
                                                                                 student_features["decoder"],
                                                                                 student_features["clstm_real"][0],
                                                                                 student_features["clstm_img"][0])
        
        # feature fusion (review kd) for student's encoder and decoder
        self.review_kd_encoder.feature_maps = student_encoder
        student_features_encoder = self.review_kd_encoder(X)

        self.review_kd_decoder.feature_maps = student_decoder
        student_features_decoder = self.review_kd_decoder(X)

        
        # calculating based-loss (Multi-resolution STFT)
        # reuse the forward pass already run inside student_extraction above
        # instead of calling self.student(X) again (was a redundant, full
        # second forward pass through the student every single step)
        student_preds = student_extraction.last_output
        base_loss = self.stft_loss(student_preds.squeeze(1),y.squeeze(1))[1]
        

        feature_maps_loss = {'encoder':0.0,'decoder':0.0,'clstm_real':0.0,'clstm_img':0.0}

        ############## ENCODER loss ######################
        loss = 0.0
        # calculating review kd loss
        for sf, tf in zip(student_features_encoder,teacher_encoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            loss += kd_loss()
        #save loss for each feature map:
        feature_maps_loss['encoder'] = loss


        ############## DECODER loss ######################
        loss = 0.0
        # calculating review kd loss
        for sf, tf in zip(student_features_decoder,teacher_decoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            loss += kd_loss()
        #save loss for each feature map:
        feature_maps_loss['decoder'] = loss


        ############## C-LSTM REAL loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_real, teacher_clstm_real,reduction='batchmean')
        feature_maps_loss['clstm_real'] = kd_loss()
        

        ############## C-LSTM IMAGE loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_img, teacher_clstm_img,reduction='batchmean')
        feature_maps_loss['clstm_img'] = kd_loss()
        

        ############## All losses: baseloss + [Encoder + Decoder + C-Lstm_real + C-Lstm_img] ##################
        loss = base_loss + sum(feature_maps_loss.values())

        # logging training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # NOTE: no more remove_hook() here - teacher_extraction/student_extraction
        # are now built once in __init__ and reused for the whole training run
        # (see feature_extraction.py: extract_feature_maps() clears its own
        # accumulator lists each call), rather than registering fresh hooks and
        # tearing them down every single step.

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y = batch
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        wer_tracker = (MockWERTracker())
        model_device = next(self.student.parameters()).device
        #print(self.val_dataset.mixture_path)
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
            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=16000,
                metrics_list=COMPUTE_METRICS)
            # With num_workers>0, each DataLoader worker gets its own copy of
            # val_dataset, so this main-process object never has
            # `mixture_path` set by __getitem__'s side effect. Informational
            # only (not used in any metric computation), so fall back safely.
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
        # Print and save summary metrics
        final_results = {}
        for metric_name in COMPUTE_METRICS:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()

        # print("Overall metrics :")
        # print(final_results)

        # logging metrics
        self.log_dict(final_results, on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
    def configure_optimizers(self):
        params = list(self.student.parameters()) + \
                  list(self.review_kd_encoder.parameters()) + \
                  list(self.review_kd_decoder.parameters())
        optimizer = optim.Adam(params, lr=cfg.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        train_dataset = VoiceBankDataset(
            csv_dir='./data/wav16k/max/train-360',
            task='enh_single',
            sample_rate=16000,
            n_src=1,
            segment=3,
        )
        train_loader = create_dataloader(mode='train',dataset=train_dataset)
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
        val_loader = create_dataloader(mode='valid',dataset=val_dataset)
        return val_loader



# setup float type
torch.set_float32_matmul_precision('high')

# Guard the training script body so DataLoader num_workers>0 (separate
# worker processes) doesn't re-execute this whole file on Windows, which
# uses process "spawn" and re-imports the main module in each worker.
if __name__ == "__main__":
    # read config file
    parser = argparse.ArgumentParser()
    with open("./conf.yml") as f:
        def_conf = yaml.safe_load(f)
        parser = prepare_parser_from_dict(def_conf, parser=parser)
    conf, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(conf)

    # initialize models
    teacher =  DCCRNet.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    student =  DCCRNet_mini(
            **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"])



    # initalize checkpoint
    checkpoint_callback = ModelCheckpoint(
                        dirpath='./checkpoint',
                        filename='model-{epoch:02d}-{stoi:.4f}-{si_sdr:.3f}',
                        save_top_k=3,
                        monitor='stoi',
                        mode='max',
                        verbose=True)


    # initialize trainer
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        # NOTE: precision="16-mixed" was tried for speed, but DCCRN's
                        # complex-valued convolutions use PyTorch's ComplexHalf, which
                        # is explicitly experimental/numerically unstable - it produced
                        # NaN losses partway through training. Staying at full (32-bit)
                        # precision; num_workers>0 in dataloader.py alone still gives a
                        # real, safe speedup.
                        default_root_dir='.',
                        callbacks=[checkpoint_callback]
                        )

    # initialize knowledge distillation module
    kd_module = KnowledgeDistillation(teacher,
                                    student,
                                    sftf_loss=MultiResolutionSTFTLoss,
                                    spkd_loss=SPKDLoss,
                                    cfg=cfg)

    # train the student network using knowledge distillation - resume from
    # the last checkpoint if a previous run of this script was interrupted
    resume_ckpt = find_latest_checkpoint('./checkpoint')
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.fit(kd_module, ckpt_path=resume_ckpt)

    # load best student model
    state_dict = torch.load(checkpoint_callback.best_model_path)
    only_student_state_dict = {}
    for key,value in state_dict['state_dict'].items():
        if key.startswith('student.'):
            only_student_state_dict[key.replace('student.','')] = value
        else:
            continue
    state_dict['state_dict'] = only_student_state_dict

    student.load_state_dict(state_dict=state_dict["state_dict"])
    student.cpu()

    # save the best student model
    to_save = student.serialize()
    torch.save(to_save,os.path.join('./checkpoint', "the_best_model.pth"))




