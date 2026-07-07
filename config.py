"""
Configuration for program
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# VoiceBank-DEMAND-16k dataset root. Defaults to ~/VoiceBank-DEMAND-16k
# (matches /home/harleen_ece/VoiceBank-DEMAND-16k on the training server);
# override with the VOICEBANK_DEMAND_ROOT env var if it lives elsewhere.
# Expected structure: clean_trainset_28spk_wav/, noisy_trainset_28spk_wav/,
# clean_testset_wav/, noisy_testset_wav/, train.scp, test.scp
DATASET_ROOT = os.environ.get("VOICEBANK_DEMAND_ROOT", os.path.expanduser("~/VoiceBank-DEMAND-16k"))

# train.scp has no dedicated validation split, so dataloader.py
# deterministically holds out VAL_FRACTION of it (fixed SPLIT_SEED) for
# per-epoch validation - test.scp is never touched for this.
VAL_FRACTION = 0.1
SPLIT_SEED = 42

#distillation
teacher = 'DCCRN'
student = 'DCCRN'
dataset = 'dns_challenge'
# Unused (distill.py loads the teacher from a pretrained HF Hub checkpoint
# instead - see DCCRNet.from_pretrained(...)); kept portable in case it's
# ever wired back in.
teacher_weight_path = os.path.join(PROJECT_ROOT, 'checkpoint_teacher', 'chkpt_100.pt')

lr_decay_rate = 0.1,
weight_decay = 5e-4,

########################### TEACHER ###########################
# model
mode = 'DCCRN'  # DCUNET / DCCRN
info = 'MODEL INFORMATION : IT IS USED FOR FILE NAME'

test = True

# model information
fs = 16000
win_len = 400
win_inc = 100
ola_ratio = win_inc / win_len
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)
window_type = 'hamming'

rnn_layers = 2
rnn_units = 256
masking_mode = 'E'
use_clstm = True
kernel_num = [32, 64, 128, 256, 256, 256]  # DCCRN
#kernel_num = [72, 72, 144, 144, 144, 160, 160, 180]  # DCUNET
loss_mode = 'SDR+PMSQE'

# hyperparameters for model train
max_epochs = 20
learning_rate = 0.0006
batch = 32  # matches the CLSKD paper (Cheng et al. 2022)'s actual experimental
# setup - a safe, literature-backed starting point now that we're on a
# dedicated H100 NVL (94GB) rather than the 6GB RTX 3050 laptop GPU this
# used to be tuned for (batch=4, since batch=8 caused VRAM pressure/slowdown
# and batch=12 crashed on that card). Given the H100's headroom this can
# likely go higher than 32 - raise and watch VRAM/throughput if you want to
# push further.


########################### STUDENT ###########################
rnn_layers_student = 2
rnn_units_student = 32  # paper (Cheng et al. 2022, CLSKD): student complex-LSTM has 32 units vs teacher's 128
kernel_num_student = [8, 16, 32, 64, 64, 64]  # DCCRN