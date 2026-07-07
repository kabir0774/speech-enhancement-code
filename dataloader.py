import glob
import os
import random
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import config as cfg


# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def find_latest_checkpoint(dirpath):
    """
    Look for a Lightning `.ckpt` file (NOT the final `*_best_model.pth`,
    which is a different, post-processed, student-only file written only
    after training fully completes) in `dirpath`, and return the most
    recently modified one, or None if there isn't one.

    Used so each distill_*.py script can resume training exactly where it
    left off (via `trainer.fit(model, ckpt_path=...)`) if a previous run
    was interrupted partway through - instead of restarting at epoch 0.
    """
    if not os.path.isdir(dirpath):
        return None
    ckpts = glob.glob(os.path.join(dirpath, "*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def create_dataloader(mode,dataset):
    # NOTE: num_workers>0 + persistent_workers was tried for speed, but
    # train_dataloader()/val_dataloader() in distill.py construct a fresh
    # VoiceBankDataset + DataLoader each time Lightning calls them, so
    # persistent worker processes from the previous DataLoader weren't being
    # cleanly replaced - they piled up across epochs (12 python.exe processes
    # observed instead of ~5), causing severe CPU/GPU contention and
    # epoch times to balloon instead of shrink. num_workers=0 is slower
    # per-epoch but stable and predictable.
    if mode == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch,  # max 3696 * snr types
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch, shuffle=False, num_workers=0,drop_last=True
        )    # max 1152

def create_dataloader_for_test(mode, type, snr):
    if mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 192


class VoiceBankDataset(Dataset):
    """VoiceBank-DEMAND speech enhancement dataset.

    Reads a mixture/source metadata CSV (columns: mixture_path,
    source_1_path, length) such as the ones under ./data/wav16k/max, which
    are generated from the local VCTK_DEMAND noisy/clean wav pairs. This is
    a self-contained replacement for asteroid.data.LibriMix (same csv-driven
    interface) so training/eval never depends on the LibriMix dataset class
    or its download helpers - this project only ever trains/evaluates on
    VoiceBank-DEMAND.
    """

    def __init__(self, csv_dir, task='enh_single', sample_rate=16000, n_src=1, segment=3, return_id=False):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
        self.csv_path = os.path.join(csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        self.df = pd.read_csv(self.csv_path)
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src
        self.mixture_path = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path

        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None

        sources_list = []
        for i in range(self.n_src):
            source_path = row[f"source_{i + 1}_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        mixture = torch.from_numpy(mixture)
        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources)

        if not self.return_id:
            return mixture, sources
        # e.g. p286_001.wav -> ["p286", "001"]; os.path.basename() (not
        # split("/")) so this works with Windows paths too.
        id1, id2 = os.path.basename(mixture_path).split(".")[0].split("_")
        return mixture, sources, [id1, id2]


class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            self.input_path = './input/train_dataset.npy'
        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            self.input_path = './input/validation_dataset.npy'

        self.input = np.load(self.input_path)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        labels = self.input[idx][1]

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels


class Wave_Dataset_for_test(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        if mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/recon_test_dataset.npy'

        self.input = np.load(self.input_path)
        self.input = self.input[type][snr]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        labels = self.input[idx][1]

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels
