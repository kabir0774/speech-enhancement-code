import glob
import os
import random
import torch
import numpy as np
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
    # NOTE: num_workers>0 + persistent_workers was tried for speed on the
    # original Windows dev machine and reverted, because
    # train_dataloader()/val_dataloader() in distill.py construct a fresh
    # VoiceBankDataset + DataLoader each time Lightning calls them, and
    # persistent worker processes from the previous DataLoader weren't being
    # cleanly replaced there - they piled up across epochs (12 python.exe
    # processes observed instead of ~5), causing severe CPU/GPU contention
    # and epoch times to balloon from ~1min to ~7min. That failure mode was
    # specifically diagnosed as Windows' spawn-based multiprocessing (each
    # spawned worker re-imports the module instead of forking the parent).
    # We're now on Linux, which uses fork-based multiprocessing and doesn't
    # have this failure mode, so num_workers>0 + persistent_workers=True is
    # safe here. If this ever runs on Windows again, watch process count
    # (e.g. `ps`/Task Manager) across several epochs - not just the first -
    # before trusting it, same as the original incident.
    if mode == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch,  # max 3696 * snr types
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True
        )    # max 1152

def create_dataloader_for_test(mode, type, snr):
    if mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 192


def _read_scp(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _train_val_ids(train_scp_path):
    """Deterministically hold out cfg.VAL_FRACTION of train.scp (fixed
    cfg.SPLIT_SEED) for validation, since VoiceBank-DEMAND-16k only ships
    train.scp/test.scp - no separate dev split. test.scp is never touched by
    this, so it stays a true held-out test set."""
    ids = sorted(_read_scp(train_scp_path))
    rng = random.Random(cfg.SPLIT_SEED)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * cfg.VAL_FRACTION))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


class VoiceBankDataset(Dataset):
    """VoiceBank-DEMAND speech enhancement dataset.

    Single-speaker denoising: each utterance ID listed in train.scp/test.scp
    (under cfg.DATASET_ROOT) maps to exactly one clean/noisy wav pair -
    {clean_dir}/{id}.wav and {noisy_dir}/{id}.wav - no mixture manifest.

    split:
      'train' - train.scp minus the held-out validation utterances
      'val'   - the held-out validation utterances from train.scp
      'test'  - test.scp, untouched
    """

    def __init__(self, split='train', sample_rate=16000, segment=3, return_id=False):
        assert split in ('train', 'val', 'test')
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.return_id = return_id

        if split == 'test':
            self.clean_dir = os.path.join(cfg.DATASET_ROOT, 'clean_testset_wav')
            self.noisy_dir = os.path.join(cfg.DATASET_ROOT, 'noisy_testset_wav')
            ids = _read_scp(os.path.join(cfg.DATASET_ROOT, 'test.scp'))
        else:
            self.clean_dir = os.path.join(cfg.DATASET_ROOT, 'clean_trainset_28spk_wav')
            self.noisy_dir = os.path.join(cfg.DATASET_ROOT, 'noisy_trainset_28spk_wav')
            train_ids, val_ids = _train_val_ids(os.path.join(cfg.DATASET_ROOT, 'train.scp'))
            ids = train_ids if split == 'train' else val_ids

        if self.segment is not None:
            self.seg_len = int(self.segment * self.sample_rate)
            self._n_frames = {}
            kept, dropped = [], 0
            for uid in ids:
                try:
                    n_frames = sf.info(os.path.join(self.noisy_dir, uid + '.wav')).frames
                except RuntimeError:
                    dropped += 1
                    continue
                if n_frames >= self.seg_len:
                    kept.append(uid)
                    self._n_frames[uid] = n_frames
                else:
                    dropped += 1
            print(
                f"Drop {dropped} utterances from {len(ids)} ({split}, "
                f"shorter than {segment} seconds or missing)"
            )
            self.ids = kept
        else:
            self.seg_len = None
            self.ids = ids

        # informational only - set on each __getitem__, read by
        # validation_step()'s getattr(self.val_dataset, "mixture_path", None)
        self.mixture_path = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        noisy_path = os.path.join(self.noisy_dir, uid + '.wav')
        clean_path = os.path.join(self.clean_dir, uid + '.wav')
        self.mixture_path = noisy_path

        if self.seg_len is not None:
            n_frames = self._n_frames[uid]
            start = random.randint(0, max(0, n_frames - self.seg_len))
            stop = start + self.seg_len
        else:
            start, stop = 0, None

        noisy, _ = sf.read(noisy_path, dtype="float32", start=start, stop=stop)
        clean, _ = sf.read(clean_path, dtype="float32", start=start, stop=stop)

        noisy = torch.from_numpy(noisy)
        # [1, T] to match the n_src=1 "sources" shape convention the
        # training/validation code expects (e.g. y.squeeze(1) in distill.py).
        clean = torch.from_numpy(clean[None, :])

        if not self.return_id:
            return noisy, clean
        return noisy, clean, [uid]


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
