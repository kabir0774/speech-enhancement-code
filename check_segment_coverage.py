"""
Quick dataset-coverage check: how many train.scp utterances survive
VoiceBankDataset's segment-length filter, at segment=2 (current) vs the old
segment=3, for both the train split and the held-out validation split.

Just constructs the dataset (reads wav headers via soundfile.info, not full
audio) - no model, no GPU, no training. Safe to run before committing to a
real training run.

Usage (from code_only/):
    python check_segment_coverage.py
"""
import os

from dataloader import VoiceBankDataset, _train_val_ids
import config as cfg

train_ids, val_ids = _train_val_ids(os.path.join(cfg.DATASET_ROOT, "train.scp"))
print(f"train.scp total: {len(train_ids) + len(val_ids)}  "
      f"(train split: {len(train_ids)}, held-out val split: {len(val_ids)})\n")

for seg in (3, 2):
    print(f"=== segment={seg} ===")
    train_ds = VoiceBankDataset(split='train', sample_rate=16000, segment=seg)
    val_ds = VoiceBankDataset(split='val', sample_rate=16000, segment=seg)
    print(f"  train kept: {len(train_ds)}/{len(train_ids)} "
          f"({100 * len(train_ds) / len(train_ids):.1f}%)")
    print(f"  val kept:   {len(val_ds)}/{len(val_ids)} "
          f"({100 * len(val_ds) / len(val_ids):.1f}%)")
    print()
