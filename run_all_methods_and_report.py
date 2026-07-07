"""
Runs all 6 distillation methods (None, MSE, STFT, SPKD, ReviewKD, CLSKD)
sequentially on the FULL VoiceBank-DEMAND-16k dataset (train.scp/test.scp
under config.DATASET_ROOT, ~11,572 training utterances with a held-out
validation slice per config.VAL_FRACTION - see dataloader.py's
VoiceBankDataset and config.py), then parses each run's PyTorch Lightning
CSV logs (lightning_logs/version_N/metrics.csv, written automatically by
Lightning's default CSVLogger) to build the same WB-PESQ/STOI-per-epoch
comparison chart + summary table used throughout this project (see
output/dataset_size_progress.png, output/before_after_original_5files.png
for the same visual format on smaller subsets).

DO NOT run this on a small/laptop GPU. On a 6GB RTX 3050, the 900-file
subset took ~30-50 minutes per method for 20 epochs; the full ~10,700-file
dataset is ~12x more data, so all 6 methods sequentially is realistically
1.5-3 days of continuous training. This script is meant to be run on a
bigger GPU (e.g. a datacenter card with 24GB+ VRAM) - see the notes at the
bottom of this file for config.py adjustments to make first.

RESUME SUPPORT (two layers):
  1. Each individual distill_*.py script resumes from its own last saved
     Lightning checkpoint (via find_latest_checkpoint() in dataloader.py)
     if it was interrupted mid-training - so re-running a method that got
     to epoch 12/20 continues from epoch 13, not from scratch.
  2. This orchestration script tracks which methods have FULLY finished
     (their final "*_best_model.pth" exists) in output/run_all_state.json,
     and skips re-running those entirely on a subsequent invocation - so
     if the whole pipeline is interrupted after finishing e.g. 3 of 6
     methods, re-running this script only trains the remaining ones.

Usage:
    python run_all_methods_and_report.py           # resume from wherever it left off
    python run_all_methods_and_report.py --fresh    # ignore state file, re-run all 6
                                                     # (each script may still resume from
                                                     # its own checkpoint dir unless you
                                                     # also delete checkpoint*/ yourself)
"""
import subprocess
import sys
import os
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # use whichever python is running this orchestration script

# (display name, script, final checkpoint file it writes when fully done, color)
METHODS = [
    ("None (no distillation)", "distill_None.py",     "./checkpoint_None/None_best_model.pth",         "#2a78d6"),
    ("MSE (Diff-style)",       "distill_MSE.py",       "./checkpoint_MSE/MSE_best_model.pth",           "#1baf7a"),
    ("STFT-output distill",    "distill_STFT.py",      "./checkpoint_STFT/STFT_best_model.pth",         "#c9970d"),
    ("SPKD",                   "distill_SPKD.py",      "./checkpoint_SPKD/SPKD_best_model.pth",         "#008300"),
    ("ReviewKD",                "distill_ReviewKD.py",  "./checkpoint_ReviewKD/best_model_ReviewKD.pth", "#4a3aa7"),
    ("CLSKD (full method)",    "distill.py",           "./checkpoint/the_best_model.pth",               "#e34948"),
]

LIGHTNING_LOGS = os.path.join(CODE_DIR, "lightning_logs")
OUT_DIR = os.path.join(CODE_DIR, "output")
STATE_PATH = os.path.join(OUT_DIR, "run_all_state.json")


def existing_version_dirs():
    if not os.path.isdir(LIGHTNING_LOGS):
        return set()
    return set(os.listdir(LIGHTNING_LOGS))


def load_state():
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def run_method(script_name):
    before = existing_version_dirs()
    print(f"\n{'=' * 70}\nRunning {script_name}\n{'=' * 70}", flush=True)
    result = subprocess.run([PYTHON, script_name], cwd=CODE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    after = existing_version_dirs()
    new_dirs = after - before
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 new lightning_logs version dir for {script_name}, "
            f"got {new_dirs} (before={len(before)}, after={len(after)})"
        )
    return os.path.join(LIGHTNING_LOGS, new_dirs.pop(), "metrics.csv")


def load_epoch_curve(metrics_csv):
    """Lightning logs both on_step and on_epoch rows to the same CSV; keep
    only the epoch-level validation rows and collapse to one row/epoch."""
    df = pd.read_csv(metrics_csv)
    val_cols = [c for c in ["pesq_epoch", "stoi_epoch", "si_sdr_epoch"] if c in df.columns]
    if not val_cols:
        raise RuntimeError(f"No expected validation columns found in {metrics_csv}")
    df = df.dropna(subset=val_cols, how="all")
    df = df.groupby("epoch", as_index=False).last()
    return df[["epoch"] + val_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore output/run_all_state.json and re-run all 6 methods "
                              "from the orchestrator's point of view (each script may still "
                              "auto-resume from its own checkpoint dir - delete checkpoint*/ "
                              "folders yourself for a truly clean restart).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    state = {} if args.fresh else load_state()
    curves = {}

    for name, script, final_ckpt, _color in METHODS:
        final_ckpt_path = os.path.join(CODE_DIR, final_ckpt)
        cached = state.get(name)
        already_done = (
            cached is not None
            and os.path.isfile(final_ckpt_path)
            and os.path.isfile(cached["metrics_csv"])
        )

        if already_done:
            print(f"\n{name}: already completed (found {final_ckpt}) - skipping, reusing logged metrics", flush=True)
            metrics_csv = cached["metrics_csv"]
        else:
            metrics_csv = run_method(script)
            if not os.path.isfile(final_ckpt_path):
                raise RuntimeError(
                    f"{script} exited successfully but {final_ckpt} was not found - "
                    f"training may not have actually completed."
                )
            state[name] = {"metrics_csv": metrics_csv}
            save_state(state)  # persist progress after every method, not just at the end

        curves[name] = load_epoch_curve(metrics_csv)
        print(f"{name}: {len(curves[name])} epochs logged -> {metrics_csv}", flush=True)

    # ---- chart: PESQ + STOI per epoch, all 6 methods ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=150)
    fig.suptitle("CLSKD distillation methods — comparison (full VCTK-DEMAND dataset)",
                 fontsize=15, fontweight="bold")

    for ax, col, title, scale in [
        (axes[0], "pesq_epoch", "WB-PESQ per epoch (validation)", 1),
        (axes[1], "stoi_epoch", "STOI per epoch (validation) %", 100),
    ]:
        for name, _script, _final_ckpt, color in METHODS:
            df = curves[name]
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col] * scale, label=name, color=color, linewidth=2)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("Epoch")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    chart_path = os.path.join(OUT_DIR, "full_dataset_results.png")
    fig.savefig(chart_path, bbox_inches="tight", facecolor="white")

    # ---- summary table (final + best epoch, matching Table 1's format) ----
    rows = []
    for name, _script, _final_ckpt, _color in METHODS:
        df = curves[name]
        row = {"method": name}
        if "pesq_epoch" in df.columns:
            row["final_pesq"] = round(float(df["pesq_epoch"].iloc[-1]), 3)
            row["best_pesq"] = round(float(df["pesq_epoch"].max()), 3)
        if "stoi_epoch" in df.columns:
            row["final_stoi_pct"] = round(float(df["stoi_epoch"].iloc[-1] * 100), 2)
            row["best_stoi_pct"] = round(float(df["stoi_epoch"].max() * 100), 2)
        if "si_sdr_epoch" in df.columns:
            row["final_si_sdr"] = round(float(df["si_sdr_epoch"].iloc[-1]), 3)
            row["best_si_sdr"] = round(float(df["si_sdr_epoch"].max()), 3)
        rows.append(row)

    table_path = os.path.join(OUT_DIR, "full_dataset_results_table.json")
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nDone. Results saved to:")
    print(" -", chart_path)
    print(" -", table_path)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# Notes for running this on a bigger GPU (e.g. datacenter-class, 24GB+ VRAM):
#
# 1. config.py's `batch = 4` was set for a 6GB laptop GPU (RTX 3050), which
#    overflowed at batch=12 and got slow/unstable at batch=8. On a GPU with
#    real headroom, raise this - the original paper (Cheng et al. 2022,
#    CLSKD) uses batch=32; that also gives the SPKD/similarity-based
#    distillation losses a much richer signal (32x32 pairwise similarity
#    matrices instead of 4x4 - see the batch-size discussion earlier in
#    this project's history).
#
# 2. config.py's `max_epochs = 20` matches the paper. Keep as-is unless you
#    want a faster first pass (e.g. 5-10 epochs) before committing to a
#    full run across all 6 methods.
#
# 3. dataloader.py currently uses `num_workers=0` (synchronous data
#    loading) - deliberately, because `num_workers>0` combined with
#    `persistent_workers=True` caused worker processes to pile up across
#    epochs on this Windows machine (train_dataloader()/val_dataloader()
#    in each distill_*.py script construct a fresh DataLoader every time
#    Lightning calls them, and the old persistent workers weren't cleanly
#    replaced - see the "epoch times ballooned from ~1min to ~7min"
#    incident earlier in this project). If you try num_workers>0 on a
#    different machine, watch `nvidia-smi`/process count across several
#    epochs (not just the first one) before trusting it.
#
# 4. Do NOT enable PyTorch Lightning's `precision="16-mixed"` (mixed
#    precision) - DCCRN's complex-valued convolutions use PyTorch's
#    ComplexHalf type, which is experimental and produced NaN losses
#    partway through a real training run here. Stay at full (32-bit)
#    precision.
# -----------------------------------------------------------------------
