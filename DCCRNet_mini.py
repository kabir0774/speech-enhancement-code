"""
DCCRNet_mini

The original repo imported `DCCRNet_mini` from `asteroid.models`, but this
class does not exist in the public asteroid package (only `DCCRNet` does).
It was almost certainly a small addition the original author made to a
private local copy of asteroid, which was never published alongside the repo.

This module recreates it properly: it builds a smaller "student" architecture
preset using the channel sizes already specified in config.py
(`kernel_num_student`), following the exact same encoder/decoder layout
convention asteroid's own "DCCRN-CL" architecture uses. It then subclasses
the REAL asteroid.models.DCCRNet with that preset baked in as the default.

Because this is a genuine DCCRNet subclass (not a separate hand-rolled
model), it automatically:
  - exposes `.masker.encoders` / `.masker.decoders`, matching what
    `feature_extraction.py`'s `DCCRNet` hook class expects
  - inherits working `.serialize()` / `.from_pretrained()` methods from
    asteroid's BaseModel, so eval.py's `DCCRNet_mini.from_pretrained(path)`
    and distill.py's `student.serialize()` work unmodified.
"""
import numpy as np
import torch
from asteroid.models import DCCRNet
from asteroid.masknn._dccrn_architectures import DCCRN_ARCHITECTURES
from asteroid.masknn.base import BaseDCUMaskNet
import asteroid.masknn.recurrent as _asteroid_recurrent
from asteroid.masknn.recurrent import DCCRMaskNet, DCCRMaskNetRNN
from asteroid import complex_nn

import config as cfg

# --- Compatibility patch -----------------------------------------------
# PyTorch 2.6+ made `torch.load` default to `weights_only=True`, which
# rejects the small `torch.torch_version.TorchVersion` metadata object
# that asteroid's own `serialize()`/`from_pretrained()` store in every
# checkpoint. This allowlists that one harmless class so checkpoints you
# created yourself with `model.serialize()` can be reloaded normally.
# (Only ever load checkpoint files you trust the origin of.)
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
# ------------------------------------------------------------------------

_STUDENT_ARCHITECTURE_NAME = "DCCRN-CL-test"


# --- Compatibility patch -----------------------------------------------
# asteroid==0.7.0's DCCRMaskNet computes its RNN input size with
# `np.prod(...)`, which returns a numpy.int64 rather than a plain Python
# int. Newer versions of PyTorch's nn.LSTM reject that with:
#   TypeError: input_size should be of type int, got: int64
# This affects ANY DCCRNet architecture (including asteroid's own stock
# "DCCRN-CL"), not just the student model below, so it's patched here
# once, at import time, rather than worked around per-architecture.
if not getattr(_asteroid_recurrent.DCCRMaskNetRNN, "_int64_patch_applied", False):
    _orig_rnn_init = _asteroid_recurrent.DCCRMaskNetRNN.__init__

    def _patched_rnn_init(self, in_size, *args, **kwargs):
        _orig_rnn_init(self, int(in_size), *args, **kwargs)

    _asteroid_recurrent.DCCRMaskNetRNN.__init__ = _patched_rnn_init
    _asteroid_recurrent.DCCRMaskNetRNN._int64_patch_applied = True
# ------------------------------------------------------------------------


def _build_student_architecture(kernel_num):
    """
    Build an (encoders, decoders) tuple pair in the same format as
    asteroid's built-in "DCCRN-CL" architecture, scaled to a given
    channel plan (e.g. config.kernel_num_student).

    encoders: (in_chan, out_chan, kernel_size, stride, padding)
    decoders: (in_chan, out_chan, kernel_size, stride, padding, output_padding)
    """
    channels = [1] + list(kernel_num)  # 1 = complex-valued input channel

    encoders = tuple(
        (channels[i], channels[i + 1], (5, 2), (2, 1), (2, 0))
        for i in range(len(channels) - 1)
    )

    decoders = []
    # walk back up the U-Net; each decoder's input is doubled because it
    # concatenates with the matching encoder's skip connection
    for i in range(len(channels) - 1, 0, -1):
        in_chan = channels[i] * 2
        out_chan = channels[i - 1]
        decoders.append((in_chan, out_chan, (5, 2), (2, 1), (2, 0), (1, 0)))

    return encoders, tuple(decoders)


# Register the student architecture once, at import time, so
# DCCRNet(architecture="DCCRN-mini-student", ...) resolves correctly
# -- including later, inside `from_pretrained`, which needs to look the
# architecture name back up when reloading a saved checkpoint.
#
# NOTE: cfg.kernel_num_student ([8,16,32,64,64,64], as literally stated in
# the CLSKD paper) is the channel count AFTER feature_extraction.py's real
# +imag concatenation, not the raw per-layer conv width fed into the
# architecture itself - confirmed empirically: building the raw
# architecture at half of these values (and leaving the complex-LSTM at
# the paper's literal 32 units) lands at ~231K total params, matching the
# paper's reported 0.23M student almost exactly; using the values
# unhalved landed at 763K, ~3.3x too big. This is also why
# framework.py's build_review_kd() in_channels/out_channels use these
# cfg.kernel_num_student values directly (unhalved) - they describe the
# post-concat feature maps that build_review_kd actually consumes.
if _STUDENT_ARCHITECTURE_NAME not in DCCRN_ARCHITECTURES:
    DCCRN_ARCHITECTURES[_STUDENT_ARCHITECTURE_NAME] = _build_student_architecture(
        [c // 2 for c in cfg.kernel_num_student]
    )


class _SmallRNNDCCRMaskNet(DCCRMaskNet):
    """
    DCCRMaskNet, but with a configurable complex-LSTM hidden size.

    Asteroid's own DCCRMaskNet.__init__ hardcodes
    `DCCRMaskNetRNN(np.prod(last_encoder_out_shape))`, i.e. hid_size=128
    always, with no way to override it via the public constructor. The
    CLSKD paper (Cheng et al. 2022) specifies the student's complex-LSTM
    at 32 units vs. the teacher's 128 - reusing the hardcoded 128 for the
    student is why the earlier reconstruction came out at 1.4M params
    instead of the paper's reported 0.23M. This subclass re-implements
    the same construction, just threading `rnn_hid_size` through to the
    RNN layer.
    """

    def __init__(self, encoders, decoders, n_freqs, rnn_hid_size=128, **kwargs):
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _ in encoders], axis=0
        )
        freq_prod, _ = self.encoders_stride_product
        last_encoder_out_shape = (encoders[-1][1], int(np.ceil(n_freqs / freq_prod)))

        from asteroid.masknn.convolutional import (
            DCUNetComplexDecoderBlock,
            DCUNetComplexEncoderBlock,
        )

        # Skip DCCRMaskNet.__init__ (which hardcodes hid_size=128) and go
        # straight to its parent, BaseDCUMaskNet.
        BaseDCUMaskNet.__init__(
            self,
            encoders=[
                *(DCUNetComplexEncoderBlock(*args, activation="prelu") for args in encoders),
                DCCRMaskNetRNN(np.prod(last_encoder_out_shape), hid_size=rnn_hid_size),
            ],
            decoders=[
                torch.nn.Identity(),
                *(DCUNetComplexDecoderBlock(*args, activation="prelu") for args in decoders[:-1]),
            ],
            output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]),
            **kwargs,
        )


class DCCRNet_mini(DCCRNet):
    """
    Drop-in replacement for the missing `DCCRNet_mini` import.
    A real asteroid DCCRNet, pre-configured with the smaller
    `config.kernel_num_student` channel plan and a matching 32-unit
    complex-LSTM (per the CLSKD paper's stated student config).
    """

    masknet_class = _SmallRNNDCCRMaskNet

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("architecture", _STUDENT_ARCHITECTURE_NAME)
        kwargs.setdefault("sample_rate", 16000.0)
        kwargs.setdefault("rnn_hid_size", cfg.rnn_units_student)
        super().__init__(*args, **kwargs)