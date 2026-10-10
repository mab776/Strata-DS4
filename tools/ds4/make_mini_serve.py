#!/usr/bin/env python3
"""tools/ds4/make_mini_serve.py - (mab776 M5) the tame mini fixture WITH routed experts that compute something.

bench/ds4-2026-10-06/dense/make_mini_tame.py (the numerically tame variant of tools/ds4/make_mini_gguf.py) keeps the
routed experts at zero bytes: random quant bytes have random f16 block scales, which dequantize to NaN.  With zero
experts the expert tier never reaches the logits, so an engine that hands the tier a wrong buffer (a stale prefill
routed-sum pointer after prefill_release, say) still passes every logits gate.  This variant writes valid blocks:
random quant bytes with a finite, small block scale (IQ2_XXS: d; Q2_K: d and dmin), sized so a routed expert's output
is about as large as the shared expert's.  Same geometry, tensors and code paths as the tame fixture otherwise
(DS4_SWA=<n> works the same way).

    python3 tools/ds4/make_mini_serve.py <out.gguf>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "ds4-2026-10-06" / "dense"))

import make_mini_tame as T                   # noqa: E402  (patches make_mini_gguf for the tame values)
from gguf import GGMLQuantizationType as Q   # noqa: E402

# block layouts (ggml-common.h): block_iq2_xxs = { f16 d; u16 qs[32] } (66 B); block_q2_K = { u8 scales[16]; u8 qs[64];
# f16 d; f16 dmin } (84 B).  |w| ~ 50 d for IQ2_XXS (grid values 8/25/43, scale (0.5 + s) / 4), ~8 d for Q2_K.
IQ2_D = 0.00125          # gate/up: fan-in 256 -> |w| ~ 1/16
Q2K_D, Q2K_DMIN = 0.008, 0.012

_add = T.GGUFWriter.add_tensor
_rng = np.random.default_rng(20261010)


def add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None):
    if raw_dtype in (Q.IQ2_XXS, Q.Q2_K):
        bs = 66 if raw_dtype == Q.IQ2_XXS else 84
        blocks = _rng.integers(0, 256, size=tensor.size, dtype=np.uint8).reshape(-1, bs)
        if raw_dtype == Q.IQ2_XXS:
            blocks[:, 0:2] = np.frombuffer(np.float16(IQ2_D).tobytes(), dtype=np.uint8)
        else:
            blocks[:, 80:82] = np.frombuffer(np.float16(Q2K_D).tobytes(), dtype=np.uint8)
            blocks[:, 82:84] = np.frombuffer(np.float16(Q2K_DMIN).tobytes(), dtype=np.uint8)
        tensor = blocks.reshape(tensor.shape)
    return _add(self, name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)


T.GGUFWriter.add_tensor = add_tensor

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini-ds4-serve.gguf")
    T.M.write_model(out)
    print("wrote", out, out.stat().st_size, "bytes")
