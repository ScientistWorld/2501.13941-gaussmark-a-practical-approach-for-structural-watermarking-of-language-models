"""Unwatermarked baseline as an evaluator plugin.

Generation is left untouched (no weight perturbation).  Detection runs the
same normalized log-likelihood-gradient score test against a Gaussian key that
was never applied to the model (layer 20 up_proj, sigma^2 = 1e-5), so the
p-values are uniform on marked and null texts alike: this is the
"watermark removed" ablation.  It sets the floor for detection and the
ceiling for quality (C4 cross-entropy ratio exactly 1).
"""

from __future__ import annotations

import math

import torch
from scipy.stats import norm

SIGMA = math.sqrt(1e-5)
TARGET = "model.layers.20.mlp.up_proj.weight"


def make_watermark(model, key_seed: int) -> dict:
    return {}


def detect(model, key_seed: int, prompts, texts) -> list[float]:
    w = dict(model.named_parameters())[TARGET]
    g = torch.Generator(device="cpu").manual_seed(int(key_seed))
    xi = torch.randn(tuple(w.shape), generator=g, dtype=torch.float32).to(w.device) * SIGMA
    out = []
    w.requires_grad_(True)
    try:
        for prompt, text in zip(prompts, texts):
            if len(text) == 0:
                out.append(1.0)
                continue
            ids = torch.tensor([list(prompt) + list(text)], dtype=torch.long, device=w.device)
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            h = hidden[len(prompt) - 1: ids.shape[1] - 1]
            logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
            ll = logp.gather(1, ids[0, len(prompt):, None]).sum()
            grad = torch.autograd.grad(ll, w)[0].float()
            gnorm = grad.norm()
            if not torch.isfinite(gnorm) or gnorm.item() == 0.0:
                out.append(1.0)
                continue
            out.append(float(norm.sf(((xi * grad).sum() / (SIGMA * gnorm)).item())))
    finally:
        w.requires_grad_(False)
        w.grad = None
    return out
