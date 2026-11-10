"""GaussMark watermarker (Block, Rakhlin, Sekhari 2025), as an evaluator plugin.

GaussMark.Generate (Algorithm 1)
    sample a key xi ~ N(0, sigma^2 I) over ONE MLP weight matrix theta and
    generate with theta + xi.  Here ``make_watermark`` returns that additive
    delta; the evaluator applies it and decodes.

GaussMark.Detect (Algorithm 2)
    psi = <xi, grad_theta log p_theta(y | x)> / (sigma * ||grad_theta log p_theta(y | x)||)
    p   = 1 - Phi(psi)
    The gradient is taken at the ORIGINAL weights; under H0 (key independent of
    the text) psi ~ N(0, 1) exactly, so p is a valid p-value (Prop. 3.2).

Rank-reduced GaussMark (Sec. 4.4 / App. E), enabled by ``drop_top_pcs > 0``:
    the key is projected onto the complement of the top-k right singular
    directions of the weight matrix, and the gradient norm in the denominator
    is taken in that same subspace.

Settings live in ``config.json`` next to this file; the defaults are the
paper's Mistral-7B settings (Table 2): layer 20, up_proj, sigma^2 = 1e-5.
``min_token_prob`` (default 0 = off) restricts the log-likelihood to tokens the
original model assigns at least that probability; the selection depends only on
(prompt, text, original weights), so the test stays exact under H0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from scipy.stats import norm

DEFAULTS = {
    "layer": 20,
    "weight": "up_proj",
    "variance": 1e-5,      # sigma^2 (paper Table 2, Mistral-7B)
    "drop_top_pcs": 0,     # rank reduction (paper Table 7 uses 1024 for Mistral)
    "min_token_prob": 0.0,
}


def _config() -> dict:
    cfg = dict(DEFAULTS)
    path = Path(__file__).with_name("config.json")
    if path.is_file():
        cfg.update(json.loads(path.read_text()))
    return cfg


CFG = _config()
SIGMA = math.sqrt(float(CFG["variance"]))
TARGET = f"model.layers.{int(CFG['layer'])}.mlp.{CFG['weight']}.weight"
_CACHE: dict = {}


def _param(model) -> torch.nn.Parameter:
    return dict(model.named_parameters())[TARGET]


def _basis(model):
    """Top-k right singular vectors of the target weight (None if no rank reduction)."""
    k = int(CFG["drop_top_pcs"])
    if k <= 0:
        return None
    if "basis" not in _CACHE:
        w = _param(model).detach().float()
        # eigenvectors of W^T W = right singular vectors of W
        evals, evecs = torch.linalg.eigh(w.T @ w)
        _CACHE["basis"] = evecs[:, -k:].contiguous()  # largest k
    return _CACHE["basis"]


def _project(t: torch.Tensor, basis) -> torch.Tensor:
    return t if basis is None else t - (t @ basis) @ basis.T


def _key(model, key_seed: int) -> torch.Tensor:
    """xi ~ N(0, sigma^2 I) (projected for rank reduction), deterministic in key_seed."""
    if _CACHE.get("key_seed") == key_seed:
        return _CACHE["key"]
    p = _param(model)
    g = torch.Generator(device="cpu").manual_seed(int(key_seed))
    xi = torch.randn(tuple(p.shape), generator=g, dtype=torch.float32).to(p.device) * SIGMA
    xi = _project(xi, _basis(model))
    _CACHE.update({"key_seed": key_seed, "key": xi})
    return xi


def make_watermark(model, key_seed: int) -> dict:
    """GaussMark.Generate: theta(xi) = theta + xi on a single MLP weight."""
    return {TARGET: _key(model, key_seed).clone()}


def _log_likelihood(model, prompt: list[int], text: list[int]) -> torch.Tensor:
    device = _param(model).device
    ids = torch.tensor([list(prompt) + list(text)], dtype=torch.long, device=device)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    h = hidden[len(prompt) - 1: ids.shape[1] - 1]
    logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
    tok = logp.gather(1, ids[0, len(prompt):, None]).squeeze(1)
    if CFG["min_token_prob"] > 0:
        keep = (tok.detach() >= math.log(float(CFG["min_token_prob"]))).float()
        return (tok * keep).sum()
    return tok.sum()


def detect(model, key_seed: int, prompts: list[list[int]], texts: list[list[int]]) -> list[float]:
    """GaussMark.Detect: one-sided p-value 1 - Phi(psi) per (prompt, text)."""
    w = _param(model)
    xi = _key(model, key_seed)
    basis = _basis(model)
    out = []
    w.requires_grad_(True)
    try:
        for prompt, text in zip(prompts, texts):
            if len(text) == 0:
                out.append(1.0)
                continue
            ll = _log_likelihood(model, prompt, text)
            grad = torch.autograd.grad(ll, w)[0].float()
            grad = _project(grad, basis)
            gnorm = grad.norm()
            if not torch.isfinite(gnorm) or gnorm.item() == 0.0:
                out.append(1.0)
                continue
            psi = (xi * grad).sum() / (SIGMA * gnorm)
            out.append(float(norm.sf(psi.item())))
            del grad, ll
    finally:
        w.requires_grad_(False)
        w.grad = None
    return out
