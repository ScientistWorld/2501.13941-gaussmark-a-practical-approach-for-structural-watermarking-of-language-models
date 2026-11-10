#!/usr/bin/env python3
"""Watermark evaluator (public TRAIN slice).

The evaluator owns every side of the comparison and runs each method on
the public development inputs this directory holds:

  * the fixed model (Mistral-7B-Instruct-v0.3, bf16) and its weights,
  * the public C4 prompts, the human continuations, the public key seeds,
  * the decoding procedure (temperature 1, top-p 0.9, exactly 200 new tokens,
    one sample per prompt, sampling noise fixed per prompt so every method
    sees the same random numbers),
  * the null texts (unwatermarked-model generations, human continuations and
    marked texts tested against a different key),
  * the token-level corruptions of the robustness experiment,
  * the C4 cross-entropy quality measurement, the thresholds and all gates.

A method is a directory ``<watermarkers-dir>/<name>/`` holding
``watermarker.py`` with two functions:

    make_watermark(model, key_seed) -> {parameter_name: additive_delta_tensor}
    detect(model, key_seed, prompts, texts) -> list of p-values

``make_watermark`` never sees a prompt; the evaluator applies the returned
deltas itself, decodes, measures quality and removes them.  ``detect`` is
called on a shuffled, unlabeled mixture of marked and null texts.  Every
scored number is recomputed here from those raw outputs.  A method whose
outputs are invalid, which tampers with the model, exceeds the quality budget,
or whose p-values reject significantly more than the nominal rate on any null
family is scored as FAILED: it receives the worst value of every scored metric
in every experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Slice configuration (fixed by eval/; not configurable by a method)
# --------------------------------------------------------------------------
SLICE = "train"
WORKSPACE = Path(__file__).resolve().parents[2]
DATA_PATH = Path(__file__).resolve().parent / "data" / "train_set.json"
DEFAULT_OUTPUT = WORKSPACE / "scoring" / "scores_train.json"
DEFAULT_WATERMARKERS = WORKSPACE / "checkpoints" / "watermarkers"
MODEL_RELATIVE = Path("models") / "mistralai" / "Mistral-7B-Instruct-v0.3"
EXPECTED_DOCS = 200
EXPECTED_KEYS = 4

TEMPERATURE = 1.0
TOP_P = 0.9
NUM_SPECIAL_TOKENS = 771          # ids 0..770 are <unk>, <s>, </s>, [control_*]
CORRUPTION_RATE = 0.20            # fraction of generated tokens edited
CE_RATIO_BUDGET = 1.015           # quality gate: CE(marked model)/CE(original) on held-out C4 text
GATE_BINOMIAL_P = 1e-3            # null-validity gate (one-sided binomial test)
ALPHAS = (0.05, 0.01)
PER_METHOD_TIMEOUT_S = 3 * 3600
DETECT_CHUNK = 100

EXP_MAIN = "watermark_detection_mistral"
EXP_ROBUST = "watermark_robustness_mistral"

WORST = {
    EXP_MAIN: {
        "detection_rate_fpr_0.05": 0.0,
        "detection_rate_fpr_0.01": 0.0,
        "auc": 0.0,
        "median_p_value": 1.0,
        "c4_cross_entropy_ratio": CE_RATIO_BUDGET,
        "false_positive_rate_0.05": 1.0,
        "false_positive_rate_0.01": 1.0,
    },
    EXP_ROBUST: {
        "detection_rate_fpr_0.05": 0.0,
        "detection_rate_fpr_0.01": 0.0,
        "auc": 0.0,
        "median_p_value": 1.0,
        "false_positive_rate_0.05": 1.0,
    },
}

NULL_FAMILIES_MAIN = ("null_unwatermarked", "null_human", "null_cross_key")
NULL_FAMILIES_ROBUST = ("null_corrupted",)


def log(msg: str) -> None:
    print(f"[eval:{SLICE}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset() -> dict[str, Any]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    docs = data["docs"]
    keys = data["key_seeds"]
    if len(docs) != EXPECTED_DOCS or len(keys) != EXPECTED_KEYS:
        raise RuntimeError(
            f"{DATA_PATH} holds {len(docs)} docs / {len(keys)} keys; expected "
            f"{EXPECTED_DOCS} / {EXPECTED_KEYS}. Rebuild it with the builder in this directory."
        )
    for d in docs:
        if len(d["prompt_ids"]) != data["prompt_tokens"] + 1 or len(d["human_ids"]) != data["max_new_tokens"]:
            raise RuntimeError("malformed document in evaluation set")
    return data


def model_path() -> Path:
    env = os.environ.get("GAUSSMARK_MODEL_PATH")
    candidates = [Path(env)] if env else []
    candidates += [WORKSPACE / "data" / "downloads" / MODEL_RELATIVE]
    for c in candidates:
        if (c / "config.json").is_file():
            return c
    raise FileNotFoundError(f"model not found; run scripts/download.sh (looked in {candidates})")


# --------------------------------------------------------------------------
# Model handling
# --------------------------------------------------------------------------
class Harness:
    def __init__(self, torch_mod):
        self.torch = torch_mod
        from transformers import AutoModelForCausalLM

        path = model_path()
        log(f"loading model from {path}")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch_mod.bfloat16, device_map="cuda", local_files_only=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.vocab = int(self.model.config.vocab_size)
        self.params = dict(self.model.named_parameters())
        self.pristine = {n: p.detach().clone() for n, p in self.params.items()}
        self.pristine_buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        self.structure = self._structure()
        self.hooks = self._hooks()
        log(f"model loaded in {time.time() - t0:.1f}s on {self.device}")

    # ---- integrity -------------------------------------------------------
    def _structure(self):
        return [(n, id(m), type(m), id(m.forward) if "forward" in m.__dict__ else None)
                for n, m in self.model.named_modules()] + [
            ("param:" + n, id(p), tuple(p.shape)) for n, p in self.model.named_parameters()
        ]

    def _hooks(self):
        import torch.nn.modules.module as mm

        sig = []
        for n, m in self.model.named_modules():
            for attr in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                         "_backward_pre_hooks"):
                sig.append((n, attr, tuple(sorted((getattr(m, attr, None) or {}).keys()))))
        for attr in ("_global_forward_hooks", "_global_forward_pre_hooks",
                     "_global_backward_hooks", "_global_backward_pre_hooks"):
            sig.append(("<global>", attr, tuple(sorted((getattr(mm, attr, None) or {}).keys()))))
        return sig

    def check_and_restore(self) -> list[str]:
        """Return a list of integrity violations and restore the pristine model."""
        problems: list[str] = []
        torch = self.torch
        if self._structure() != self.structure:
            problems.append("model structure changed (modules/parameters/forward replaced)")
        if self._hooks() != self.hooks:
            problems.append("hooks added to or removed from the model")
        if problems:
            raise RuntimeError("; ".join(problems))  # unrecoverable in-process, handled by caller
        changed = []
        for n, p in self.params.items():
            p.requires_grad_(False)
            p.grad = None
            if not torch.equal(p.data, self.pristine[n]):
                changed.append(n)
                p.data.copy_(self.pristine[n])
        for n, b in self.model.named_buffers():
            if n in self.pristine_buffers and not torch.equal(b, self.pristine_buffers[n]):
                changed.append(n)
                b.copy_(self.pristine_buffers[n])
        if changed:
            problems.append(f"method modified model weights in place: {changed[:3]}")
        self.model.eval()
        return problems

    def apply_delta(self, delta: dict[str, Any]) -> None:
        for n, d in delta.items():
            p = self.params[n]
            p.data.copy_((self.pristine[n].float() + d.to(p.device, self.torch.float32)).to(p.dtype))

    def remove_delta(self, delta: dict[str, Any]) -> None:
        for n in delta:
            self.params[n].data.copy_(self.pristine[n])

    def validate_delta(self, delta: Any) -> dict[str, Any]:
        torch = self.torch
        if not isinstance(delta, dict):
            raise ValueError("make_watermark must return a dict {parameter_name: tensor}")
        out = {}
        for n, d in delta.items():
            if n not in self.params:
                raise ValueError(f"unknown parameter name {n!r}")
            if not torch.is_tensor(d):
                raise ValueError(f"delta for {n} is not a tensor")
            if tuple(d.shape) != tuple(self.params[n].shape):
                raise ValueError(f"delta for {n} has shape {tuple(d.shape)}, expected {tuple(self.params[n].shape)}")
            if not torch.isfinite(d).all():
                raise ValueError(f"delta for {n} contains non-finite values")
            out[n] = d.detach()
        return out

    # ---- decoding ----------------------------------------------------------
    def generate(self, prompts: list[list[int]], seeds: list[int], max_new: int) -> list[list[int]]:
        """Temperature-1, top-p 0.9 sampling of exactly ``max_new`` tokens.

        Each prompt owns a CUDA generator seeded from the evaluation set, and
        the Gumbel-max trick turns its uniforms into a sample from the top-p
        distribution, so every method receives identical random numbers.
        """
        torch = self.torch
        model = self.model
        ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
        gens = [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        out_tokens = []
        with torch.no_grad():
            out = model.model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1]
            for _ in range(max_new):
                logits = model.lm_head(h).float() / TEMPERATURE
                logits[:, :NUM_SPECIAL_TOKENS] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                remove_sorted = (cum - sp) >= TOP_P
                remove_sorted[:, 0] = False
                remove = remove_sorted.scatter(1, si, remove_sorted)
                logits = logits.masked_fill(remove, -float("inf"))
                u = torch.stack([torch.rand(self.vocab, generator=g, device=self.device) for g in gens])
                gumbel = -torch.log(-torch.log(u.clamp_min(1e-30)))
                nxt = torch.argmax(logits + gumbel, dim=-1)
                out_tokens.append(nxt)
                out = model.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                h = out.last_hidden_state[:, -1]
        del past, out
        toks = torch.stack(out_tokens, dim=1).cpu().tolist()
        return toks

    # ---- quality -----------------------------------------------------------
    def doc_nll(self, docs: list[list[int]]) -> list[float]:
        """Summed next-token negative log-likelihood of each full document."""
        torch = self.torch
        model = self.model
        res = []
        with torch.no_grad():
            for start in range(0, len(docs), 25):
                batch = torch.tensor(docs[start:start + 25], dtype=torch.long, device=self.device)
                hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state
                for b in range(batch.shape[0]):
                    logits = model.lm_head(hidden[b, :-1]).float()
                    nll = torch.nn.functional.cross_entropy(logits, batch[b, 1:], reduction="sum")
                    res.append(float(nll.item()))
                del hidden
        return res


# --------------------------------------------------------------------------
# Corruption (paper Sec. 4.3: random insertions, deletions, substitutions)
# --------------------------------------------------------------------------
def corrupt(tokens: list[int], attack: str, seed: int, vocab: int) -> list[int]:
    rng = random.Random(int(seed))
    toks = list(tokens)
    n = len(toks)
    k = int(round(CORRUPTION_RATE * n))
    if attack == "substitute":
        for pos in rng.sample(range(n), k):
            toks[pos] = rng.randrange(NUM_SPECIAL_TOKENS, vocab)
        return toks
    if attack == "delete":
        drop = set(rng.sample(range(n), k))
        return [t for i, t in enumerate(toks) if i not in drop]
    if attack == "insert":
        for _ in range(k):
            toks.insert(rng.randrange(len(toks) + 1), rng.randrange(NUM_SPECIAL_TOKENS, vocab))
        return toks
    raise ValueError(attack)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def auc_lower_p_is_positive(pos: list[float], neg: list[float]) -> float:
    """P(p_marked < p_null) + 0.5 P(tie), via average ranks."""
    allv = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    allv.sort(key=lambda t: -t[0])  # ascending score = descending p
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for t in range(i, j + 1):
            ranks[t] = r
        i = j + 1
    rpos = sum(r for r, (_, lab) in zip(ranks, allv) if lab == 1)
    npos, nneg = len(pos), len(neg)
    return (rpos - npos * (npos + 1) / 2.0) / (npos * nneg)


def binom_sf_ge(x: int, n: int, a: float) -> float:
    """P(X >= x) for X ~ Binomial(n, a)."""
    if x <= 0:
        return 1.0
    total = 0.0
    for i in range(x, n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                          + i * math.log(a) + (n - i) * math.log1p(-a))
    return min(1.0, total)


def rate(ps: list[float], a: float) -> float:
    return sum(p < a for p in ps) / len(ps)


def null_gates(pvals: dict[str, list[float]], families) -> tuple[list[str], dict[str, Any]]:
    failures, info = [], {}
    pooled = [p for f in families for p in pvals[f]]
    for fam, ps in [(f, pvals[f]) for f in families] + [("pooled", pooled)]:
        if fam == "pooled" and len(families) == 1:
            continue
        for a in ALPHAS:
            x = sum(p < a for p in ps)
            tail = binom_sf_ge(x, len(ps), a)
            info[f"{fam}@{a}"] = {"rejections": x, "n": len(ps), "rate": x / len(ps), "binomial_tail_p": tail}
            if tail < GATE_BINOMIAL_P:
                failures.append(
                    f"{fam}: {x}/{len(ps)} null texts rejected at p<{a} "
                    f"(binomial tail {tail:.2e} < {GATE_BINOMIAL_P})"
                )
    return failures, info


# --------------------------------------------------------------------------
# Running one method
# --------------------------------------------------------------------------
class MethodFailure(Exception):
    pass


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout(f"method exceeded {PER_METHOD_TIMEOUT_S}s")


def load_plugin(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(f"wm_plugin_{digest}_{path.parent.name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("make_watermark", "detect"):
        if not callable(getattr(mod, fn, None)):
            raise MethodFailure(f"watermarker.py does not define {fn}()")
    return mod


def plugin_digest(method_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(method_dir.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(method_dir)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def validate_pvalues(ps: Any, n: int) -> list[float]:
    try:
        out = [float(p) for p in ps]
    except Exception as e:  # noqa: BLE001
        raise MethodFailure(f"detect() returned non-numeric output: {e}")
    if len(out) != n:
        raise MethodFailure(f"detect() returned {len(out)} p-values for {n} texts")
    for p in out:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise MethodFailure(f"detect() returned an impossible p-value {p!r} (must be finite, in [0,1])")
    return out


def run_method(h: Harness, data: dict[str, Any], shared: dict[str, Any], method_dir: Path,
               dump: dict[str, Any] | None) -> dict[str, Any]:
    torch = h.torch
    plugin_file = method_dir / "watermarker.py"
    if not plugin_file.is_file():
        raise MethodFailure("no watermarker.py in method directory")
    plugin = load_plugin(plugin_file)
    problems = h.check_and_restore()
    if problems:
        raise MethodFailure("; ".join(problems))

    docs = data["docs"]
    keys = data["key_seeds"]
    max_new = data["max_new_tokens"]
    groups = [[i for i, d in enumerate(docs) if d["key_index"] == k] for k in range(len(keys))]
    marked: list[list[int] | None] = [None] * len(docs)
    nll_wm: list[float | None] = [None] * len(docs)

    # ---- phase 1: watermark, decode, measure quality -----------------------
    for k, key_seed in enumerate(keys):
        raw = plugin.make_watermark(h.model, int(key_seed))
        problems = h.check_and_restore()
        if problems:
            raise MethodFailure("; ".join(problems))
        delta = h.validate_delta(raw)
        del raw
        h.apply_delta(delta)
        try:
            idx = groups[k]
            gen = h.generate([docs[i]["prompt_ids"] for i in idx], [docs[i]["gen_seed"] for i in idx], max_new)
            for i, g in zip(idx, gen):
                marked[i] = g
            nll = h.doc_nll([docs[i]["prompt_ids"] + docs[i]["human_ids"] for i in idx])
            for i, v in zip(idx, nll):
                nll_wm[i] = v
        finally:
            h.remove_delta(delta)
        del delta
        gc.collect()
        torch.cuda.empty_cache()
        problems = h.check_and_restore()
        if problems:
            raise MethodFailure("; ".join(problems))

    # ---- phase 2: blinded detection ---------------------------------------
    pv: dict[str, list[float]] = {f: [] for f in
                                  ("marked", "marked_corrupted") + NULL_FAMILIES_MAIN + NULL_FAMILIES_ROBUST}
    pv_doc: dict[str, list[int]] = {f: [] for f in pv}
    order_rng = random.Random(int(data["detect_order_seed"]))
    for k, key_seed in enumerate(keys):
        prev = groups[(k - 1) % len(keys)]
        items = []
        for i in groups[k]:
            d = docs[i]
            items.append(("marked", i, d["prompt_ids"], marked[i]))
            items.append(("null_unwatermarked", i, d["prompt_ids"], shared["null"][i]))
            items.append(("null_human", i, d["prompt_ids"], d["human_ids"]))
            items.append(("marked_corrupted", i, d["prompt_ids"],
                          corrupt(marked[i], d["attack"], d["corrupt_seed"], h.vocab)))
            items.append(("null_corrupted", i, d["prompt_ids"], shared["null_corrupted"][i]))
        for i in prev:
            items.append(("null_cross_key", i, docs[i]["prompt_ids"], marked[i]))
        order_rng.shuffle(items)
        for start in range(0, len(items), DETECT_CHUNK):
            chunk = items[start:start + DETECT_CHUNK]
            prompts = [list(c[2]) for c in chunk]
            texts = [list(c[3]) for c in chunk]
            out = plugin.detect(h.model, int(key_seed), prompts, texts)
            problems = h.check_and_restore()
            if problems:
                raise MethodFailure("; ".join(problems))
            ps = validate_pvalues(out, len(chunk))
            for (fam, i, _, _), p in zip(chunk, ps):
                pv[fam].append(p)
                pv_doc[fam].append(i)
        gc.collect()
        torch.cuda.empty_cache()

    # ---- metrics and gates ---------------------------------------------------
    ce_wm = sum(nll_wm) / (len(docs) * (len(docs[0]["prompt_ids"]) + max_new - 1))
    ce_0 = shared["ce_0"]
    ce_ratio = ce_wm / ce_0
    gate_fail: list[str] = []
    if not math.isfinite(ce_ratio) or ce_ratio > CE_RATIO_BUDGET:
        gate_fail.append(f"quality budget exceeded: C4 cross-entropy ratio {ce_ratio:.5f} > {CE_RATIO_BUDGET}")
    f_main, g_main = null_gates(pv, NULL_FAMILIES_MAIN)
    f_rob, g_rob = null_gates(pv, NULL_FAMILIES_ROBUST)
    gate_fail += f_main + f_rob

    pooled_null = [p for f in NULL_FAMILIES_MAIN for p in pv[f]]
    main = {
        "detection_rate_fpr_0.05": rate(pv["marked"], 0.05),
        "detection_rate_fpr_0.01": rate(pv["marked"], 0.01),
        "auc": auc_lower_p_is_positive(pv["marked"], pooled_null),
        "median_p_value": median(pv["marked"]),
        "c4_cross_entropy_ratio": ce_ratio,
        "false_positive_rate_0.05": rate(pooled_null, 0.05),
        "false_positive_rate_0.01": rate(pooled_null, 0.01),
    }
    robust = {
        "detection_rate_fpr_0.05": rate(pv["marked_corrupted"], 0.05),
        "detection_rate_fpr_0.01": rate(pv["marked_corrupted"], 0.01),
        "auc": auc_lower_p_is_positive(pv["marked_corrupted"], pv["null_corrupted"]),
        "median_p_value": median(pv["marked_corrupted"]),
        "false_positive_rate_0.05": rate(pv["null_corrupted"], 0.05),
    }
    if dump is not None:
        dump.update({"pvalues": pv, "doc_index": pv_doc, "nll_watermarked": nll_wm,
                     "nll_original": shared["nll_0"]})
    return {
        "measured": {EXP_MAIN: main, EXP_ROBUST: robust},
        "gate_failures": gate_fail,
        "gates": {"quality": {"c4_cross_entropy_ratio": ce_ratio, "budget": CE_RATIO_BUDGET,
                              "ce_original": ce_0, "ce_watermarked": ce_wm},
                  "null_main": g_main, "null_robust": g_rob},
        "counts": {f: len(v) for f, v in pv.items()},
    }


def worst_results() -> dict[str, dict[str, float]]:
    return {e: dict(m) for e, m in WORST.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def write_scores(path: Path, results: dict[str, dict[str, dict[str, float]]], status: dict[str, Any]) -> None:
    scores = {"experiments": {e: {"results": {}} for e in WORST}, "_slice": SLICE, "_status": status}
    for name in sorted(results):
        for e in WORST:
            scores["experiments"][e]["results"][name] = results[name][e]
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(scores, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--watermarkers-dir", default=str(DEFAULT_WATERMARKERS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--resume-dir", default=None, help="(audit harness only) cache per-method results")
    ap.add_argument("--dump-dir", default=None, help="(audit harness only) write per-text p-values")
    args = ap.parse_args()

    wm_dir = Path(args.watermarkers_dir)
    out_path = Path(args.output)
    methods = sorted(p for p in wm_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
                     and p.name != "__pycache__") if wm_dir.is_dir() else []
    log(f"{len(methods)} method(s) on disk in {wm_dir}: {[m.name for m in methods]}")

    results = {m.name: worst_results() for m in methods}
    status: dict[str, Any] = {m.name: {"status": "pending"} for m in methods}
    write_scores(out_path, results, status)  # stale scores never survive a crash
    if not methods:
        return

    data = load_dataset()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to run the watermark evaluator.")
    torch.backends.cuda.matmul.allow_tf32 = False
    h = Harness(torch)

    # ---- shared, method-independent quantities -------------------------------
    t0 = time.time()
    docs = data["docs"]
    keys = data["key_seeds"]
    groups = [[i for i, d in enumerate(docs) if d["key_index"] == k] for k in range(len(keys))]
    null: list[list[int] | None] = [None] * len(docs)
    for idx in groups:
        gen = h.generate([docs[i]["prompt_ids"] for i in idx], [docs[i]["null_seed"] for i in idx],
                         data["max_new_tokens"])
        for i, g in zip(idx, gen):
            null[i] = g
    nll_0 = h.doc_nll([d["prompt_ids"] + d["human_ids"] for d in docs])
    ce_0 = sum(nll_0) / (len(docs) * (len(docs[0]["prompt_ids"]) + data["max_new_tokens"] - 1))
    shared = {
        "null": null,
        "null_corrupted": [corrupt(null[i], d["attack"], d["null_corrupt_seed"], h.vocab) for i, d in enumerate(docs)],
        "nll_0": nll_0,
        "ce_0": ce_0,
    }
    log(f"shared null texts and original C4 cross-entropy {ce_0:.5f} ready in {time.time() - t0:.1f}s")

    resume = Path(args.resume_dir) if args.resume_dir else None
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    for d in (resume, dump_dir):
        if d:
            d.mkdir(parents=True, exist_ok=True)

    for mdir in methods:
        name = mdir.name
        t1 = time.time()
        digest = plugin_digest(mdir)
        cache = resume / f"{name}.json" if resume else None
        if cache and cache.is_file():
            cached = json.load(open(cache))
            if cached.get("digest") == digest and cached.get("data_digest") == data.get("digest"):
                results[name], status[name] = cached["results"], cached["status"]
                log(f"{name}: reused cached result ({status[name]['status']})")
                write_scores(out_path, results, status)
                continue
        log(f"{name}: evaluating")
        dump: dict[str, Any] | None = {} if dump_dir else None
        st: dict[str, Any] = {"plugin_digest": digest}
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(PER_METHOD_TIMEOUT_S)
        try:
            r = run_method(h, data, shared, mdir, dump)
            signal.alarm(0)
            st.update({"gates": r["gates"], "counts": r["counts"], "measured": r["measured"]})
            if r["gate_failures"]:
                st.update({"status": "failed", "reason": r["gate_failures"]})
                results[name] = worst_results()
            else:
                st["status"] = "ok"
                results[name] = r["measured"]
        except BaseException as e:  # noqa: BLE001 - any failure of the method scores as failed
            signal.alarm(0)
            if isinstance(e, KeyboardInterrupt):
                raise
            st.update({"status": "failed", "reason": [f"{type(e).__name__}: {e}"],
                       "traceback": traceback.format_exc()[-4000:]})
            results[name] = worst_results()
            try:
                h.check_and_restore()
            except Exception:  # structure/hook tampering: reload a clean model
                log("model structure was altered; reloading model")
                del h
                gc.collect()
                torch.cuda.empty_cache()
                h = Harness(torch)
        finally:
            signal.signal(signal.SIGALRM, old)
        st["elapsed_s"] = round(time.time() - t1, 1)
        status[name] = st
        log(f"{name}: {st['status']} in {st['elapsed_s']}s "
            f"{st.get('reason', '')} {json.dumps(results[name])}")
        write_scores(out_path, results, status)
        if cache:
            with open(cache, "w") as f:
                json.dump({"digest": digest, "data_digest": data.get("digest"),
                           "results": results[name], "status": st}, f)
        if dump_dir and dump is not None and dump:
            with open(dump_dir / f"{name}.json", "w") as f:
                json.dump(dump, f)
        gc.collect()
        torch.cuda.empty_cache()

    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
