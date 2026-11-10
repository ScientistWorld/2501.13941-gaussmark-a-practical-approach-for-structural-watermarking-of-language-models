"""Actual GaussMark runs, a real resource fix, and brute-force generation.

The Gaussian parameter perturbation and normalized likelihood-gradient test are
the actual method. Nothing here trains a surrogate or changes model/tokenizer.
All model computation is performed in the submitted GPU job.
"""
from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, '/home/user/method')
from gaussmark import GaussMark
import run_pipeline as pipeline
from common import (AUDIT, SOURCE, WM_FILE, UW_FILE, read, write, evaluate_trial)

SIGMA = math.sqrt(1e-5)
CASES = ['reference_seed_42', 'reference_seed_1042', 'reference_seed_2042',
         'honest_efficiency_fix', 'honest_residual64', 'shortcut_bruteforce_rejection']


class ResidualGaussMark(GaussMark):
    """Gaussian noise confined to the complement of 64 leading right directions.

    Uses the projected-gradient norm for a valid Gaussian null test in the same
    subspace. This is a genuine methodology variant selected before test scores.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(31415)
        with torch.no_grad():
            _, _, v = torch.svd_lowrank(self.target_param.float(), q=64, niter=3)
        self.v = v

    def project(self, tensor):
        return tensor - (tensor @ self.v) @ self.v.T

    def regenerate_watermark_key(self, seed):
        raw = super().regenerate_watermark_key(seed)
        with torch.no_grad():
            key = self.project(torch.from_numpy(raw).to(self.device))
        return key.cpu().numpy()

    def detect(self, input_ids, generated_ids, watermark_key, attention_mask=None, alpha=0.05):
        # Same log-likelihood gradient as GaussMark; restrict its norm to the
        # Gaussian noise subspace. Use fp32 accumulation for the statistic.
        inp = input_ids.to(self.device)
        gen = generated_ids.to(self.device)
        full = torch.cat([inp, gen], dim=-1)
        self.target_param.requires_grad_(True)
        try:
            out = self.model(full, attention_mask=torch.ones_like(full))
            logits = out.logits[:, inp.shape[1]-1:full.shape[1]-1]
            lp = logits.log_softmax(-1).gather(2, gen.unsqueeze(-1)).sum()
            grad = torch.autograd.grad(lp, self.target_param)[0].float()
            grad = self.project(grad)
            xi = torch.from_numpy(watermark_key).to(self.device)
            z = float((torch.sum(xi * grad) / (self.sigma * grad.norm())).item())
        finally:
            self.target_param.requires_grad_(False)
        return {'p_value': float(norm.sf(z)), 'psi': z}


def save_progress(path, value):
    write(path, value)


def detector(wm, sample, key):
    return wm.detect(torch.tensor([sample['prompt_ids']]),
                     torch.tensor([sample['generated_ids']]), key)


def generate_one(wm, tokenizer, prompt_ids, seed, sampling_seed=None):
    # Regenerate with exactly the reference NumPy Gaussian key, including dtype.
    key = wm.regenerate_watermark_key(seed)
    torch.manual_seed(seed if sampling_seed is None else sampling_seed)
    torch.cuda.manual_seed_all(seed if sampling_seed is None else sampling_seed)
    inp = torch.tensor([prompt_ids], device=wm.device)
    old = wm.target_param.data.clone()
    wm.target_param.data = old + torch.from_numpy(key).to(wm.device, dtype=old.dtype)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            out = wm.model.generate(inp, attention_mask=torch.ones_like(inp),
                max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=getattr(wm.model.config, 'pad_token_id', None) or 0)
        torch.cuda.synchronize()
    finally:
        wm.target_param.data = old
    ids = out[0, len(prompt_ids):].cpu().tolist()
    return {'prompt_ids': prompt_ids, 'generated_ids': ids,
            'generated_text': tokenizer.decode(ids, skip_special_tokens=True),
            'sample_seed': seed, 'gen_time': time.perf_counter()-start}, key


def generate_null(wm, tokenizer, prompt_ids, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inp = torch.tensor([prompt_ids], device=wm.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = wm.model.generate(inp, attention_mask=torch.ones_like(inp),
            max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.9,
            pad_token_id=getattr(wm.model.config, 'pad_token_id', None) or 0)
    torch.cuda.synchronize()
    ids = out[0, len(prompt_ids):].cpu().tolist()
    return {'prompt_ids': prompt_ids, 'generated_ids': ids,
            'generated_text': tokenizer.decode(ids, skip_special_tokens=True),
            'sample_seed': seed, 'gen_time': time.perf_counter()-start}


def quality_check(wm, tokenizer, prompts):
    def mean_ce():
        losses = []
        with torch.no_grad():
            for prompt in prompts[:25]:
                x = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256).to(wm.device)
                losses.append(float(wm.model(**x, labels=x['input_ids']).loss.item()))
        return statistics.mean(losses)
    base = mean_ce()
    original = wm.target_param.data.clone()
    key = wm.regenerate_watermark_key(42)
    try:
        wm.target_param.data = original + torch.from_numpy(key).to(wm.device, dtype=original.dtype)
        marked = mean_ce()
    finally:
        wm.target_param.data = original
    return {'c4_cross_entropy_unwatermarked': base,
            'c4_cross_entropy_watermarked': marked,
            'c4_cross_entropy_ratio': marked/base,
            'c4_cross_entropy_delta': marked-base}


def execute(name, model, tokenizer, prompts):
    outdir = AUDIT / name
    outdir.mkdir(parents=True, exist_ok=True)
    fixed = name == 'honest_efficiency_fix'
    residual = name == 'honest_residual64'
    brute = name == 'shortcut_bruteforce_rejection'
    cached_generation = fixed or brute
    seedbase = int(name.rsplit('_', 1)[1]) if name.startswith('reference_seed_') else 42
    # Reset original flags. The published implementation leaves non-target
    # parameters trainable, retaining unneeded activation/gradient graphs.
    for param in model.parameters():
        param.requires_grad_(not fixed and not residual and not brute)
    wmclass = ResidualGaussMark if residual else GaussMark
    wm = wmclass(model=model, layer_idx=20, weight_name='up_proj',
                 sigma=1.5*SIGMA if residual else SIGMA, device='cuda')
    original_marked = read(SOURCE / 'watermarked_wm_l20_up_proj_s00031622776601683794.json')['samples']
    original_null = read(SOURCE / 'unwatermarked_baseline_exact.json')['samples']
    prompt_ids = [s['prompt_ids'] for s in original_marked]
    statepath = outdir / 'measurements.json'
    state = read(statepath) if statepath.exists() else {
        'case': name, 'seed_base': seedbase, 'marked': [], 'null': [],
        'sigma': wm.sigma, 'max_trials': 12 if brute else 1,
        'model': pipeline.MODEL_PATH, 'prompt_count': 100,
        'generation': {'max_new_tokens': 200, 'temperature': 1.0, 'top_p': 0.9}}
    for i in range(len(state['marked']), 100):
        if cached_generation:
            sample = original_marked[i].copy()
            key = wm.regenerate_watermark_key(42+i)
        else:
            sample, key = generate_one(wm, tokenizer, prompt_ids[i], seedbase+i)
        trials = 1
        gen_time = sample['gen_time']
        detect_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        result = detector(wm, sample, key) if len(sample['generated_ids']) >= 50 else None
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-detect_start
        peak = torch.cuda.max_memory_allocated()
        if brute:
            # Enumerate fresh continuations using the SAME key, until p<.01.
            # This trades generation latency for detection, which test omits.
            while trials < 12 and (result is None or result['p_value'] >= 0.01):
                sample, key = generate_one(wm, tokenizer, prompt_ids[i], 42+i,
                                           42+i+100003*trials)
                gen_time += sample['gen_time']
                begin = time.perf_counter()
                result = detector(wm, sample, key) if len(sample['generated_ids']) >= 50 else None
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-begin
                trials += 1
        state['marked'].append({'prompt_idx': i, 'sample': sample, 'result': result,
            'detect_seconds': elapsed, 'peak_allocated_bytes': peak,
            'trials': trials, 'total_generation_seconds': gen_time})
        save_progress(statepath, state)
        print(name, 'marked', i+1, 'p=', None if result is None else result['p_value'], flush=True)
        del key
    for i in range(len(state['null']), 100):
        sample = original_null[i].copy() if fixed or residual or brute else generate_null(
            wm, tokenizer, prompt_ids[i], seedbase+i)
        key = wm.regenerate_watermark_key(seedbase+i)
        start = time.perf_counter()
        result = detector(wm, sample, key) if len(sample['generated_ids']) >= 50 else None
        torch.cuda.synchronize()
        state['null'].append({'prompt_idx': i, 'sample': sample, 'result': result,
                              'detect_seconds': time.perf_counter()-start})
        save_progress(statepath, state)
        print(name, 'null', i+1, flush=True)
        del key
    state['quality'] = quality_check(wm, tokenizer, prompts)
    state['detection_seconds'] = sum(s['detect_seconds'] for s in state['marked']+state['null'])
    state['generation_time_ratio'] = (sum(s['total_generation_seconds'] for s in state['marked']) /
                                      sum(s['sample']['gen_time'] for s in state['null']))
    state['peak_allocated_bytes'] = max(s['peak_allocated_bytes'] for s in state['marked'])
    state['torch_version'] = torch.__version__
    state['gpu'] = torch.cuda.get_device_name()
    state['cached_generation_reused'] = cached_generation
    state['timing_note'] = ('Cached first-trial generation times come from the reference run; '
        'new generation timings exclude CPU key construction. Detection timings are newly '
        'measured on the same audit GPU. Brute-force detector time is extra generation work.')
    save_progress(statepath, state)
    artifacts = {
        WM_FILE: {'p_values': [s['result']['p_value'] for s in state['marked'] if s['result'] is not None],
                  **state['quality'], 'generation_time_ratio': state['generation_time_ratio']},
        UW_FILE: {'p_values': [s['result']['p_value'] for s in state['null'] if s['result'] is not None]},
    }
    # Scoring is launched by score_gpu.py after this model process has exited.
    # Retain only passive artifacts across the method/test boundary.
    for filename, obj in artifacts.items():
        write(outdir/'artifacts'/filename, obj)
    del wm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case', choices=CASES+['all'])
    args = p.parse_args()
    model, tokenizer = pipeline.load_model_and_tokenizer(pipeline.MODEL_PATH)
    model.tokenizer = tokenizer
    prompts = pipeline.load_c4_prompts(100, tokenizer)
    errors = {}
    for case in CASES if args.case == 'all' else [args.case]:
        try:
            execute(case, model, tokenizer, prompts)
        except Exception:
            import traceback
            errors[case] = traceback.format_exc()
            write(AUDIT/'gpu_errors.json', errors)
            print(errors[case], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    if errors:
        raise RuntimeError(f'Failed cases: {list(errors)}')
