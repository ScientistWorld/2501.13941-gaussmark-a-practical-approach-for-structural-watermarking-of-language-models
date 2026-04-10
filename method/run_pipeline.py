#!/usr/bin/env python3
"""
GaussMark end-to-end pipeline for reproduction.

This script runs the full pipeline:
1. Generate watermarked text with sigma=sqrt(1e-5), because Table 2 reports
   variance sigma^2=1e-5 for Mistral.
2. Run detection on watermarked text (using saved keys)
3. Run detection on unwatermarked baseline
4. Evaluate and write scores.json

The key fix: watermark keys are saved with names matching the generation seed,
and detection loads them from the correct paths.
"""

import json
import os
import sys
import time
import statistics
import gzip
from pathlib import Path

import torch
import numpy as np
from scipy.stats import norm
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, '/home/user/method')
from gaussmark import GaussMark

MODEL_PATH = "/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
CHECKPOINT_DIR = "/home/user/checkpoints"
C4_DATA_PATH = "/home/user/data/c4_realnewslike"

# Paper parameters for Mistral-7B (from Table 2)
LAYER_IDX = 20
WEIGHT_NAME = "up_proj"
# Table 2 reports variance sigma^2=1e-5 for Mistral-7B. The code parameter is
# the standard deviation sigma used in xi ~ N(0, sigma^2 I).
SIGMA = math.sqrt(1e-5)
NUM_TOKENS = 200
NUM_PROMPTS = 100  # 100 prompts × 1 seed = 100 samples for better stats
NUM_SEEDS = 1
SEED_BASE = 42
QUALITY_PROMPTS = 25


def _truncate_prompt(text, tokenizer, max_tokens=200):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(enc["input_ids"], skip_special_tokens=True)


def load_c4_prompts(num_prompts=100, tokenizer=None, max_tokens=200):
    """Load prompts from C4 realnewslike dataset."""
    print(f"[+] Loading C4 prompts...")

    # Try examples.json first (list of dicts with 'text' key)
    examples_path = Path(C4_DATA_PATH) / "examples.json"
    if examples_path.exists():
        with open(examples_path, 'r') as f:
            data = json.load(f)
        prompts = []
        for item in data[:num_prompts]:
            text = item.get('text', '')
            truncated = _truncate_prompt(text, tokenizer, max_tokens) if tokenizer else text
            prompts.append(truncated)
        print(f"[+] Loaded {len(prompts)} prompts from examples.json")
        return prompts

    # Fallback: load from compressed JSONL files
    gz_files = sorted(Path(C4_DATA_PATH).glob("*.json.gz"))
    if gz_files:
        prompts = []
        for gz_file in gz_files:
            with gzip.open(gz_file, 'rt') as f:
                for line in f:
                    if len(prompts) >= num_prompts:
                        break
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    truncated = _truncate_prompt(text, tokenizer, max_tokens) if tokenizer else text
                    prompts.append(truncated)
            if len(prompts) >= num_prompts:
                break
        prompts = prompts[:num_prompts]
        print(f"[+] Loaded {len(prompts)} prompts from gzip")
        return prompts

    raise FileNotFoundError(f"No C4 data found at {C4_DATA_PATH}")


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load model and tokenizer."""
    print(f"[+] Loading model from {model_path}...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()
    print(f"[+] Model loaded in {time.time()-t0:.1f}s. Device: {next(model.parameters()).device}")
    return model, tokenizer


def generate_watermarked(model, tokenizer, prompts, layer_idx, weight_name, sigma,
                          num_tokens, num_seeds, seed_base, output_path):
    """Generate watermarked text with GaussMark."""
    print(f"\n[+] GaussMark generation config:")
    print(f"    layer={layer_idx}, weight={weight_name}, sigma={sigma}")
    print(f"    tokens={num_tokens}, seeds={num_seeds}, prompts={len(prompts)}")

    wm = GaussMark(model=model, layer_idx=layer_idx, weight_name=weight_name,
                    sigma=sigma, device="cuda")
    model.tokenizer = tokenizer

    # Output path for this configuration
    cfg_name = f"wm_l{layer_idx}_{weight_name}_s{sigma}".replace('.', '')
    results_file = os.path.join(CHECKPOINT_DIR, f"watermarked_{cfg_name}.json")
    keys_dir = os.path.join(CHECKPOINT_DIR, f"wm_keys_{cfg_name}")
    os.makedirs(keys_dir, exist_ok=True)

    # Check if already generated
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = json.load(f)
        if (
            existing.get("config", {}).get("exact_prompt_ids")
            and len(existing.get('samples', [])) >= NUM_PROMPTS * NUM_SEEDS
        ):
            # Ensure .pt keys exist for all samples
            samples = existing.get('samples', [])
            pt_missing = []
            for s in samples:
                expected_seed = s.get("sample_seed", s.get("seed", SEED_BASE))
                pt = os.path.join(keys_dir, f"key_seed{expected_seed}.pt")
                if not os.path.exists(pt):
                    pt_missing.append(s)
                else:
                    s["watermark_key_path"] = pt
            if not pt_missing:
                print(f"[+] Generation + per-sample .pt keys already exist at {results_file}, skipping...")
                with open(results_file, 'w') as f:
                    json.dump(existing, f, indent=2)
                return results_file
            else:
                print(f"[+] Generation exists but .pt keys missing for {len(pt_missing)} samples, regenerating...")
                # Regenerate missing .pt keys using the same RNG seed
                wm = GaussMark(model=model, layer_idx=layer_idx, weight_name=weight_name,
                               sigma=sigma, device="cuda")
                for s in pt_missing:
                    seed = s.get("sample_seed", s.get("seed", SEED_BASE))
                    key = wm.regenerate_watermark_key(seed)
                    pt = os.path.join(keys_dir, f"key_seed{seed}.pt")
                    torch.save(torch.from_numpy(key), pt)
                    s["watermark_key_path"] = pt
                with open(results_file, 'w') as f:
                    json.dump(existing, f, indent=2)
                print(f"[+] Regenerated {len(pt_missing)} .pt keys")
                return results_file

    results = {
        "config": {
            "layer_idx": layer_idx, "weight_name": weight_name,
            "sigma": sigma, "num_tokens": num_tokens,
            "num_seeds": num_seeds, "num_prompts": len(prompts),
            "total_samples": len(prompts) * num_seeds,
            "exact_prompt_ids": True,
        },
        "samples": [],
        "generation_times": [],
    }

    sample_idx = 0
    total_start = time.time()

    for seed_idx in range(num_seeds):
        seed = seed_base + seed_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n[+] Seed {seed_idx+1}/{num_seeds} (seed={seed})")

        for i, prompt in enumerate(prompts):
            if i % 20 == 0:
                print(f"  Seed {seed}: {i}/{len(prompts)}")

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

            gen_start = time.time()
            generated_text, watermark_key, metadata = wm.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=num_tokens, do_sample=True,
                temperature=1.0, top_p=0.9,
                seed=seed + i,  # per-sample seed advances RNG
            )
            gen_time = time.time() - gen_start
            results["generation_times"].append(gen_time)

            # Use the exact generated token IDs from model.generate. Re-tokenizing
            # decoded text can alter token boundaries and corrupt the detection
            # statistic because Detect needs log p_theta(y | x) for the sampled y.
            generated_ids = metadata["generated_ids"]

            # Save the exact per-sample key produced by wm.generate(seed=seed+i).
            sample_seed = seed + i
            key_path = os.path.join(keys_dir, f"key_seed{sample_seed}.pt")
            torch.save(torch.from_numpy(watermark_key), key_path)

            sample = {
                "sample_idx": sample_idx,
                "prompt_idx": i,
                "seed": seed,
                "sample_seed": sample_seed,
                "prompt_text": prompt[:500],
                "prompt_ids": input_ids[0].detach().cpu().tolist(),
                "generated_text": generated_text,
                "generated_ids": generated_ids,
                "watermark_key_shape": list(watermark_key.shape),
                "watermark_key_path": key_path,
                "gen_time": gen_time,
                "num_generated_tokens": len(generated_ids),
                **metadata,
            }
            results["samples"].append(sample)
            sample_idx += 1

    total_time = time.time() - total_start
    times = results["generation_times"]

    results["_keys_dir"] = keys_dir
    results["wallclock_time"] = total_time
    results["timing_stats"] = {
        "mean": float(statistics.mean(times)),
        "median": float(statistics.median(times)),
        "total": float(sum(times)),
    }

    os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Generation done: {sample_idx} samples in {total_time:.1f}s")
    print(f"[+] Results: {results_file}")
    print(f"[+] Keys: {keys_dir}/")
    return results_file


def generate_unwatermarked(model, tokenizer, prompts, num_tokens, seed_base):
    """Generate a fresh unwatermarked baseline with the same prompts/settings."""
    results_file = os.path.join(CHECKPOINT_DIR, "unwatermarked_baseline_exact.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = json.load(f)
        if (
            existing.get("config", {}).get("exact_token_ids")
            and existing.get("config", {}).get("exact_prompt_ids")
            and len(existing.get("samples", [])) >= len(prompts)
        ):
            print(f"[+] Fresh unwatermarked baseline already exists at {results_file}, skipping generation...")
            return results_file

    print(f"\n[+] Generating fresh unwatermarked baseline ({len(prompts)} prompts)...")
    results = {
        "config": {
            "watermarked": False,
            "num_tokens": num_tokens,
            "num_prompts": len(prompts),
            "seed_base": seed_base,
            "exact_token_ids": True,
            "exact_prompt_ids": True,
        },
        "samples": [],
        "generation_times": [],
    }

    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            print(f"  Unwatermarked: {i}/{len(prompts)}")
        sample_seed = seed_base + i
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=getattr(model.config, "pad_token_id", None) or 0,
            )
        gen_time = time.time() - start
        generated_ids = outputs[0][input_ids.shape[1]:].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results["generation_times"].append(gen_time)
        results["samples"].append({
            "sample_idx": i,
            "prompt_idx": i,
            "seed": sample_seed,
            "sample_seed": sample_seed,
            "prompt_text": prompt[:500],
            "prompt_ids": input_ids[0].detach().cpu().tolist(),
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "gen_time": gen_time,
            "num_generated_tokens": len(generated_ids),
        })

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Unwatermarked baseline saved to {results_file}")
    return results_file


def run_detection(model, tokenizer, samples, config, layer_idx, weight_name, sigma, output_prefix):
    """Run GaussMark.Detect on samples."""
    print(f"\n[+] Detection config: layer={layer_idx}, weight={weight_name}, sigma={sigma}")

    wm = GaussMark(model=model, layer_idx=layer_idx, weight_name=weight_name,
                    sigma=sigma, device="cuda")

    p_values = []
    psi_values = []
    grad_norms = []
    dot_products = []
    keys_dir = config.get('_keys_dir', CHECKPOINT_DIR)

    total_start = time.time()

    # Filter samples with too few generated tokens (likely hit EOS early)
    MIN_TOKENS = 50
    filtered = [s for s in samples if len(s["generated_ids"]) >= MIN_TOKENS]
    if len(filtered) < len(samples):
        print(f"[!] Skipping {len(samples) - len(filtered)} samples with <{MIN_TOKENS} tokens")
    samples = filtered

    for i, sample in enumerate(samples):
        if i % 20 == 0:
            print(f"  Detecting: {i}/{len(samples)}")

        generated_ids = sample["generated_ids"]
        # The per-sample seed is the key used during generation.
        # We regenerate the key here rather than loading from file because
        # the saved keys may have been saved with the wrong (outer) seed.
        # sample["sample_seed"] = seed + i where i is the prompt index.
        per_sample_seed = sample.get("sample_seed", sample.get("seed", 42))
        outer_seed = sample.get("seed", 42)

        # Try to load the key from a per-sample path first.
        # If generation was done correctly, keys are saved per-sample_seed.
        key_path = sample.get("watermark_key_path")
        loaded_from_file = False
        watermark_key = None

        if key_path:
            # Try per-sample path: key_seed{sample_seed}.npy (legacy)
            npy_path = os.path.join(os.path.dirname(key_path), f"key_seed{per_sample_seed}.npy")
            # Try per-sample path: key_seed{sample_seed}.pt
            pt_path = os.path.join(os.path.dirname(key_path), f"key_seed{per_sample_seed}.pt")

            if os.path.exists(pt_path):
                watermark_key = torch.load(pt_path, weights_only=False, map_location="cpu").numpy()
                loaded_from_file = True
            elif os.path.exists(npy_path):
                npy_arr = np.load(npy_path)
                expected_elements = wm.target_param.numel()
                if npy_arr.size != expected_elements:
                    flat = npy_arr.flatten()
                    if flat.size < expected_elements:
                        flat = np.pad(flat, (0, expected_elements - flat.size), mode='constant', constant_values=0)
                    else:
                        flat = flat[:expected_elements]
                    watermark_key = flat.reshape(wm.target_shape)
                else:
                    watermark_key = npy_arr
                loaded_from_file = True
                # Convert to .pt for future use
                torch.save(torch.from_numpy(watermark_key), pt_path)

        if not loaded_from_file:
            # Regenerate key using the per-sample seed.
            # This reproduces exactly what was done during generation.
            if per_sample_seed != outer_seed:
                print(f"  NOTE: Regen key for sample {i} using per_sample_seed={per_sample_seed} (outer={outer_seed})")
            watermark_key = wm.regenerate_watermark_key(per_sample_seed)

        if "prompt_ids" in sample:
            input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            prompt_text = sample["prompt_text"]
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = prompt_enc["input_ids"]
            attention_mask = prompt_enc.get("attention_mask")
        gen_tensor = torch.tensor([generated_ids]).long()

        result = wm.detect(
            input_ids=input_ids, generated_ids=gen_tensor,
            watermark_key=watermark_key, attention_mask=attention_mask, alpha=0.05,
        )

        p_values.append(result["p_value"])
        psi_values.append(result["psi"])
        grad_norms.append(result["grad_norm"])
        dot_products.append(result["dot_product"])

    total_time = time.time() - total_start
    if not p_values:
        raise RuntimeError(
            f"No samples with at least {MIN_TOKENS} generated tokens for {output_prefix}; "
            "cannot compute detection metrics."
        )

    metrics = {
        "num_samples": len(p_values),
        "sigma": sigma,
        "median_p_value": float(statistics.median(p_values)),
        "mean_p_value": float(statistics.mean(p_values)),
        "std_p_value": float(statistics.stdev(p_values)) if len(p_values) > 1 else 0.0,
        "min_p_value": float(min(p_values)),
        "max_p_value": float(max(p_values)),
        "median_psi": float(statistics.median(psi_values)),
        "detection_rate_fpr_0.05": float(sum(1 for p in p_values if p < 0.05) / len(p_values)),
        "detection_rate_fpr_0.01": float(sum(1 for p in p_values if p < 0.01) / len(p_values)),
        "num_detected_fpr_0.05": int(sum(1 for p in p_values if p < 0.05)),
        "num_detected_fpr_0.01": int(sum(1 for p in p_values if p < 0.01)),
        "total_time": total_time,
        "mean_grad_norm": float(statistics.mean(grad_norms)),
        "median_grad_norm": float(statistics.median(grad_norms)),
        "mean_dot_product": float(statistics.mean(dot_products)),
        "p_values": [float(p) for p in p_values],
        "psi_values": [float(psi) for psi in psi_values],
    }

    # Save full results
    full_path = f"{output_prefix}_full.json"
    with open(full_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[+] Detection results ({len(p_values)} samples, {total_time:.1f}s):")
    print(f"    Median p-value: {metrics['median_p_value']:.6f}")
    print(f"    Detection @FPR=0.05: {metrics['detection_rate_fpr_0.05']*100:.1f}% ({metrics['num_detected_fpr_0.05']}/{len(p_values)})")
    print(f"    Detection @FPR=0.01: {metrics['detection_rate_fpr_0.01']*100:.1f}% ({metrics['num_detected_fpr_0.01']}/{len(p_values)})")
    print(f"    Mean dot product: {metrics['mean_dot_product']:.6f}")
    print(f"    Mean grad norm: {metrics['mean_grad_norm']:.4f}")
    print(f"[+] Saved to {full_path}")
    return metrics


def compute_c4_cross_entropy(model, tokenizer, prompts, layer_idx, weight_name, sigma):
    """Compute a lightweight C4 CE quality constraint before/after watermarking."""
    texts = prompts[:QUALITY_PROMPTS]
    print(f"\n[+] Computing C4 cross-entropy quality check on {len(texts)} prompts...")

    def mean_ce():
        losses = []
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                losses.append(float(out.loss.item()))
        return float(statistics.mean(losses))

    wm = GaussMark(model=model, layer_idx=layer_idx, weight_name=weight_name,
                   sigma=sigma, device="cuda")
    base_ce = mean_ce()

    key = wm.regenerate_watermark_key(SEED_BASE)
    key_tensor = torch.from_numpy(key).to(model.device, dtype=wm.target_param.dtype)
    original_data = wm.target_param.data.clone()
    wm.target_param.data = (wm.target_param.data + key_tensor).to(dtype=wm.target_param.dtype)
    try:
        watermarked_ce = mean_ce()
    finally:
        wm.target_param.data = original_data

    metrics = {
        "c4_cross_entropy_unwatermarked": base_ce,
        "c4_cross_entropy_watermarked": watermarked_ce,
        "c4_cross_entropy_delta": watermarked_ce - base_ce,
        "c4_cross_entropy_ratio": watermarked_ce / base_ce if base_ce > 0 else float("inf"),
    }
    print(f"    Unwatermarked CE: {base_ce:.4f}")
    print(f"    Watermarked CE:   {watermarked_ce:.4f}")
    print(f"    CE ratio:         {metrics['c4_cross_entropy_ratio']:.4f}")
    return metrics


def main():
    print("=" * 60)
    print("  GaussMark Reproduction Pipeline")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.tokenizer = tokenizer

    # Load C4 prompts
    prompts = load_c4_prompts(num_prompts=NUM_PROMPTS, tokenizer=tokenizer)

    # Step 1: Generate watermarked text with sigma=sqrt(1e-5)
    cfg_name = f"wm_l{LAYER_IDX}_{WEIGHT_NAME}_s{SIGMA}".replace('.', '')
    wm_results_file = generate_watermarked(
        model=model, tokenizer=tokenizer, prompts=prompts,
        layer_idx=LAYER_IDX, weight_name=WEIGHT_NAME, sigma=SIGMA,
        num_tokens=NUM_TOKENS, num_seeds=NUM_SEEDS, seed_base=SEED_BASE,
        output_path=os.path.join(CHECKPOINT_DIR, f"watermarked_{cfg_name}.json"),
    )

    # Load generated results
    with open(wm_results_file) as f:
        wm_data = json.load(f)
    wm_samples = wm_data["samples"]
    print(f"\n[+] Loaded {len(wm_samples)} watermarked samples")

    # Step 2: Run detection on watermarked text
    wm_metrics = run_detection(
        model=model, tokenizer=tokenizer, samples=wm_samples,
        config=wm_data["config"], layer_idx=LAYER_IDX,
        weight_name=WEIGHT_NAME, sigma=SIGMA,
        output_prefix=os.path.join(CHECKPOINT_DIR, f"wm_{cfg_name}_detection"),
    )

    # Step 3: Generate and detect a fresh unwatermarked baseline in this run.
    uw_results_file = generate_unwatermarked(
        model=model, tokenizer=tokenizer, prompts=prompts,
        num_tokens=NUM_TOKENS, seed_base=SEED_BASE,
    )
    with open(uw_results_file) as f:
        uw_data = json.load(f)
    uw_metrics = run_detection(
        model=model, tokenizer=tokenizer, samples=uw_data["samples"],
        config={"_keys_dir": None}, layer_idx=LAYER_IDX,
        weight_name=WEIGHT_NAME, sigma=SIGMA,
        output_prefix=os.path.join(CHECKPOINT_DIR, "unwatermarked_baseline_unwatermarked_detection"),
    )

    # Step 4: Compute AUC using both watermarked p-values and unwatermarked p-values
    if wm_metrics["p_values"] and uw_metrics.get("p_values"):
        p_wm = np.array(wm_metrics["p_values"])
        p_uw = np.array(uw_metrics["p_values"])
        labels = np.concatenate([np.ones(len(p_wm)), np.zeros(len(p_uw))])
        scores = np.concatenate([p_wm, p_uw])
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, -scores)
            wm_metrics["auc"] = float(auc)
            uw_metrics["auc"] = float(auc)
            print(f"\n[+] ROC AUC: {auc:.4f}")
        except:
            pass

    # Step 5: Compute a lightweight no-quality-loss constraint.
    quality_metrics = compute_c4_cross_entropy(
        model=model, tokenizer=tokenizer, prompts=prompts,
        layer_idx=LAYER_IDX, weight_name=WEIGHT_NAME, sigma=SIGMA,
    )
    wm_metrics.update(quality_metrics)
    detection_full_path = os.path.join(CHECKPOINT_DIR, f"wm_{cfg_name}_detection_full.json")
    if os.path.exists(detection_full_path):
        with open(detection_full_path) as f:
            detection_full = json.load(f)
        detection_full.update(quality_metrics)
        with open(detection_full_path, "w") as f:
            json.dump(detection_full, f, indent=2)

    # Step 6: Write scores.json
    wm_generation_time = float(statistics.mean(wm_data.get("generation_times", [1.0])))
    uw_generation_time = float(statistics.mean(uw_data.get("generation_times", [wm_generation_time])))
    generation_time_ratio = wm_generation_time / uw_generation_time if uw_generation_time > 0 else float("inf")

    scores = {
        "experiments": {
            "watermark_detection_mistral": {
                "results": {
                    "gaussmark_200_tokens": {
                        "detection_rate_fpr_0.05": wm_metrics["detection_rate_fpr_0.05"],
                        "detection_rate_fpr_0.01": wm_metrics["detection_rate_fpr_0.01"],
                        "median_p_value": wm_metrics["median_p_value"],
                        "auc": wm_metrics.get("auc", 0.5),
                        "c4_cross_entropy_ratio": quality_metrics["c4_cross_entropy_ratio"],
                        "c4_cross_entropy_delta": quality_metrics["c4_cross_entropy_delta"],
                        "generation_time_ratio": generation_time_ratio,
                    },
                    "unwatermarked": {
                        "detection_rate_fpr_0.05": uw_metrics["detection_rate_fpr_0.05"],
                        "detection_rate_fpr_0.01": uw_metrics["detection_rate_fpr_0.01"],
                        "median_p_value": uw_metrics["median_p_value"],
                        "c4_cross_entropy_ratio": 1.0,
                        "c4_cross_entropy_delta": 0.0,
                        "generation_time_ratio": 1.0,
                    },
                }
            }
        }
    }

    scores_path = "/home/user/scoring/scores.json"
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"\n[+] Scores written to {scores_path}")
    print(json.dumps(scores, indent=2))

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
