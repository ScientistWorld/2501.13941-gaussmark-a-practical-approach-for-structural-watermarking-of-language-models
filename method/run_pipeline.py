#!/usr/bin/env python3
"""
GaussMark end-to-end pipeline for reproduction.

This script runs the full pipeline:
1. Generate watermarked text with sigma=1e-5 (paper's parameter)
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

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, '/home/user/method')
from gaussmark import GaussMark

MODEL_PATH = "/home/user/models/mistralai/Mistral-7B-Instruct-v0.3"
CHECKPOINT_DIR = "/home/user/checkpoints"
C4_DATA_PATH = "/home/user/data/c4_realnewslike"

# Paper parameters for Mistral-7B (from Table 2)
LAYER_IDX = 20
WEIGHT_NAME = "up_proj"
SIGMA = 1e-5  # Paper's sigma for Mistral-7B (primary experiment)
NUM_TOKENS = 200
NUM_PROMPTS = 50   # 50 prompts × 1 seed = 50 samples
NUM_SEEDS = 1
SEED_BASE = 42


def load_c4_prompts(num_prompts=100, max_words=200):
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
            words = text.split()
            truncated = ' '.join(words[:max_words])
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
                    words = text.split()
                    truncated = ' '.join(words[:max_words])
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
        if len(existing.get('samples', [])) >= NUM_PROMPTS * NUM_SEEDS:
            print(f"[+] Generation already exists at {results_file}, skipping...")
            return results_file

    results = {
        "config": {
            "layer_idx": layer_idx, "weight_name": weight_name,
            "sigma": sigma, "num_tokens": num_tokens,
            "num_seeds": num_seeds, "num_prompts": len(prompts),
            "total_samples": len(prompts) * num_seeds,
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

            # Tokenize generated text
            gen_enc = tokenizer(generated_text, return_tensors="pt", add_special_tokens=False)
            generated_ids = gen_enc["input_ids"][0].cpu().tolist()

            # Save watermark key to file with path that includes seed
            # Key is generated with outer seed (seed) for this seed_idx
            key_path = os.path.join(keys_dir, f"key_seed{seed}.npy")
            np.save(key_path, watermark_key)

            sample = {
                "sample_idx": sample_idx,
                "prompt_idx": i,
                "seed": seed,
                "sample_seed": seed + i,
                "prompt_text": prompt[:500],
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

    for i, sample in enumerate(samples):
        if i % 20 == 0:
            print(f"  Detecting: {i}/{len(samples)}")

        prompt_text = sample["prompt_text"]
        generated_ids = sample["generated_ids"]
        seed = sample.get("seed", 42)

        # Load watermark key from saved file
        key_path = sample.get("watermark_key_path")
        if key_path and os.path.exists(key_path):
            watermark_key = np.load(key_path)
        else:
            # Fallback: regenerate
            print(f"  WARNING: Key not found at {key_path}, regenerating...")
            shape = tuple(sample.get("watermark_key_shape", config.get("watermark_key_shape", [])))
            rng = np.random.RandomState(int(seed))
            watermark_key = rng.normal(0, sigma, shape).astype(np.float32)

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


def load_unwatermarked_baseline(num_samples=100):
    """Load or generate unwatermarked baseline samples."""
    baseline_path = os.path.join(CHECKPOINT_DIR, "unwatermarked_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            d = json.load(f)
        samples = d.get("samples", d) if isinstance(d, dict) else d
        # Check if samples have enough generated tokens
        if samples and len(samples[0].get("generated_ids", [])) >= 50:
            print(f"[+] Loading existing unwatermarked baseline ({len(samples)} samples)")
            return samples
        else:
            print(f"[+] Existing baseline has only {len(samples[0].get('generated_ids', []))} tokens, regenerating...")

    print(f"[+] Generating unwatermarked baseline ({num_samples} samples)...")
    return None


def main():
    print("=" * 60)
    print("  GaussMark Reproduction Pipeline")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.tokenizer = tokenizer

    # Load C4 prompts
    prompts = load_c4_prompts(num_prompts=NUM_PROMPTS)

    # Step 1: Generate watermarked text with sigma=1e-5
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

    # Step 3: Ensure unwatermarked baseline exists
    uw_samples = load_unwatermarked_baseline()
    if uw_samples is None:
        print(f"\n[+] Generating unwatermarked baseline (100 samples)...")
        uw_results = {"samples": [], "generation_times": []}
        uw_prompts = prompts[:100] if len(prompts) >= 100 else prompts * (100 // len(prompts) + 1)
        uw_prompts = uw_prompts[:100]
        for i, prompt in enumerate(uw_prompts):
            if i % 20 == 0:
                print(f"  Unwatermarked: {i}/100")
            # Set seeds for reproducibility
            uw_seed = 1000 + i
            np.random.seed(uw_seed)
            torch.manual_seed(uw_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(uw_seed)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=NUM_TOKENS, do_sample=True,
                    temperature=1.0, top_p=0.9,
                )
            prompt_len = input_ids.shape[1]
            generated_ids = outputs[0][prompt_len:].cpu().tolist()
            uw_results["samples"].append({
                "prompt_idx": i,
                "seed": uw_seed,
                "prompt_text": prompt[:500],
                "generated_ids": generated_ids,
                "num_generated_tokens": len(generated_ids),
            })
        baseline_path = os.path.join(CHECKPOINT_DIR, "unwatermarked_baseline.json")
        with open(baseline_path, 'w') as f:
            json.dump(uw_results, f, indent=2)
        uw_samples = uw_results["samples"]
        print(f"[+] Saved unwatermarked baseline to {baseline_path}")

    print(f"\n[+] Loaded {len(uw_samples)} unwatermarked samples")

    # Step 4: Run detection on unwatermarked text
    uw_config = {"_keys_dir": None}
    uw_metrics = run_detection(
        model=model, tokenizer=tokenizer, samples=uw_samples,
        config=uw_config, layer_idx=LAYER_IDX,
        weight_name=WEIGHT_NAME, sigma=SIGMA,
        output_prefix=os.path.join(CHECKPOINT_DIR, "uw_baseline_detection"),
    )

    # Step 5: Compute AUC
    if wm_metrics["p_values"] and uw_metrics["p_values"]:
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

    # Step 6: Write scores.json
    scores = {
        "watermark_detection_mistral": {
            "results": {
                "gaussmark_200_tokens": {
                    "detection_rate_fpr_0.05": wm_metrics["detection_rate_fpr_0.05"],
                    "detection_rate_fpr_0.01": wm_metrics["detection_rate_fpr_0.01"],
                    "median_p_value": wm_metrics["median_p_value"],
                    "auc": wm_metrics.get("auc", 0.5),
                },
                "unwatermarked": {
                    "detection_rate_fpr_0.05": uw_metrics["detection_rate_fpr_0.05"],
                    "detection_rate_fpr_0.01": uw_metrics["detection_rate_fpr_0.01"],
                    "median_p_value": uw_metrics["median_p_value"],
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
