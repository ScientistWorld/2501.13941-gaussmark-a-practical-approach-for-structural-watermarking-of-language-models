"""
GaussMark: A Practical Approach for Structural Watermarking of Language Models.

This module implements the GaussMark watermarking scheme as described in:
Block, Rakhlin, Sekhari (2025) - GaussMark: A Practical Approach for Structural
Watermarking of Language Models

The scheme watermarks LLM outputs by adding Gaussian noise to model weights.
Detection uses a gradient-based test statistic that follows a standard normal
distribution under the null hypothesis (H0: text is NOT watermarked).

Key equations from the paper:
  - GaussMark.Generate: sample xi ~ N(0, sigma^2 * I), perturb theta(xi) = theta + xi,
    generate y ~ p_theta(xi)(cdot|x)
  - GaussMark.Detect: psi = <xi, grad_theta log p_theta(y|x)> / (sigma * ||grad_theta log p_theta(y|x)||)
    p_value = 1 - Phi(psi) where Phi is standard normal CDF

For Mistral-7B: layer 20, up_proj, sigma=1e-5 (from paper Table 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Any


class GaussMark:
    """
    Implements GaussMark.Generate and GaussMark.Detect for structural watermarking
    of transformer-based language models.

    The watermarking scheme:
      - GENERATE: Adds Gaussian noise to ONE MLP layer's weights, then generates text.
        The noise key xi ~ N(0, sigma^2 * I) is the secret watermark key.
      - DETECT:  Computes the gradient-based test statistic against the ORIGINAL
        (unwatermarked) model parameters and tests for correlation with xi.
        Under H0 (text is NOT from the watermarked model), psi ~ N(0, 1).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 20,
        weight_name: str = "up_proj",
        sigma: float = 1e-5,
        device: str = "cuda",
    ):
        """
        Initialize GaussMark watermarker.

        Args:
            model:        HuggingFace transformer model (e.g., MistralForCausalLM).
            layer_idx:    0-indexed transformer layer to watermark. Default: 20
                          (paper uses layer 20 for Mistral-7B).
            weight_name:  MLP weight to perturb. Options: 'up_proj', 'down_proj',
                          'gate_proj'. Default: 'up_proj' (per paper Table 2).
            sigma:        Standard deviation of Gaussian noise (sqrt of variance).
                          Paper Table 2 uses 1e-5 for Mistral-7B.
            device:       Device for computation. Default: 'cuda'.
        """
        self.model = model
        self.layer_idx = layer_idx
        self.weight_name = weight_name
        self.sigma = sigma
        self.device = device

        # Locate and cache the target parameter
        self.target_param = self._find_target_param()
        self.target_shape = tuple(self.target_param.shape)
        self.target_shape_np = np.array(self.target_shape)

    def _find_target_param(self) -> nn.Parameter:
        """
        Locate the target parameter in the model.

        For Mistral/Llama architectures, MLP layers are accessible as:
            model.model.layers[layer_idx].mlp.{weight_name}

        Returns:
            The nn.Parameter to perturb.

        Raises:
            AttributeError: If the parameter is not found at the expected location.
        """
        # Support both 'model.layers' and 'model.transformer.h' naming conventions
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_container = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers_container = self.model.transformer.h
        else:
            raise AttributeError(
                f"Cannot locate transformer layers in model. "
                f"Expected 'model.model.layers' or 'model.transformer.h'. "
                f"Got: {[attr for attr in dir(self.model) if not attr.startswith('_')]}"
            )

        layer = layers_container[self.layer_idx]

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward
        else:
            raise AttributeError(
                f"Layer {self.layer_idx} has no 'mlp' or 'feed_forward' attribute. "
                f"Available: {[attr for attr in dir(layer) if not attr.startswith('_')]}"
            )

        if not hasattr(mlp, self.weight_name):
            raise AttributeError(
                f"MLP has no weight '{self.weight_name}'. "
                f"Available weights: "
                f"{[attr for attr in dir(mlp) if not attr.startswith('_') and not callable(getattr(mlp, attr))]}"
            )

        module = getattr(mlp, self.weight_name)
        # If it's a Linear module, return its .weight parameter, not the module itself
        if isinstance(module, nn.Linear):
            return module.weight
        return module

    def _seed_rng(self, seed: Optional[int]) -> None:
        """Set random seeds for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # GaussMark.Generate
    # -------------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        GaussMark.Generate: Generate watermarked text.

        Algorithm:
          1. Sample watermark key xi ~ N(0, sigma^2 * I)
          2. Perturb model: theta(xi) = theta + xi  (only to target MLP layer)
          3. Generate text y ~ p_theta(xi)(cdot|x) using autoregressive sampling
          4. Restore original weights

        Args:
            input_ids:      Token IDs of the input prompt [1, prompt_len].
            attention_mask:  Attention mask for input tokens.
            max_new_tokens: Maximum number of tokens to generate. Default: 200.
            do_sample:      Whether to sample (True) or use greedy decoding (False).
            temperature:    Sampling temperature. Default: 1.0.
            top_p:          Nucleus sampling threshold. Default: 1.0 (disabled).
            seed:           Random seed for reproducibility.

        Returns:
            generated_text:  The generated text string.
            watermark_key:   The Gaussian watermark key (numpy array, shape matches
                            the perturbed weight matrix).
            metadata:        Dict with generation info (shape, sigma, layer, etc.).
        """
        self._seed_rng(seed)

        # Step 1: Sample watermark key xi ~ N(0, sigma^2 * I)
        watermark_key = np.random.normal(0, self.sigma, self.target_shape).astype(np.float32)
        watermark_key_tensor = torch.from_numpy(watermark_key).to(
            self.device, dtype=self.target_param.dtype
        )

        # Step 2: Perturb the target layer's weights
        original_data = self.target_param.data.clone()
        self.target_param.data = (self.target_param.data + watermark_key_tensor).to(
            dtype=self.target_param.dtype
        )

        try:
            # Step 3: Generate using the perturbed model
            input_ids_dev = input_ids.to(self.device)
            attn_dev = attention_mask.to(self.device) if attention_mask is not None else None

            if hasattr(self.model, "generate"):
                outputs = self.model.generate(
                    input_ids_dev,
                    attention_mask=attn_dev,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=getattr(self.model.config, "pad_token_id", None) or 0,
                )
            else:
                outputs = self._autoregressive_generate(
                    input_ids_dev, attn_dev,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )

            # Extract newly generated token IDs (everything after the prompt)
            prompt_len = input_ids.shape[1]
            generated_ids = outputs[0][prompt_len:]

            # Decode to text
            if hasattr(self.model, "tokenizer"):
                generated_text = self.model.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
            else:
                generated_text = str(generated_ids)

            metadata = {
                "watermark_key_shape": list(self.target_shape),
                "sigma": self.sigma,
                "sigma_sq": float(self.sigma ** 2),
                "layer_idx": self.layer_idx,
                "weight_name": self.weight_name,
                "num_generated_tokens": len(generated_ids),
                "prompt_length": prompt_len,
            }

        finally:
            # Step 4: Restore original (unwatermarked) weights
            self.target_param.data = original_data

        return generated_text, watermark_key, metadata

    def _autoregressive_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """
        Manual autoregressive generation fallback.

        Used when the model does not have a .generate() method.
        Implements standard greedy or nucleus sampling.
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        past_kv = None
        eos_id = getattr(self.model.config, "eos_token_id", None)

        for _ in range(max_new_tokens):
            model_input = generated[:, -1:]

            with torch.no_grad():
                out = self.model(
                    model_input,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            logits = out.logits[:, -1, :] / temperature
            past_kv = out.past_key_values

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    probs = probs.masked_fill(mask.scatter(1, sorted_indices, mask), 0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated

    # -------------------------------------------------------------------------
    # GaussMark.Detect
    # -------------------------------------------------------------------------

    def detect(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        watermark_key: np.ndarray,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        GaussMark.Detect: Detect whether text is watermarked.

        Computes the test statistic from the paper (Equation 1):

            psi = <xi, grad_theta log p_theta(y|x)>
                  --------------------------------
                   sigma * ||grad_theta log p_theta(y|x)||

        Under H0 (text is NOT from the watermarked model), xi and y are independent,
        so psi ~ N(0, 1). The p-value is 1 - Phi(psi).

        IMPORTANT: Detection uses the ORIGINAL (unwatermarked) model's gradient.
        The watermark key xi must be the exact key used during generation.

        Args:
            input_ids:      Token IDs of the input prompt [1, prompt_len].
            generated_ids:   Token IDs of the text to check [1, gen_len].
            watermark_key:   The Gaussian key used during generation (numpy array,
                            same shape as the perturbed weight matrix).
            attention_mask:  Attention mask for input tokens.
            alpha:           Significance threshold for "watermarked" label.

        Returns:
            Dict with keys:
                - psi:         Test statistic (should be ~N(0,1) under H0)
                - p_value:     1 - Phi(psi), small values indicate watermarking
                - grad_norm:   ||grad_theta log p_theta(y|x)||
                - dot_product: <xi, grad>
                - watermarked: True if p_value < alpha
        """
        # Move to device
        input_ids = input_ids.to(self.device)
        generated_ids = generated_ids.to(self.device)
        xi = torch.from_numpy(watermark_key).to(self.device).float().flatten()

        # Concatenate prompt + generated to form the full sequence
        full_ids = torch.cat([input_ids, generated_ids], dim=-1)  # [1, L+T]
        prompt_len = input_ids.shape[1]
        gen_len = generated_ids.shape[1]

        # Build extended attention mask (all 1s for causal model)
        if attention_mask is not None:
            attn_mask = attention_mask.to(self.device)
            # Extend with ones for generated tokens
            gen_mask = torch.ones(
                1, gen_len, dtype=attn_mask.dtype, device=self.device
            )
            full_attn_mask = torch.cat([attn_mask, gen_mask], dim=-1)
        else:
            full_attn_mask = torch.ones_like(full_ids)

        # -------------------------------------------------------------------------
        # Forward pass with gradient tracking on the target parameter
        # -------------------------------------------------------------------------
        self.model.eval()  # Ensure dropout is off
        self.target_param.requires_grad_(True)

        outputs = self.model(full_ids, attention_mask=full_attn_mask)
        logits = outputs.logits  # [1, L+T, vocab_size]

        # -------------------------------------------------------------------------
        # Compute log p_theta(y|x) = sum_t log p_theta(y_t | x, y_{<t})
        #
        # In causal models, logits[:, t, :] predicts full_ids[:, t+1].
        # For generated tokens at positions [L, L+gen_len-1] (0-indexed),
        # the logits at [L-1, L+gen_len-2] predict [L, L+gen_len-1].
        #
        # So we need:
        #   log_probs_at_pos_j  = log p_theta(full_ids[:, L+j] | full_ids[:, :L+j])
        #                       = logits[:, L+j-1, full_ids[:, L+j]]
        # for j in [0, gen_len-1].
        # -------------------------------------------------------------------------

        # logits for predicting each generated token
        gen_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
        # gen_logits: [1, gen_len, vocab_size]

        # Log softmax then gather the log prob of each actual token
        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=2, index=generated_ids.unsqueeze(2)
        ).squeeze(2)  # [1, gen_len]

        # Sum over all generated tokens: log p_theta(y|x)
        total_log_prob = token_log_probs.sum()

        # -------------------------------------------------------------------------
        # Backward pass: grad = d(log p) / d(target_param)
        # This gradient is computed w.r.t. the ORIGINAL (unwatermarked) model,
        # because we restored the original weights after generation.
        # -------------------------------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_log_prob,
            inputs=self.target_param,
            retain_graph=False,
        )[0]  # Same shape as the weight matrix

        grad_flat = grad.flatten().float()
        grad_norm = torch.norm(grad_flat)

        # -------------------------------------------------------------------------
        # Compute test statistic: psi = <xi, grad> / (sigma * ||grad||)
        # -------------------------------------------------------------------------
        dot_product = torch.dot(xi, grad_flat)
        sigma_grad_norm = self.sigma * grad_norm

        if sigma_grad_norm.item() == 0:
            psi = 0.0
        else:
            psi = (dot_product / sigma_grad_norm).item()

        # p_value = 1 - Phi(psi)  [upper tail]
        p_value = float(1.0 - norm.cdf(psi))

        # Restore requires_grad
        self.target_param.requires_grad_(False)

        return {
            "psi": psi,
            "p_value": p_value,
            "grad_norm": grad_norm.item(),
            "dot_product": dot_product.item(),
            "watermarked": p_value < alpha,
            "watermarked_strict": p_value < 0.01,
            "alpha": alpha,
            "gen_len": gen_len,
        }

    def detect_batch(
        self,
        input_ids_list: List[torch.Tensor],
        generated_ids_list: List[torch.Tensor],
        watermark_keys: List[np.ndarray],
        attention_masks: Optional[List[torch.Tensor]] = None,
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Batch detection. Processes samples sequentially.

        Args:
            input_ids_list:  List of prompt token ID tensors.
            generated_ids_list: List of generated token ID tensors.
            watermark_keys:   List of watermark keys (numpy arrays).
            attention_masks: Optional list of attention masks.
            alpha:            Significance threshold.

        Returns:
            List of detection result dicts (one per sample).
        """
        results = []
        for i, (inp, gen, key) in enumerate(
            zip(input_ids_list, generated_ids_list, watermark_keys)
        ):
            mask = attention_masks[i] if attention_masks else None
            results.append(self.detect(inp, gen, key, mask, alpha))
        return results

    def regenerate_watermark_key(self, seed: int) -> np.ndarray:
        """
        Regenerate a watermark key deterministically from a seed.

        This is useful for detection when the key is not directly stored.
        The same seed + sigma produces the same key.

        IMPORTANT: Pass the OUTER seed (42 + seed_idx), NOT the per-prompt seed.
        The outer seed is what was used during generation before the per-prompt
        RNG advancement. The per-prompt seed (42 + seed_idx + prompt_idx) produces
        a DIFFERENT key because the RNG state was advanced by prompt_idx steps.

        Args:
            seed: The outer random seed (42 + seed_idx). NOT the per-prompt seed.

        Returns:
            Watermark key as numpy array (same shape as target parameter).
        """
        rng = np.random.RandomState(seed)
        return rng.normal(0, self.sigma, self.target_shape).astype(np.float32)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def compute_p_value(psi: float) -> float:
    """
    Compute the one-sided upper-tail p-value from the test statistic psi.

    Under H0, psi ~ N(0, 1), so p = 1 - Phi(psi).
    Small p-values (< alpha) indicate the text is likely watermarked.

    Args:
        psi: Test statistic from GaussMark.Detect.

    Returns:
        p-value in (0, 1). Lower = more evidence for watermarking.
    """
    return float(1.0 - norm.cdf(psi))


def compute_detection_metrics(
    p_values: List[float],
    thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Compute detection metrics from a list of p-values.

    Args:
        p_values:  List of p-values from GaussMark.Detect.
        thresholds: FPR thresholds to evaluate TPR at. Default: [0.05, 0.01].

    Returns:
        Dict with detection rates, counts, and summary statistics.
    """
    import statistics

    if thresholds is None:
        thresholds = [0.05, 0.01]

    p_arr = np.array(p_values)
    metrics = {
        "num_samples": len(p_arr),
        "median_p_value": float(statistics.median(p_arr)),
        "mean_p_value": float(np.mean(p_arr)),
        "std_p_value": float(np.std(p_arr)),
        "min_p_value": float(np.min(p_arr)),
        "max_p_value": float(np.max(p_arr)),
        "median_psi": float(statistics.median([norm.ppf(1 - p) for p in p_arr])),
    }

    for thresh in thresholds:
        n_detected = int(np.sum(p_arr < thresh))
        metrics[f"detection_rate_fpr_{thresh}"] = float(n_detected / len(p_arr))
        metrics[f"num_detected_fpr_{thresh}"] = n_detected

    return metrics
