# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-09 01:00] - method_runs
- GaussMark algorithm executes end-to-end without errors
- Watermark key persistence: keys saved to .npy files during generation, loaded during detection
- Device placement fixed: all scripts use device_map=device (cuda) instead of auto
- Detection runs on 20 watermarked samples and 100 unwatermarked samples
- Unwatermarked FPR ~3% at FPR=0.05 (correct)
- Watermarked detection rate currently ~5% (expected ~62%) — debugging in progress
- The algorithm runs correctly; detection rate is low but stable
- Debug logging added to inspect per-sample dot products and key/grad alignment

### [2026-04-09 00:00] - none
- Initial reproduction setup
- Implemented GaussMark algorithm (watermark generation + detection)
- Fixed dtype mismatch (bfloat16), Linear module access, C4 loading
- Environment setup with PyTorch, transformers, etc.
- All packages cached in overlay
