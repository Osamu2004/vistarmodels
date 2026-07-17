# Experiment Log: 2026-07-17

## Goal of This Week

Complete the corrected FLUX.1 Fill-dev zero-shot SECOND baseline under the
shared one-class protocol without changing its generation semantics.

## Experiment 1: Remove repeated per-sample model/text transfers

### Purpose

Diagnose and reduce the long pause at every sample boundary while preserving
the 512-to-256 resolution protocol, 50 sampling steps, guidance 30, masks,
prompts, and deterministic seeds.

### Setting

- Data: SECOND test, both directions, 3,388 shared one-class records.
- Observed run: default model CPU offload, BF16, one sample at a time.
- Failure evidence: a 50-step sample required about 85 seconds; after eight
  previously saved records were skipped, the outer loop reached row 9 at about
  2:07 and paused again before the next 50-step loop.

### Results

- Code audit: the old default always enabled model CPU offload.
- Code audit: every generated record passed one of only six repeated captions
  back through both FLUX text encoders, forcing repeated CLIP/T5 execution and
  module transitions before denoising.
- Implemented: cache all unique prompt embeddings once; default to full-CUDA
  residency so the 12B Transformer remains resident across samples; preserve
  explicit CPU-offload and automatic-placement fallbacks; skip model allocation
  when all predictions exist.
- Added runtime evidence: `runtime_config.json`, per-row generation seconds,
  placement, resolved offload state, and prompt-cache hit.
- Static verification: Python byte compilation, Bash syntax, targeted
  whitespace checks, CLI option discovery, and a mocked three-prompt batch
  cache test pass. Official Diffusers v0.32.0 source confirms that
  `FluxFillPipeline.encode_prompt` returns prompt/pooled embeddings and that the
  pipeline accepts both precomputed tensors.

### Analysis

- Primary cause: `enable_model_cpu_offload()` rotates VAE, text encoders, and
  Transformer between host and GPU memory. VAE decoding at the end of one
  sample offloads the Transformer, so the next sample reloads it. Repeated text
  encoding adds another avoidable boundary cost. Image loading and manifest
  construction are not the main bottlenecks.
- Secondary cost: the 12B Transformer still performs 50 denoising evaluations;
  this cost is intentionally retained for protocol comparability.
- The optimization does not change prompts, masks, source images, sampling
  steps, guidance, seeds, output size, or evaluator inputs.

### Next Steps

- [ ] Pull the fix and resume the same output directory; existing PNGs skip.
- [ ] Confirm `placement=full_cuda` and `cached 6 unique prompts` in the log.
- [ ] Compare the new `generation_seconds` against the observed 85 seconds.
- [ ] If CUDA OOM occurs, use `CPU_OFFLOAD=1`; prompt caching still removes the
      repeated text-encoder boundary overhead.
- [ ] Run the complete common SECOND metric suite after all 3,388 outputs exist.
