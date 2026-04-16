# Dense 27B/31B NVFP4 Comparative Benchmark

**Four dense NVFP4 models — two architectures, two quantization sources — tested head-to-head on NVIDIA RTX PRO 6000 Blackwell with FP8 KV cache at 128K context.**

## Models Tested

| Model | Params | Source | Size | KV Cache | Port |
|-------|--------|--------|------|----------|------|
| **Qwen3.5-27B** | 27B | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-Qwen3.5-27B-abliterated-NVFP4) | 20.6 GB | FP8 | 8016 |
| **Qwopus3.5-27B** | 27B | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-Qwopus3.5-27B-v3-abliterated-NVFP4) | 19.8 GB | FP8 | 8017 |
| **Gemma4-31B** | 31B | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4) | 20.5 GB | FP8 | 8018 |
| **Gemma4-31B (lyf)** | 31B | [Community](https://huggingface.co/lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4) | 20.5 GB | FP16 (baseline) | 8019 |

All models are abliterated (uncensored) NVFP4. First 3 use `--kv-cache-dtype fp8` for 2x KV cache efficiency; lyf baseline uses default FP16 KV.

## Hardware & Configuration

| | |
|---|---|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell (96 GB) — 1 GPU per model |
| **Context** | 128K tokens |
| **KV Cache** | FP8 (Lna-Lab models) / FP16 (lyf baseline) |
| **CUDA Graph** | PIECEWISE mode |
| **Framework** | vLLM 0.19.1rc1 nightly (cu130) |

## Quality Results (128K Context, FP8 KV)

![Quality Comparison](figures/01_quality_comparison.png)

### Scores by Test (Concurrency = 1)

| Test | Qwen3.5-27B | Qwopus-27B | Gemma4-31B | Gemma4-31B (lyf) | Winner |
|------|:-----------:|:----------:|:----------:|:----------------:|--------|
| **English Critique** | **0.96** | 0.64 | **0.98** | **0.98** | Gemma4 |
| **Japanese** | **0.90** | 0.78 | 0.74 | 0.74 | Qwen3.5 |
| **Math Reasoning** | 0.73 | **0.90** | 0.65 | 0.70 | Qwopus |
| **Coding** | 0.85 | **0.90** | **0.90** | **0.90** | Tie (Qwopus/Gemma4) |
| **System Design** | 0.79 | 0.70 | 0.80 | 0.80 | Gemma4 |

> **FP8 KV cache introduces no quality degradation.** Scores are comparable to or better than FP16 KV baseline, demonstrating that 8-bit KV compression is lossless for practical purposes.

![Radar Profile](figures/06_radar_profile.png)

### Key Findings

1. **Gemma4-31B dominates English** (0.98) — deep 31B reasoning produces the best prose.
2. **Qwopus dominates Math** (0.90) — Opus distillation gives the sharpest chain-of-thought.
3. **Qwen3.5-27B leads Japanese** (0.90) — strongest multilingual performance at 128K context.
4. **Coding is a 3-way tie** (0.90) — Qwopus, Gemma4 Lna-Lab, and Gemma4 lyf all equal.
5. **FP8 KV is free** — no quality penalty vs FP16, but 2x more KV cache capacity.

## Speed Results (128K Context, FP8 KV)

![Per-Request Speed](figures/02_speed_per_request.png)

### Per-Request Speed (tok/s)

| Model | x1 | x4 (per-req) | x4 (aggregate) | Scaling |
|-------|:--:|:------------:|:--------------:|:-------:|
| Qwen3.5-27B | 57 | 57 | **461** | **8.1x** |
| Qwopus-27B | 58 | 58 | **465** | **8.0x** |
| Gemma4-31B | 50 | 39 | 318 | 6.4x |
| Gemma4-31B (lyf) | 50 | 38 | 312 | 6.2x |

**Qwen3.5 with FP8 KV maintains full speed at 4x concurrency** — per-request drops from 57 to 57 tok/s (essentially zero degradation). Gemma4 drops from 50 to 39 tok/s. MLA + FP8 KV is a powerful combination.

![Throughput Scaling](figures/03_throughput_scaling.png)

## VRAM Usage @ 128K Context

| GPU | Model | KV Cache | VRAM Used | KV Slots (est.) |
|:---:|-------|:--------:|:---------:|:---------------:|
| 0 | Qwen3.5-27B | **FP8** | 92,470 MB | ~2x baseline |
| 1 | Qwopus-27B | **FP8** | 92,470 MB | ~2x baseline |
| 2 | Gemma4-31B | **FP8** | 94,208 MB | ~2x baseline |
| 3 | Gemma4-31B (lyf) | FP16 | 94,210 MB | 1x baseline |

VRAM reservation is the same (95%), but **FP8 KV stores 2x more tokens in the same memory**. This means:
- Same VRAM → **2x longer context** at same concurrency
- Same context → **2x more concurrent requests**

### FP8 KV Impact on Concurrency Scaling

| Model | FP16 KV (Phase 1) | FP8 KV (This Test) | Improvement |
|-------|:------------------:|:-------------------:|:-----------:|
| Qwen3.5-27B | 7.8x | **8.1x** | +4% |
| Qwopus-27B | 7.7x | **8.0x** | +4% |
| Gemma4-31B | 6.0x | **6.4x** | +7% |

Qwen3.5's MLA already compresses KV natively, so FP8 stacks multiplicatively. Gemma4 benefits more from FP8 since it has no native KV compression.

### Architecture Comparison

| Aspect | Qwen3.5 (27B) | Gemma4 (31B) |
|--------|:-------------:|:------------:|
| English prose | Good (0.96) | **Excellent (0.98)** |
| Japanese | **Excellent (0.90)** | Good (0.74) |
| Math | Good (0.73) | Decent (0.65) |
| Coding | Good (0.85) | **Excellent (0.90)** |
| Speed | **57-58 tok/s** | 50 tok/s |
| MLA | **Yes** | No |
| Concurrency scaling | **8.1x** | 6.4x |
| Context | 262K | 262K |

![Latency Distribution](figures/04_latency_distribution.png)

![Output Length](figures/05_output_length.png)

## Which Model Should You Use?

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **English / essays** | Gemma4-31B | 0.98 quality |
| **Math reasoning** | Qwopus-27B | 0.90, Opus distillation |
| **Coding** | Gemma4-31B or Qwopus | Both 0.90 |
| **Japanese** | Qwen3.5-27B | 0.90 at 128K context |
| **All-rounder** | Qwen3.5-27B | No weak spots + MTP + MLA |
| **Max throughput** | Qwen3.5-27B | 8.1x scaling, zero per-req degradation |
| **Long context** | Any + `--kv-cache-dtype fp8` | 2x KV capacity, no quality loss |

## Recommended vLLM Flags

```bash
vllm serve <model> \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95
```

## Reproducibility

```bash
git clone https://github.com/lna-lab/27b-31b-nvfp4-bench
cd 27b-31b-nvfp4-bench

# FP8 KV benchmark (128K context)
docker compose -f docker-compose.bench-fp8kv.yml up -d
pip install aiohttp
python bench.py --output results/benchmark_fp8kv.json

# FP16 KV baseline (8K context)
docker compose -f docker-compose.bench.yml up -d
python bench.py --output results/benchmark.json

pip install matplotlib
python generate_figures.py
```

## License

Benchmark code: MIT. Models subject to their respective licenses (Qwen / Gemma).

## Credits

- Models: [huihui-ai](https://huggingface.co/huihui-ai), [Jackrong](https://huggingface.co/Jackrong), [Google](https://huggingface.co/google), [Qwen](https://huggingface.co/Qwen)
- Community baseline: [lyf](https://huggingface.co/lyf)
- Quantization: [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Benchmark: [Lna-Lab](https://github.com/lna-lab)
