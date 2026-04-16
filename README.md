# NVFP4 Showdown: 7 Models on 7 GPUs — Qwen3.6 Enters the Arena

**The new MoE king, Qwen3.6-35B-A3B (3B active, 256 experts), faces off against 6 established models. Does it change the game?**

All 7 NVIDIA RTX PRO 6000 Blackwell GPUs. All NVFP4. All FP8 KV cache. 128K context. One benchmark.

## The Lineup

### MoE (sparse, fast)

| GPU | Model | Active | Experts | Speed | Source |
|:---:|-------|:------:|:-------:|:-----:|--------|
| 0 | **Qwen3.6-35B** | **3B** | **256** | **144 tok/s** | [Lna-Lab](https://huggingface.co/sakamakismile/Qwen3.6-35B-A3B-NVFP4) |
| 1 | RedHatAI Gemma4-26B | 3.8B | 128 | 130 tok/s | [RedHatAI](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4) |
| 2 | Huihui Gemma4-26B | 3.8B | 128 | 130 tok/s | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-gemma-4-26B-A4B-it-abliterated-NVFP4) |
| 6 | Jiunsong SuperGemma4 | 3.8B | 128 | 129 tok/s | [Lna-Lab](https://huggingface.co/sakamakismile/SuperGemma4-26B-Abliterated-Multimodal-NVFP4) |

### Dense (all params active, deep)

| GPU | Model | Active | Speed | Source |
|:---:|-------|:------:|:-----:|--------|
| 3 | Qwen3.5-27B | 27B | 56 tok/s | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-Qwen3.5-27B-abliterated-NVFP4) |
| 4 | Qwopus3.5-27B (Opus) | 27B | 57 tok/s | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-Qwopus3.5-27B-v3-abliterated-NVFP4) |
| 5 | Gemma4-31B | 31B | 50 tok/s | [Lna-Lab](https://huggingface.co/sakamakismile/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4) |

## Results

![Quality Comparison](figures/01_quality_comparison.png)

### Quality Scores (Concurrency = 1)

| Test | Qwen3.6 MoE | RedHat MoE | Huihui MoE | Jiunsong MoE | Qwen Dense | Qwopus Dense | Gemma31 Dense |
|------|:-----------:|:----------:|:----------:|:------------:|:----------:|:------------:|:-------------:|
| **English** | 0.93 | 0.91 | 0.82 | 0.78 | **0.97** | 0.65 | **0.98** |
| **Japanese** | 0.74 | 0.73 | 0.73 | 0.75 | **0.90** | 0.83 | 0.74 |
| **Math** | 0.75 | 0.80 | 0.75 | 0.63 | 0.73 | **0.90** | 0.65 |
| **Coding** | 0.80 | 0.85 | 0.78 | **0.90** | **0.90** | 0.85 | **0.90** |
| **Design** | 0.78 | 0.80 | 0.80 | **0.84** | 0.79 | 0.71 | 0.80 |

![Radar Profile](figures/06_radar_profile.png)

## What Qwen3.6 Changes

### Speed: New Champion

![Per-Request Speed](figures/02_speed_per_request.png)

| Model | tok/s | vs Gemma4 MoE | vs Dense |
|-------|:-----:|:-------------:|:--------:|
| **Qwen3.6 MoE** | **144** | **+11%** | **+2.6x** |
| Gemma4 MoE (avg) | 130 | baseline | +2.3x |
| Dense (avg) | 54 | -58% | baseline |

Qwen3.6 achieves 144 tok/s with only **3B active parameters** — 20% fewer active params than Gemma4 MoE (3.8B) yet 11% faster. The 256-expert architecture with Gated DeltaNet is remarkably efficient.

### Quality: Competitive with MoE, Shy of Dense

Within the MoE class, Qwen3.6 leads on English (0.93) and ties on Math (0.75). But Dense models — especially Qwopus (Opus-distilled) on Math (0.90) and Qwen3.5 on Japanese (0.90) — still hold the quality crown on reasoning-heavy tasks.

**The gap between MoE and Dense is real but narrowing.** Qwen3.6 closes the English quality gap to within 5% of Dense, while running 2.6x faster.

### Throughput Scaling

![Throughput Scaling](figures/03_throughput_scaling.png)

## VRAM Usage (7 GPUs, 128K Context, FP8 KV)

| GPU | Model | Type | VRAM |
|:---:|-------|:----:|-----:|
| 0 | Qwen3.6-35B | MoE | 92,316 MB |
| 1 | RedHatAI | MoE | 92,862 MB |
| 2 | Huihui Gemma4 | MoE | 92,862 MB |
| 3 | Qwen3.5-27B | Dense | 92,470 MB |
| 4 | Qwopus-27B | Dense | 92,470 MB |
| 5 | Gemma4-31B | Dense | 94,210 MB |
| 6 | Jiunsong | MoE | 84,195 MB |

All models fit on a single 96 GB Blackwell GPU at 128K context with FP8 KV cache.

![Latency Distribution](figures/04_latency_distribution.png)

![Output Length](figures/05_output_length.png)

## The Big Picture

### Does Qwen3.6 change the game?

**Yes, but not how you might expect.**

It doesn't make Dense models obsolete — Dense still wins on deep reasoning. What it does is raise the MoE speed floor while closing the quality gap:

| Era | Best MoE | Best Dense | Speed Gap | Quality Gap |
|-----|----------|-----------|:---------:|:-----------:|
| Before Qwen3.6 | 130 tok/s | 57 tok/s | 2.3x | Large |
| **After Qwen3.6** | **144 tok/s** | 57 tok/s | **2.6x** | **Narrowing** |

For production deployments:
- **More workloads can use MoE** — the quality threshold is now high enough for most tasks
- **Dense is justified only for premium reasoning** — math, complex analysis, multilingual
- **Fleet strategy matters more than ever** — route by task type

### Fleet Recommendation (7-GPU Node)

```
GPU 0-1: Qwen3.6 MoE × 2  — high-volume, agentic, coding
GPU 2:   Huihui Gemma4 MoE — abliterated search/summarization
GPU 3-4: Dense Qwen/Qwopus — reasoning, math, Japanese
GPU 5:   Gemma4-31B Dense  — English writing, deep analysis
GPU 6:   Jiunsong MoE      — design tasks, multimodal
```

## Scoring Note

Quality scores are automated heuristics (text structure, vocabulary, code indicators). They provide relative ranking, not absolute quality. Within-class differences (MoE-to-MoE, Dense-to-Dense) are within noise at n=2. The MoE-vs-Dense gap and Qwen3.6's speed advantage are consistent and real. For authoritative quality assessment, see [official benchmarks](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

## Reproducibility

```bash
git clone https://github.com/lna-lab/27b-35b-nvfp4-bench
cd 27b-35b-nvfp4-bench

# 7-model benchmark (requires 7 GPUs)
docker compose -f docker-compose.bench-7models.yml up -d
pip install aiohttp
python bench.py --output results/benchmark_7models.json

pip install matplotlib
python generate_figures.py
```

## License

Benchmark code: MIT. Models subject to their respective licenses.

## Credits

- Models: [Qwen](https://huggingface.co/Qwen), [huihui-ai](https://huggingface.co/huihui-ai), [Jiunsong](https://huggingface.co/Jiunsong), [RedHatAI](https://huggingface.co/RedHatAI), [Google](https://huggingface.co/google), [Jackrong](https://huggingface.co/Jackrong)
- Quantization: [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Benchmark: [Lna-Lab](https://github.com/lna-lab)
