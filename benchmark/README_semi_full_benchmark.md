# semi_full_benchmark

`semi_full_benchmark` runs a deterministic medium-sized benchmark set:

- 50 previously slow/failed regression words from `debug_cases_slow_failed_top50.json`
- 1000 random words sampled from `words.txt` with seed `20260614`
- Tracks 1, 2, and 3
- Seed 0

Total games: `(50 + 1000) * 3 = 3150`.

The GitHub Actions workflow runs these in 20 parallel chunks and then merges the chunk artifacts.

## Local smoke test

```bash
python benchmark/semi_full_benchmark.py \
  --solver team00.py \
  --words words.txt \
  --hard-cases benchmark/debug_cases_slow_failed_top50.json \
  --random-count 10 \
  --chunk-index 0 \
  --chunk-count 20 \
  --max-games 3 \
  --out-dir benchmark/results/semi_full_smoke
```

## Local full semi benchmark without chunking

```bash
python benchmark/semi_full_benchmark.py \
  --solver team00.py \
  --words words.txt \
  --hard-cases benchmark/debug_cases_slow_failed_top50.json \
  --random-count 1000 \
  --tracks 1,2,3 \
  --seeds 0 \
  --out-dir benchmark/results/semi_full_local
```

## Local 20-way chunk example

```bash
python benchmark/semi_full_benchmark.py \
  --solver team00.py \
  --words words.txt \
  --hard-cases benchmark/debug_cases_slow_failed_top50.json \
  --chunk-index 0 \
  --chunk-count 20 \
  --out-dir benchmark/results/semi_full_chunk_00
```

## Merge downloaded GitHub Actions artifacts locally

```bash
python benchmark/merge_semi_full_benchmark.py \
  --merge-root benchmark/results/semi_full_benchmark_artifacts \
  --out-dir benchmark/results/semi_full_benchmark_merged
```
