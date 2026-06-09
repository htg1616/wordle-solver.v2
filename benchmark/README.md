# Benchmark pipeline

`benchmark/benchmark.py` is the only benchmark entry point. It builds local
Wordle problems from `words.txt` or from an existing problem JSON, runs the
solver, and saves the generated problems and benchmark results.

## Defaults

When an argument is not given, the benchmark uses these defaults:

```text
candidate list : every valid word in words.txt
tracks         : 1,2,3
results dir    : benchmark/results/YYYYMMDD_HHMMSS_<bench>_<mode>/
latest copy    : benchmark/results/latest/
```

Each run also stores the run date and time in Asia/Seoul time. The timestamp is
written to `benchmark_run.json`, `benchmark_config.json`,
`benchmark_summary.json`, `benchmark_problem_manifest.json`, and each row of
`benchmark_games.csv` / `benchmark_games.jsonl`.

## Output files

Each run writes:

```text
benchmark_run.json                  # run id, date/time, elapsed time, output dir
benchmark_games.csv                 # one row per track/seed/problem game
benchmark_games.jsonl               # same data, plus traces when --trace is set
benchmark_summary.json              # aggregate success rate and score-turn stats
benchmark_problems.json             # generated benchmark problems with candidates
benchmark_problem_manifest.json     # compact reproducibility manifest
benchmark_config.json               # CLI/source configuration
single_problem_official_format.json # only when exactly one problem was generated
```

## Single benchmark from words.txt

Minimal command. This uses the full `words.txt` as the candidate list, runs all
tracks, and chooses an automatic results directory:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello
```

Use a sampled candidate list instead of the full `words.txt`:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 42
```

Use an exact candidate list:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidates hello,world,crane,slate,trace
```

Use a candidate file:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret rooks \
  --candidate-file my_candidates.txt
```

If the exact candidate list omits the secret, the benchmark automatically adds
it unless `--strict-candidates` is set.

## Bulk benchmark from words.txt

Minimal command. This samples 10 secrets by default, gives each generated
problem the full `words.txt` candidate list, and runs tracks 1, 2, and 3:

```bash
python benchmark/benchmark.py --benchmark bulk
```

Sample many secrets and use sampled candidate lists:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --num-problems 100 \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 20260609 \
  --seeds 0,1,2
```

Run a specific secret list with one shared candidate list:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --secrets hello,rooks,crane,slate \
  --candidate-size 1000 \
  --shared-candidates
```

Use every word in `words.txt` as a secret:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --all-secrets
```

This can be slow because it runs one game per secret, track, and seed.

## Existing problem JSON

Run the official-style problem JSON in HTTP mode:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --problem team00_problem.json \
  --mode http
```

## Custom output directory

The output directory is automatic unless `--results-dir` or `--out-dir` is set:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --results-dir benchmark/results/manual_single_hello
```
