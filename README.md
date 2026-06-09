# DCCP Noisy Wordle Solver

This repository keeps the actual course submission files at the repository root
and puts all local benchmarking tools under `benchmark/`.

## Submission files

```text
team00.py             # solver HTTP server; run with: python team00.py
team00_problem.json   # one official-format problem instance
requirements.txt      # NumPy dependency
```

The solver reads `PORT` from the environment and implements the required:

```text
POST /start_problem
POST /act
```

It does not read `words.txt` during grading. The grader provides all candidate
words through `/start_problem`, and every guess/submit word is selected from that
candidate list.

## Benchmark layout

```text
benchmark/
├── benchmark.py      # single/bulk benchmark entry point
├── README.md         # detailed benchmark examples
└── results/          # generated locally; ignored by git

words.txt             # source word list for local benchmark generation
scripts/
├── run_quick_benchmark.sh
├── run_http_smoke.sh
└── make_submission_zip.sh
```

`benchmark/benchmark.py` can generate both the secret word and candidate list
from `words.txt`, then save the generated benchmark problems and results.

## Benchmark defaults

When omitted, these values are used:

```text
candidate list : every valid word in words.txt
tracks         : 1,2,3
results dir    : benchmark/results/YYYYMMDD_HHMMSS_<bench>_<mode>/
latest copy    : benchmark/results/latest/
```

The benchmark run date/time is saved in Asia/Seoul time in `benchmark_run.json`,
`benchmark_config.json`, `benchmark_summary.json`,
`benchmark_problem_manifest.json`, and every game row.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Single benchmark

Minimal full-candidate, all-track run:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello
```

Sample the candidate list instead of using all `words.txt`:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 42
```

Use exact candidates:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidates hello,world,crane,slate,trace
```

## Bulk benchmark

Minimal bulk run:

```bash
python benchmark/benchmark.py --benchmark bulk
```

Larger sampled run:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --num-problems 100 \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 20260609 \
  --seeds 0,1,2
```

## HTTP smoke test

```bash
bash scripts/run_http_smoke.sh
```

This starts `python team00.py` as a subprocess and sends official-style HTTP
requests.

## Quick direct benchmark

```bash
bash scripts/run_quick_benchmark.sh
```

Direct mode imports the solver class and is faster for algorithm tuning.

## Build eTL submission zip

```bash
bash scripts/make_submission_zip.sh
```

Outputs:

```text
dist/team00_code.zip
dist/team00_problem.json
```

The submission zip intentionally excludes `benchmark/`, `words.txt`, generated
results, and other local-only files.
