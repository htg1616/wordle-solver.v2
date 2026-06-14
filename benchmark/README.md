# semi_full_benchmark package

Copy these files into the repository root:

```text
benchmark/semi_full_benchmark.py
benchmark/debug_cases_slow_failed_top50.json
benchmark/README_semi_full_benchmark.md
.github/workflows/semi_full_benchmark.yml
```

The workflow runs 20 parallel GitHub Actions chunks and merges the result artifact as `semi_full_merged`.
