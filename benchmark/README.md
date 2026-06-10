# 벤치마크 파이프라인

`benchmark/benchmark.py`가 유일한 벤치마크 파이프라인 진입점입니다. 이 스크립트는
`words.txt` 또는 기존 problem JSON에서 로컬 Wordle 문제를 만들고,
solver를 실행한 뒤 생성된 문제와 벤치마크 결과를 저장합니다.

## 기본값

인자를 지정하지 않으면 벤치마크는 다음 기본값을 사용합니다.

```text
candidate list : every valid word in words.txt
tracks         : 1,2,3
results dir    : benchmark/results/YYYYMMDD_HHMMSS_<bench>_<mode>/
latest copy    : benchmark/results/latest/
```

각 실행은 날짜와 시간을 Asia/Seoul 시간대로 저장합니다. timestamp는
`benchmark_run.json`, `benchmark_config.json`, `benchmark_summary.json`,
`benchmark_problem_manifest.json`, 그리고 `benchmark_games.csv` /
`benchmark_games.jsonl`의 각 row에 기록됩니다.

## 출력 파일

각 실행은 다음 파일을 만듭니다.

```text
benchmark_run.json                  # run id, date/time, elapsed time, output dir
benchmark_games.csv                 # track/seed/problem game마다 한 row
benchmark_games.jsonl               # 같은 데이터와 --trace 사용 시 trace 포함
benchmark_summary.json              # success rate와 score-turn 통계
benchmark_problems.json             # candidate가 포함된 생성 문제
benchmark_problem_manifest.json     # 재현을 위한 간단한 manifest
benchmark_config.json               # CLI/source configuration
single_problem_official_format.json # 문제가 정확히 하나 생성된 경우에만 생성
```

## `words.txt`에서 Single benchmark 실행

최소 명령입니다. 전체 `words.txt`를 candidate list로 사용하고, 모든 track을
실행하며, results directory는 자동으로 선택됩니다.

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello
```

전체 `words.txt` 대신 샘플링된 candidate list를 사용하려면:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 42
```

candidate list를 직접 지정하려면:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --candidates hello,world,crane,slate,trace
```

candidate file을 사용하려면:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret rooks \
  --candidate-file my_candidates.txt
```

직접 지정한 candidate list에 secret이 빠져 있으면, `--strict-candidates`를
설정하지 않은 경우 벤치마크가 secret을 자동으로 추가합니다.

## `words.txt`에서 Bulk benchmark 실행

최소 명령입니다. 기본적으로 secret 10개를 샘플링하고, 생성된 각 문제에
전체 `words.txt` candidate list를 제공하며, track 1, 2, 3을 실행합니다.

```bash
python benchmark/benchmark.py --benchmark bulk
```

많은 secret과 샘플링된 candidate list를 사용하려면:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --num-problems 100 \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 20260609 \
  --seeds 0,1,2
```

특정 secret list와 하나의 공유 candidate list를 사용하려면:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --secrets hello,rooks,crane,slate \
  --candidate-size 1000 \
  --shared-candidates
```

`words.txt`의 모든 단어를 secret으로 사용하려면:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --all-secrets
```

이 방식은 secret, track, seed 조합마다 한 game을 실행하므로 오래 걸릴 수
있습니다.

## 기존 problem JSON 실행

공식 형식의 problem JSON을 HTTP mode로 실행하려면:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --problem team00_problem.json \
  --mode http
```

## 사용자 지정 출력 디렉터리

`--results-dir` 또는 `--out-dir`을 지정하지 않으면 output directory는
자동으로 정해집니다.

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello \
  --results-dir benchmark/results/manual_single_hello
```
