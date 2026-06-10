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

---

## 전체 벤치마크 결과 해석 (full_benchmark.zip)

GitHub Actions에서 **Full Benchmark (Exhaustive)** 워크플로우를 실행하면
`full-benchmark-merged` 아티팩트가 생성됩니다. 다운로드하면 zip 파일이며,
압축을 풀면 아래 파일들이 들어 있습니다.

```
full-benchmark-merged/
  summary.json        ← 전체 통계 요약 (가장 먼저 볼 파일)
  slow_words.json     ← 턴 수가 기준치를 초과한 단어 목록
  failed_words.json   ← 한 번이라도 틀린 단어 목록
  games.csv           ← 전체 게임 결과 (38,916행)
  games.jsonl         ← games.csv와 동일한 내용, JSON Lines 형식
```

### summary.json

```json
{
  "games": 38916,          // 전체 실행 게임 수
  "solved": 38800,         // 성공한 게임 수
  "failed": 116,           // 실패한 게임 수
  "success_rate": 0.99702, // 전체 성공률
  "slow_words_count": 42,  // 느린 단어 수 (score_turns > threshold)
  "failed_words_count": 23 // 한 번이라도 틀린 단어 수
}
```

> **참고**: `games`는 단어 수가 아닙니다. 단어 1개당 트랙 3개 × 시드 1개 = 3 game이므로
> `games = 단어 수 × 3`입니다.

---

### slow_words.json

`score_turns > slow_threshold` (기본값 15)인 케이스가 있는 단어 목록입니다.
`max_score_turns` 내림차순으로 정렬되어 있으므로 **가장 어려웠던 단어가 맨 위**에 옵니다.

```json
{
  "threshold_turns": 15,
  "count": 42,
  "words": [
    {
      "secret": "soggy",
      "max_score_turns": 24,   // 이 단어에서 가장 나빴던 score_turns
      "cases": [               // threshold를 초과한 케이스 전부 (score_turns 내림차순)
        {
          "track": 3,
          "seed": 0,
          "success": false,    // true = 맞힘, false = 틀림
          "status": "wrong_submit",  // 실패 원인 (아래 상태 코드 설명 참고)
          "score_turns": 100,  // 실패 시 score_cap(100), 성공 시 실제 턴 수
          "turns": 24,         // 실제로 몇 턴을 사용했는지
          "act_wall_s": "9.68" // /act 호출 총 소요 시간 (초)
        },
        {
          "track": 2,
          "seed": 0,
          "success": true,
          "status": "solved",
          "score_turns": 18,
          "turns": 18,
          "act_wall_s": "7.21"
        }
      ]
    },
    ...
  ]
}
```

**`score_turns`와 `turns`의 차이**

| 경우 | `turns` | `score_turns` |
|------|---------|---------------|
| 정상 성공 | 실제 턴 수 (예: 8) | 실제 턴 수와 동일 (8) |
| 틀리고 종료 | 실제 턴 수 (예: 20) | score_cap = 100 |

따라서 틀린 단어는 `score_turns = 100`으로 고정되므로,
threshold가 15이면 **실패한 게임은 항상 slow_words에도 포함**됩니다.

---

### failed_words.json

성공률과 무관하게 **한 번이라도 틀린 단어** 목록입니다. 알파벳 순으로 정렬됩니다.

```json
{
  "count": 23,
  "words": [
    {
      "secret": "zoeae",
      "failures": [          // 실패한 케이스 전부
        {
          "track": 2,
          "seed": 0,
          "status": "wrong_submit",  // 실패 원인
          "score_turns": 100,        // 실패 시 항상 100
          "turns": 20,               // 실제로 몇 턴 시도했는지
          "act_wall_s": "8.45"       // 소요 시간 (초)
        },
        {
          "track": 3,
          "seed": 0,
          "status": "max_turns",
          "score_turns": 100,
          "turns": 100,
          "act_wall_s": "61.3"
        }
      ]
    },
    ...
  ]
}
```

**`status` 코드 의미**

| status | 의미 |
|--------|------|
| `solved` | 정답 제출 성공 |
| `wrong_submit` | submit을 했지만 정답이 아님 |
| `max_turns` | 최대 턴(100) 도달 전에 못 맞힘 |
| `budget_exceeded` | 60초 시간 초과 |
| `error` | solver 내부 오류 |

---

### games.csv

전체 38,916 게임의 원시 데이터입니다. 엑셀 또는 pandas로 열 수 있습니다.

주요 컬럼:

| 컬럼 | 설명 |
|------|------|
| `secret` | 이번 게임의 정답 단어 |
| `track` | 트랙 번호 (1=노이즈 없음, 2=중간, 3=강함) |
| `seed` | 노이즈 시드 |
| `success` | 성공 여부 (1=성공, 0=실패) |
| `status` | 상태 코드 (위 표 참고) |
| `turns` | 실제 사용 턴 수 |
| `score_turns` | 성공 시 turns와 동일, 실패 시 100 |
| `act_wall_s` | /act 호출 총 소요 시간 (초) |

pandas로 분석하는 예시:

```python
import pandas as pd

df = pd.read_csv("games.csv")

# 트랙별 성공률
print(df.groupby("track")["success"].mean())

# 성공한 게임의 평균 턴 수 (트랙별)
print(df[df["success"] == 1].groupby("track")["turns"].mean())

# 실패한 단어 목록
failed = df[df["success"] == 0]["secret"].unique()
print(f"실패한 단어 {len(failed)}개:", failed[:10])

# 가장 오래 걸린 상위 10개 단어
slow = df.groupby("secret")["act_wall_s"].max().nlargest(10)
print(slow)
```
