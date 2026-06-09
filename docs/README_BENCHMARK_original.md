# Junior Algorithm Local Benchmark Kit

이 폴더는 `words.txt` 단어 목록으로 Wordle solver를 로컬에서 벤치마크하기 위한 패키지입니다.

## 파일 구성

- `team00.py`: 개선된 solver 서버
- `words.txt`: 업로드한 전체 단어 목록
- `benchmark_words.py`: `words.txt`에서 secret/candidate 문제를 샘플링해서 바로 벤치마크
- `benchmark.py`: 단일 `teamXX_problem.json` 형식 문제를 벤치마크하는 범용 로컬 그레이더
- `make_problem_from_words.py`: `words.txt`에서 `teamXX_problem.json` 형식 문제 생성
- `words_problem_1000.json`: `words.txt`에서 만든 후보 1000개짜리 예시 문제
- `run_quick_benchmark.sh`: 빠른 direct-mode 벤치마크 예시
- `run_http_smoke.sh`: HTTP 서버 프로토콜 smoke test 예시

## 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 `numpy`만 들어 있습니다.

## 가장 빠른 확인

```bash
python benchmark_words.py \
  --solver team00.py \
  --words words.txt \
  --mode direct \
  --num-problems 3 \
  --candidate-size 500 \
  --tracks 1,2,3 \
  --seeds 0 \
  --out-dir benchmark_out_quick
```

또는:

```bash
./run_quick_benchmark.sh
```

`direct` 모드는 `team00.py`를 import해서 `initialize_problem()`과 `solver_act()`를 직접 호출합니다. 알고리즘 튜닝용으로 빠릅니다.

## 실제 제출 서버에 가까운 HTTP 확인

```bash
python benchmark_words.py \
  --solver team00.py \
  --words words.txt \
  --mode http \
  --num-problems 1 \
  --candidate-size 200 \
  --tracks 1 \
  --seeds 0 \
  --out-dir benchmark_out_http_smoke
```

또는:

```bash
./run_http_smoke.sh
```

`http` 모드는 매 게임마다 `python team00.py`를 새 프로세스로 띄우고, `PORT` 환경변수를 지정한 뒤 `/start_problem`, `/act` 요청을 보냅니다. 느리지만 제출 패키징과 서버 프로토콜 검증에 좋습니다.

## 전체 단어 목록으로 더 빡세게 돌리기

후보 리스트를 업로드한 `words.txt` 전체로 쓰려면 `--candidate-size all`을 사용하세요.

```bash
python benchmark_words.py \
  --solver team00.py \
  --words words.txt \
  --mode direct \
  --num-problems 20 \
  --candidate-size all \
  --tracks 1,2,3 \
  --seeds 0 \
  --out-dir benchmark_out_full20
```

전체 12,972개 후보를 쓰면 Track 2/3에서 훨씬 느려질 수 있습니다. 튜닝할 때는 `--candidate-size 500`, `1000`, `2000`처럼 키워가며 확인하는 것을 추천합니다.

## 여러 noise seed로 안정성 보기

```bash
python benchmark_words.py \
  --solver team00.py \
  --words words.txt \
  --mode direct \
  --num-problems 10 \
  --candidate-size 1000 \
  --tracks 2,3 \
  --seeds 0:5 \
  --out-dir benchmark_out_noise5
```

`--seeds 0:5`는 seed 0, 1, 2, 3, 4를 의미합니다.

## 실패 케이스 추적

```bash
python benchmark_words.py \
  --solver team00.py \
  --words words.txt \
  --mode direct \
  --num-problems 10 \
  --candidate-size 1000 \
  --tracks 1,2,3 \
  --seeds 0 \
  --trace \
  --out-dir benchmark_out_trace
```

`--trace`를 켜면 실패한 게임의 turn-by-turn 기록이 출력되고, `benchmark_games.jsonl`에도 저장됩니다. 각 guess에 대해 feedback code, noise type, effective secret이 들어갑니다.

## 결과 파일

벤치마크가 끝나면 `--out-dir` 아래에 다음 파일이 생깁니다.

- `benchmark_summary.json`: track별 성공률, 평균 score-turn, p90 score-turn, act wall time
- `benchmark_games.csv`: 게임별 요약
- `benchmark_games.jsonl`: 게임별 요약 및 선택적으로 trace
- `benchmark_problem_manifest.json`: 샘플링된 secret과 candidate count

콘솔 표의 주요 컬럼은 다음과 같습니다.

- `solved`: 정답 submit 성공 횟수
- `succ%`: 성공률
- `avg_score`: 실패를 100턴으로 계산한 평균 score-turn
- `avg_solved`: 성공한 게임만의 평균 제출 turn
- `avg_act_s`: `/act` 처리에 사용한 누적 시간의 평균

## problem JSON 만들기

제출용 또는 `benchmark.py`용 문제 파일을 만들려면:

```bash
python make_problem_from_words.py \
  --words words.txt \
  --candidate-size 1000 \
  --out my_problem.json \
  --seed 42
```

전체 단어를 candidate로 쓰려면:

```bash
python make_problem_from_words.py \
  --words words.txt \
  --candidate-size all \
  --out words_problem_full.json
```

생성된 문제를 `benchmark.py`로 직접 돌릴 수도 있습니다.

```bash
python benchmark.py \
  --solver team00.py \
  --problem words_problem_1000.json \
  --mode direct \
  --sample-secrets 10 \
  --tracks 1,2,3 \
  --seeds 0
```

## 다른 팀 번호로 바꾸기

제출 파일명이 `team29.py`라면 benchmark 명령에서 `--solver team29.py`로 바꾸면 됩니다. HTTP 모드도 해당 파일이 `PORT` 환경변수를 읽고 `/start_problem`, `/act`를 처리하면 그대로 동작합니다.
