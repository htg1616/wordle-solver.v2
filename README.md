# DCCP Noisy Wordle Solver 안내

이 저장소는 실제 수업 제출 파일을 루트에 두고, 로컬 벤치마크 도구는
`benchmark/` 아래에 모아 둡니다.

## 제출 파일

```text
team00.py             # solver HTTP 서버; 실행: python team00.py
team00_problem.json   # 공식 형식의 문제 예시
requirements.txt      # NumPy 의존성
```

solver는 환경변수 `PORT`를 읽고, 필수 엔드포인트를 구현합니다.

```text
POST /start_problem
POST /act
```

채점 중에는 `words.txt`를 읽지 않습니다. 채점기는 `/start_problem`으로
후보 단어 전체를 전달하며, 모든 guess/submit 단어는 그 후보 목록에서
선택됩니다.

## 벤치마크 구성

```text
benchmark/
|-- benchmark.py      # single/bulk 벤치마크 진입점
|-- README.md         # 자세한 벤치마크 예시
`-- results/          # 로컬 생성 결과; git에서 무시

words.txt             # 로컬 벤치마크 문제 생성을 위한 단어 목록
scripts/
|-- run_quick_benchmark.sh
|-- run_http_smoke.sh
`-- make_submission_zip.sh
```

`benchmark/benchmark.py`는 `words.txt`에서 secret word와 candidate list를
만들고, 생성된 벤치마크 문제와 결과를 저장할 수 있습니다.

## 벤치마크 기본값

인자를 생략하면 다음 값이 사용됩니다.

```text
candidate list : every valid word in words.txt
tracks         : 1,2,3
results dir    : benchmark/results/YYYYMMDD_HHMMSS_<bench>_<mode>/
latest copy    : benchmark/results/latest/
```

벤치마크 실행 날짜와 시간은 Asia/Seoul 시간대로 저장됩니다. 이 timestamp는
`benchmark_run.json`, `benchmark_config.json`, `benchmark_summary.json`,
`benchmark_problem_manifest.json`, 그리고 각 game row에 기록됩니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Single benchmark 실행

전체 candidate list와 모든 track을 사용하는 최소 실행:

```bash
python benchmark/benchmark.py \
  --benchmark single \
  --secret hello
```

전체 `words.txt` 대신 candidate list를 샘플링하려면:

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

## Bulk benchmark 실행

최소 bulk 실행:

```bash
python benchmark/benchmark.py --benchmark bulk
```

더 큰 샘플을 사용하는 실행:

```bash
python benchmark/benchmark.py \
  --benchmark bulk \
  --num-problems 100 \
  --candidate-size 1000 \
  --candidate-policy random \
  --problem-seed 20260609 \
  --seeds 0,1,2
```

## HTTP smoke test 실행

```bash
bash scripts/run_http_smoke.sh
```

이 명령은 `python team00.py`를 subprocess로 실행하고, 공식 형식에 가까운
HTTP request를 보냅니다.

## 빠른 direct benchmark

```bash
bash scripts/run_quick_benchmark.sh
```

direct mode는 solver class를 import해서 실행하므로 알고리즘 튜닝 시 더
빠릅니다.

## eTL 제출 zip 만들기

```bash
bash scripts/make_submission_zip.sh
```

출력 파일:

```text
dist/team00_code.zip
dist/team00_problem.json
```

제출 zip에는 `benchmark/`, `words.txt`, 생성된 결과, 기타 로컬 전용 파일이
의도적으로 포함되지 않습니다.
