#!/usr/bin/env python3
"""
전체 벤치마크: words.txt의 모든 단어를 secret으로, words.txt 전체를 candidate list로 테스트.

기능:
- 체크포인트/재시작: 중단 시 checkpoint.jsonl에 진행 상황 저장, 재시작 시 이어서 실행
- 느린 단어 저장: act_wall_s > --slow-threshold 인 단어들을 slow_words.json에 저장
- 실패 단어 저장: 못 맞힌 단어들을 failed_words.json에 저장
- 청크 분할: --chunk-index / --chunk-count 로 병렬 실행 가능 (GitHub Actions 등)

사용 예:
  # 전체 실행 (모든 트랙, 시드 0)
  python benchmark/run_full_benchmark.py

  # 트랙 1만 빠르게 테스트
  python benchmark/run_full_benchmark.py --tracks 1 --seeds 0

  # 중단 후 같은 디렉토리로 재시작
  python benchmark/run_full_benchmark.py --out-dir benchmark/results/full_20260610_120000

  # GitHub Actions용 분할 (26청크 중 0번째)
  python benchmark/run_full_benchmark.py --chunk-index 0 --chunk-count 26
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import (
    DEFAULT_SOLVER,
    DEFAULT_WORDS,
    GameResult,
    Problem,
    iso_dt,
    load_solver_module,
    load_words_txt,
    now_kst,
    parse_int_list_or_range,
    result_to_row,
    run_direct_game,
    run_http_game,
    summarize_results,
)

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_SLOW_THRESHOLD = 10   # 이 턴 수 이상 걸린 게임의 단어를 slow_words에 기록
DEFAULT_CHECKPOINT_INTERVAL = 200


# ---------------------------------------------------------------------------
# 체크포인트 I/O
# ---------------------------------------------------------------------------

def game_key(secret: str, track: int, seed: int) -> str:
    return f"{secret}__{track}__{seed}"


def load_checkpoint(path: Path) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """체크포인트 JSONL에서 완료된 게임 키와 결과 행을 로드."""
    completed: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return completed, rows
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = game_key(row["secret"], int(row["track"]), int(row["seed"]))
                if key not in completed:
                    completed.add(key)
                    rows.append(row)
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"  [경고] 체크포인트 {lineno}번 줄 파싱 실패, 건너뜀", file=sys.stderr)
    return completed, rows


def append_checkpoint(path: Path, row: Dict[str, Any]) -> None:
    """결과 한 행을 체크포인트 JSONL에 추가."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 최종 출력 파일 작성
# ---------------------------------------------------------------------------

def write_outputs(
    out_dir: Path,
    all_rows: List[Dict[str, Any]],
    slow_threshold: int,
    run_info: Dict[str, Any],
) -> None:
    """games.csv, games.jsonl, slow_words.json, failed_words.json, summary.json 저장."""
    if not all_rows:
        print("결과가 없어 최종 출력을 건너뜁니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # games.csv
    csv_path = out_dir / "games.csv"
    fields = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # games.jsonl
    jsonl_path = out_dir / "games.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 느린 단어: 어떤 (track, seed)에서든 score_turns > threshold 이면 기록
    # score_turns: 성공 시 실제 턴 수, 실패 시 score_cap(100)
    # 단어별로 느린 케이스를 모두 저장, max_score_turns 내림차순 정렬
    slow_by_secret: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        try:
            score_turns = int(row.get("score_turns", 0))
        except (TypeError, ValueError):
            score_turns = 0
        if score_turns > slow_threshold:
            secret = row["secret"]
            slow_by_secret.setdefault(secret, []).append({
                "track": int(row["track"]),
                "seed": int(row["seed"]),
                "success": bool(int(row.get("success", 0))),
                "status": row.get("status", ""),
                "score_turns": score_turns,
                "turns": int(row.get("turns", 0)),
                "act_wall_s": row.get("act_wall_s", ""),
            })
    slow_list = sorted(
        [
            {
                "secret": k,
                "max_score_turns": max(c["score_turns"] for c in cases),
                "cases": sorted(cases, key=lambda c: -c["score_turns"]),
            }
            for k, cases in slow_by_secret.items()
        ],
        key=lambda x: -x["max_score_turns"],
    )
    slow_path = out_dir / "slow_words.json"
    with slow_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"threshold_turns": slow_threshold, "count": len(slow_list), "words": slow_list},
            f, indent=2, ensure_ascii=False,
        )

    # 실패한 단어: 어떤 (track, seed)에서든 success == 0 이면 기록
    failed_by_secret: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        if int(row.get("success", 1)) == 0:
            secret = row["secret"]
            failed_by_secret.setdefault(secret, []).append({
                "track": int(row["track"]),
                "seed": int(row["seed"]),
                "status": row.get("status", ""),
                "score_turns": int(row.get("score_turns", 0)),
                "turns": int(row.get("turns", 0)),
                "act_wall_s": row.get("act_wall_s", ""),
            })
    failed_list = sorted(
        [{"secret": k, "failures": v} for k, v in failed_by_secret.items()],
        key=lambda x: x["secret"],
    )
    failed_path = out_dir / "failed_words.json"
    with failed_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"count": len(failed_list), "words": failed_list},
            f, indent=2, ensure_ascii=False,
        )

    # summary.json: 트랙별/전체 통계를 담은 요약
    # all_rows를 GameResult 없이 직접 집계하는 간이 요약
    total = len(all_rows)
    solved = sum(1 for r in all_rows if int(r.get("success", 0)) == 1)
    failed_count = total - solved
    avg_act = sum(float(r.get("act_wall_s", 0)) for r in all_rows) / total if total else 0.0
    summary = {
        "run": run_info,
        "games": total,
        "solved": solved,
        "failed": failed_count,
        "success_rate": round(solved / total, 6) if total else 0.0,
        "slow_words_count": len(slow_list),
        "failed_words_count": len(failed_list),
        "avg_act_wall_s": round(avg_act, 4),
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "slow_words": str(slow_path),
            "failed_words": str(failed_path),
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n저장된 파일:")
    print(f"  games.csv       : {csv_path}  ({total}행)")
    print(f"  games.jsonl     : {jsonl_path}")
    print(f"  slow_words.json : {slow_path}  ({len(slow_list)}개, >{slow_threshold}턴)")
    print(f"  failed_words.json: {failed_path}  ({len(failed_list)}개)")
    print(f"  summary.json    : {summary_path}")


# ---------------------------------------------------------------------------
# 경로 해석 헬퍼
# ---------------------------------------------------------------------------

def resolve_path(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise SystemExit(f"{label} 파일을 찾을 수 없음: {raw}")
    for base in (Path.cwd(), REPO_ROOT, SCRIPT_DIR):
        cand = base / path
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"{label} 파일을 찾을 수 없음: {raw!r}")


# ---------------------------------------------------------------------------
# CLI 및 메인 루프
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="전체 벤치마크: words.txt 모든 단어를 secret으로, 전체를 candidate로 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--solver", default=str(DEFAULT_SOLVER), help="solver 파일 경로 (기본: team00.py)")
    p.add_argument("--words", default=str(DEFAULT_WORDS), help="단어 목록 파일 (기본: words.txt)")
    p.add_argument("--mode", choices=("direct", "http"), default="direct", help="실행 모드")
    p.add_argument("--tracks", default="1,2,3", help="테스트할 트랙, 예: 1,2,3 또는 2")
    p.add_argument("--seeds", default="0", help="노이즈 시드, 예: 0 또는 0,1,2")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--budget", type=float, default=60.0, help="게임당 최대 /act 총 시간(초)")
    p.add_argument("--score-cap", type=int, default=100)
    p.add_argument(
        "--slow-threshold", type=int, default=DEFAULT_SLOW_THRESHOLD,
        help=f"느린 단어 기준 score_turns(턴 수), 기본 {DEFAULT_SLOW_THRESHOLD}턴",
    )
    p.add_argument("--out-dir", default=None, help="결과 저장 디렉토리 (기본: benchmark/results/full_<timestamp>)")
    p.add_argument("--chunk-index", type=int, default=None, help="청크 인덱스 (0-based)")
    p.add_argument("--chunk-count", type=int, default=None, help="전체 청크 수")
    p.add_argument("--http-request-timeout", type=float, default=65.0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    started_at = now_kst()

    # 경로 해석
    solver_path = resolve_path(args.solver, "solver")
    words_path = resolve_path(args.words, "words.txt")
    all_words: List[str] = load_words_txt(words_path)

    tracks = parse_int_list_or_range(args.tracks, allowed={1, 2, 3})
    seeds = parse_int_list_or_range(args.seeds)

    # 청크 분할
    if args.chunk_index is not None and args.chunk_count is not None:
        if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
            raise SystemExit(f"--chunk-index {args.chunk_index}가 --chunk-count {args.chunk_count} 범위를 벗어남")
        chunk_words = [w for i, w in enumerate(all_words) if i % args.chunk_count == args.chunk_index]
        chunk_suffix = f"_chunk{args.chunk_index:03d}of{args.chunk_count:03d}"
    else:
        chunk_words = list(all_words)
        chunk_suffix = ""

    # 출력 디렉토리
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        stamp = started_at.strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_RESULTS_ROOT / f"full_{stamp}{chunk_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.jsonl"

    # 체크포인트 로드 (재시작)
    completed: Set[str] = set()
    existing_rows: List[Dict[str, Any]] = []
    if checkpoint_path.exists():
        completed, existing_rows = load_checkpoint(checkpoint_path)
        if completed:
            print(f"체크포인트 로드 완료: {len(completed)}개 게임 이미 완료됨, 이어서 실행합니다.")

    # 총 게임 수 계산
    total_games = len(chunk_words) * len(tracks) * len(seeds)
    remaining = total_games - len(completed)

    print(f"\n=== 전체 벤치마크 ===")
    print(f"  words.txt 단어 수 : {len(all_words)}")
    print(f"  이번 청크 단어 수 : {len(chunk_words)}" + (f"  (청크 {args.chunk_index+1}/{args.chunk_count})" if args.chunk_count else ""))
    print(f"  트랙: {tracks},  시드: {seeds}")
    print(f"  총 게임 수: {total_games}  (완료: {len(completed)},  남은: {remaining})")
    print(f"  느린 단어 기준   : score_turns > {args.slow_threshold}턴")
    print(f"  출력 디렉토리    : {out_dir}")
    print(f"  체크포인트       : {checkpoint_path}")
    print()

    if remaining == 0:
        print("모든 게임이 이미 완료됐습니다. 최종 출력만 다시 생성합니다.")
        run_info = {"note": "all_completed_at_start"}
        write_outputs(out_dir, existing_rows, args.slow_threshold, run_info)
        return 0

    # solver 로드
    solver_module = None
    if args.mode == "direct":
        solver_module = load_solver_module(solver_path)

    # 메인 루프
    new_rows: List[Dict[str, Any]] = []
    done_this_run = 0
    start_all = time.perf_counter()

    for secret in chunk_words:
        # 이 secret의 candidate list = 전체 단어 목록
        problem = Problem(
            problem_id=f"full_{secret}",
            secret_word=secret,
            candidate_words=list(all_words),
        )
        for track in tracks:
            for seed in seeds:
                key = game_key(secret, track, seed)
                if key in completed:
                    continue

                if args.mode == "direct":
                    assert solver_module is not None
                    result = run_direct_game(
                        solver_module, problem, track, seed,
                        args.max_turns, args.budget, args.score_cap,
                        trace_enabled=False,
                    )
                else:
                    result = run_http_game(
                        solver_path, problem, track, seed,
                        args.max_turns, args.budget, args.score_cap,
                        trace_enabled=False,
                        request_timeout=args.http_request_timeout,
                    )

                row = result_to_row(result)
                append_checkpoint(checkpoint_path, row)
                new_rows.append(row)
                completed.add(key)
                done_this_run += 1

                global_done = len(existing_rows) + done_this_run
                mark = "OK  " if result.success else "FAIL"
                elapsed_so_far = time.perf_counter() - start_all
                rate = done_this_run / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (remaining - done_this_run) / rate if rate > 0 else 0
                print(
                    f"[{global_done:>6}/{total_games}] {mark} T{track} s={seed} "
                    f"secret={secret} status={result.status:14s} "
                    f"score={result.score_turns:>3} act={result.act_wall_s:6.2f}s"
                    f"  ETA:{eta/60:5.1f}min",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_all
    finished_at = now_kst()
    print(f"\n이번 실행 완료: {done_this_run}개 게임, {elapsed:.1f}초")

    # 전체 결과 (기존 + 신규)
    all_rows = existing_rows + new_rows
    total_completed = len(all_rows)
    solved_count = sum(1 for r in all_rows if int(r.get("success", 0)) == 1)
    success_pct = 100.0 * solved_count / total_completed if total_completed else 0.0
    print(f"전체 완료: {total_completed}게임  성공: {solved_count}  ({success_pct:.1f}%)")

    run_info = {
        "run_id": out_dir.name,
        "started_at": iso_dt(started_at),
        "finished_at": iso_dt(finished_at),
        "elapsed_s": round(elapsed, 3),
        "chunk_index": args.chunk_index,
        "chunk_count": args.chunk_count,
        "tracks": tracks,
        "seeds": seeds,
        "words_total": len(all_words),
        "chunk_words": len(chunk_words),
        "games_total": total_games,
        "games_completed": total_completed,
        "slow_threshold_turns": args.slow_threshold,
    }
    write_outputs(out_dir, all_rows, args.slow_threshold, run_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
