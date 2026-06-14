#!/usr/bin/env python3
"""
semi_full_benchmark.py

Semi-full benchmark for the noisy Wordle solver.

Default benchmark set:
  1. 50 known slow/failed words from debug_cases_slow_failed_top50.json.
  2. 1000 deterministic random words from words.txt, excluding the 50 words above.
  3. Every selected word is run on Track 1, Track 2, and Track 3 with seed 0.

Default total: (50 + 1000) * 3 = 3150 games.

The game list is split by modulo over game index, so GitHub Actions can run
20 chunks in parallel.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import (  # type: ignore  # noqa: E402
    DEFAULT_SOLVER,
    DEFAULT_WORDS,
    Problem,
    iso_dt,
    load_solver_module,
    load_words_txt,
    now_kst,
    parse_int_list_or_range,
    result_to_row,
    run_direct_game,
    run_http_game,
)

DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_CASES_FILE = SCRIPT_DIR / "debug_cases_slow_failed_top50.json"
DEFAULT_RANDOM_COUNT = 1000
DEFAULT_HARD_COUNT = 50
DEFAULT_RANDOM_SEED = 20260614
DEFAULT_SLOW_THRESHOLD = 15


def resolve_path(raw: str, label: str, must_exist: bool = True) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path, SCRIPT_DIR / path]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    if must_exist:
        tried = ", ".join(str(c) for c in candidates)
        raise SystemExit(f"Could not find {label}: {raw!r}. Tried: {tried}")
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def game_key(secret: str, track: int, seed: int) -> str:
    return f"{secret}__{int(track)}__{int(seed)}"


def game_key_from_row(row: Dict[str, Any]) -> str:
    return game_key(str(row["secret"]), int(row["track"]), int(row["seed"]))


def load_checkpoint(path: Path) -> Tuple[Set[str], List[Dict[str, Any]]]:
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
                key = game_key_from_row(row)
            except Exception as exc:
                print(f"[warn] bad checkpoint line {lineno}: {exc}", file=sys.stderr)
                continue
            if key not in completed:
                completed.add(key)
                rows.append(row)
    return completed, rows


def append_checkpoint(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hard_words(path: Path, hard_count: int) -> List[str]:
    if hard_count <= 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"hard cases file has no cases list: {path}")
    words: List[str] = []
    seen: Set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        w = str(case.get("secret", "")).strip().lower()
        if len(w) != 5 or not w.isalpha() or not w.islower():
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
        if len(words) >= hard_count:
            break
    if len(words) < hard_count:
        print(f"[warn] requested {hard_count} hard words but found {len(words)}", file=sys.stderr)
    return words


def deterministic_random_words(words: Sequence[str], exclude: Set[str], count: int, seed: int) -> List[str]:
    pool = [w for w in words if w not in exclude]
    if count > len(pool):
        raise ValueError(f"random_count={count} exceeds available pool={len(pool)} after exclusions")
    rng = random.Random(int(seed))
    return rng.sample(pool, int(count))


def build_selected_words(
    all_words: Sequence[str],
    hard_cases_path: Path,
    hard_count: int,
    random_count: int,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    word_set = set(all_words)
    hard_raw = load_hard_words(hard_cases_path, hard_count)
    hard_words: List[str] = []
    missing: List[str] = []
    for w in hard_raw:
        if w in word_set:
            hard_words.append(w)
        else:
            missing.append(w)
    hard_set = set(hard_words)
    random_words = deterministic_random_words(all_words, hard_set, random_count, random_seed)

    records: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for i, w in enumerate(hard_words):
        if w not in seen:
            records.append({"secret": w, "case_source": "hard50", "case_rank": i})
            seen.add(w)
    for i, w in enumerate(random_words):
        if w not in seen:
            records.append({"secret": w, "case_source": "random1000", "case_rank": i})
            seen.add(w)

    meta = {
        "words_total": len(all_words),
        "hard_cases_path": str(hard_cases_path),
        "hard_count_requested": hard_count,
        "hard_count_loaded": len(hard_words),
        "random_count_requested": random_count,
        "random_count_loaded": len(random_words),
        "random_seed": random_seed,
        "selected_unique_words": len(records),
        "missing_hard_words": missing,
    }
    return records, meta


def build_game_plan(records: Sequence[Dict[str, Any]], tracks: Sequence[int], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    for rec in records:
        for track in tracks:
            for seed in seeds:
                games.append({
                    "secret": rec["secret"],
                    "case_source": rec["case_source"],
                    "case_rank": int(rec["case_rank"]),
                    "track": int(track),
                    "seed": int(seed),
                })
    for i, g in enumerate(games):
        g["game_index"] = i
    return games


def chunk_games(games: Sequence[Dict[str, Any]], chunk_index: Optional[int], chunk_count: Optional[int]) -> List[Dict[str, Any]]:
    if chunk_index is None and chunk_count is None:
        return list(games)
    if chunk_index is None or chunk_count is None:
        raise SystemExit("--chunk-index and --chunk-count must be provided together")
    if chunk_count <= 0 or chunk_index < 0 or chunk_index >= chunk_count:
        raise SystemExit(f"invalid chunk index/count: {chunk_index}/{chunk_count}")
    # Game-level modulo split keeps all chunks balanced across tracks and sources.
    return [g for i, g in enumerate(games) if i % chunk_count == chunk_index]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(x) for x in values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_rows(rows: Sequence[Dict[str, Any]], slow_threshold: int) -> Dict[str, Any]:
    def summarize_group(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}
        scores = [int(r.get("score_turns", 0)) for r in group]
        acts: List[float] = []
        turns_success = [int(r.get("turns", 0)) for r in group if int(r.get("success", 0)) == 1]
        statuses: Dict[str, int] = {}
        for r in group:
            statuses[str(r.get("status", ""))] = statuses.get(str(r.get("status", "")), 0) + 1
            try:
                acts.append(float(r.get("act_wall_s", 0)))
            except Exception:
                pass
        solved = sum(1 for r in group if int(r.get("success", 0)) == 1)
        return {
            "games": len(group),
            "solved": solved,
            "failed": len(group) - solved,
            "success_rate": round(solved / len(group), 6),
            "avg_score_turns": round(sum(scores) / len(scores), 4),
            "avg_success_turns": round(sum(turns_success) / len(turns_success), 4) if turns_success else None,
            "median_score_turns": percentile(scores, 0.50),
            "p90_score_turns": percentile(scores, 0.90),
            "p95_score_turns": percentile(scores, 0.95),
            "p99_score_turns": percentile(scores, 0.99),
            "max_score_turns": max(scores),
            "slow_games_count": sum(1 for x in scores if x > slow_threshold),
            "avg_act_wall_s": round(sum(acts) / len(acts), 4) if acts else 0.0,
            "statuses": dict(sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    by_track: Dict[str, List[Dict[str, Any]]] = {}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    by_source_track: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        track = str(r.get("track", ""))
        source = str(r.get("case_source", "unknown"))
        by_track.setdefault(track, []).append(r)
        by_source.setdefault(source, []).append(r)
        by_source_track.setdefault(f"{source}_T{track}", []).append(r)
    return {
        "overall": summarize_group(rows),
        "tracks": {k: summarize_group(v) for k, v in sorted(by_track.items())},
        "sources": {k: summarize_group(v) for k, v in sorted(by_source.items())},
        "source_tracks": {k: summarize_group(v) for k, v in sorted(by_source_track.items())},
    }


def write_outputs(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    slow_threshold: int,
    run_info: Dict[str, Any],
    selection_meta: Dict[str, Any],
    selected_records: Sequence[Dict[str, Any]],
    game_plan: Sequence[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r.get("game_index", 0)), str(r.get("secret", "")), int(r.get("track", 0)), int(r.get("seed", 0))))

    preferred = [
        "run_id", "game_index", "case_source", "case_rank", "track", "seed", "problem_id", "secret",
        "success", "status", "turns", "score_turns", "guesses", "submits", "start_wall_s", "act_wall_s",
        "total_wall_s", "last_action", "last_word", "error",
    ]
    extra = sorted({k for r in rows for k in r.keys()} - set(preferred))
    fields = [f for f in preferred if any(f in r for r in rows)] + extra

    games_csv = out_dir / "games.csv"
    with games_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    games_jsonl = out_dir / "games.jsonl"
    with games_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    slow_by_secret: Dict[str, List[Dict[str, Any]]] = {}
    failed_by_secret: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        score = int(row.get("score_turns", 0))
        entry = {
            "case_source": row.get("case_source", ""),
            "track": int(row.get("track", 0)),
            "seed": int(row.get("seed", 0)),
            "success": bool(int(row.get("success", 0))),
            "status": row.get("status", ""),
            "score_turns": score,
            "turns": int(row.get("turns", 0)),
            "act_wall_s": row.get("act_wall_s", ""),
            "last_word": row.get("last_word", ""),
        }
        if score > slow_threshold:
            slow_by_secret.setdefault(str(row.get("secret", "")), []).append(entry)
        if int(row.get("success", 1)) == 0:
            failed_by_secret.setdefault(str(row.get("secret", "")), []).append(entry)

    slow_list = sorted(
        ({"secret": s, "max_score_turns": max(e["score_turns"] for e in es), "cases": sorted(es, key=lambda e: -e["score_turns"])} for s, es in slow_by_secret.items()),
        key=lambda x: (-int(x["max_score_turns"]), str(x["secret"])),
    )
    failed_list = sorted(({"secret": s, "failures": es} for s, es in failed_by_secret.items()), key=lambda x: str(x["secret"]))

    (out_dir / "slow_words.json").write_text(json.dumps({"threshold_turns": slow_threshold, "count": len(slow_list), "words": slow_list}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "failed_words.json").write_text(json.dumps({"count": len(failed_list), "words": failed_list}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "selected_words.json").write_text(json.dumps({"metadata": selection_meta, "words": list(selected_records)}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "game_plan.json").write_text(json.dumps({"count": len(game_plan), "games": list(game_plan)}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "name": "semi_full_benchmark",
        "run": run_info,
        "metadata": selection_meta,
        "summary": summarize_rows(rows, slow_threshold),
        "slow_words_count": len(slow_list),
        "failed_words_count": len(failed_list),
        "outputs": {
            "games_csv": str(games_csv),
            "games_jsonl": str(games_jsonl),
            "slow_words": str(out_dir / "slow_words.json"),
            "failed_words": str(out_dir / "failed_words.json"),
            "selected_words": str(out_dir / "selected_words.json"),
            "game_plan": str(out_dir / "game_plan.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  {games_csv}")
    print(f"  {games_jsonl}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'slow_words.json'}")
    print(f"  {out_dir / 'failed_words.json'}")


def print_summary(rows: List[Dict[str, Any]], slow_threshold: int) -> None:
    summary = summarize_rows(rows, slow_threshold)
    print("\n=== summary ===")
    for label, data in [("all", summary["overall"])] + [(f"T{k}", v) for k, v in summary["tracks"].items()]:
        print(
            f"{label:>5} games={data.get('games', 0):>5} solved={data.get('solved', 0):>5} "
            f"succ={100*data.get('success_rate', 0):6.2f}% avg={data.get('avg_score_turns', 0):7.3f} "
            f"p95={data.get('p95_score_turns', 0):5.1f} slow>{slow_threshold}={data.get('slow_games_count', 0):>4} "
            f"max={data.get('max_score_turns', 0):>3} statuses={data.get('statuses', {})}"
        )
    for label, data in summary["sources"].items():
        print(
            f"{label:>10} games={data.get('games', 0):>5} solved={data.get('solved', 0):>5} "
            f"succ={100*data.get('success_rate', 0):6.2f}% avg={data.get('avg_score_turns', 0):7.3f} "
            f"max={data.get('max_score_turns', 0):>3}"
        )


def load_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_command(args: argparse.Namespace) -> int:
    merge_root = resolve_path(args.merge_root, "merge root")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (DEFAULT_RESULTS_ROOT / "semi_full_merged")
    rows_by_key: Dict[str, Dict[str, Any]] = {}
    csv_files = sorted(merge_root.rglob("games.csv"))
    if not csv_files:
        raise SystemExit(f"No games.csv files found under {merge_root}")
    for path in csv_files:
        for row in load_rows_from_csv(path):
            key = game_key_from_row(row)
            rows_by_key.setdefault(key, row)
    rows = list(rows_by_key.values())

    metadata: Dict[str, Any] = {"note": "merged chunk results", "input_games_csv_count": len(csv_files)}
    selected_records: List[Dict[str, Any]] = []
    game_plan: List[Dict[str, Any]] = []
    for path in sorted(merge_root.rglob("selected_words.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            metadata = obj.get("metadata", metadata)
            selected_records = obj.get("words", selected_records)
            break
        except Exception:
            pass

    run_info = {
        "run_id": out_dir.name,
        "mode": "merge",
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "merge_root": str(merge_root),
        "input_games_csv_count": len(csv_files),
        "merged_games": len(rows),
        "slow_threshold_turns": args.slow_threshold,
    }
    write_outputs(out_dir, rows, args.slow_threshold, run_info, metadata, selected_records, game_plan)
    print_summary(rows, args.slow_threshold)
    return 0


def run_command(args: argparse.Namespace) -> int:
    started_at = now_kst()
    solver_path = resolve_path(args.solver, "solver")
    words_path = resolve_path(args.words, "words.txt")
    cases_path = resolve_path(args.cases_file, "cases file", must_exist=args.hard_count > 0)
    all_words = load_words_txt(words_path)
    tracks = parse_int_list_or_range(args.tracks, allowed={1, 2, 3})
    seeds = parse_int_list_or_range(args.seeds)

    selected_records, selection_meta = build_selected_words(all_words, cases_path, args.hard_count, args.random_count, args.random_seed)
    full_plan = build_game_plan(selected_records, tracks, seeds)
    chunk_plan = chunk_games(full_plan, args.chunk_index, args.chunk_count)
    chunk_suffix = "" if args.chunk_index is None else f"_chunk{args.chunk_index:02d}of{args.chunk_count:02d}"

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        stamp = started_at.strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_RESULTS_ROOT / f"semi_full_benchmark_{stamp}{chunk_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"

    completed, existing_rows = load_checkpoint(checkpoint_path)
    remaining = len(chunk_plan) - len(completed)
    print("=== semi_full_benchmark ===")
    print(f"solver       : {solver_path}")
    print(f"words        : {words_path} ({len(all_words)})")
    print(f"cases file   : {cases_path}")
    print(f"hard words   : {selection_meta['hard_count_loaded']}")
    print(f"random words : {selection_meta['random_count_loaded']} (seed={args.random_seed})")
    print(f"tracks/seeds : {tracks} / {seeds}")
    print(f"all games    : {len(full_plan)}")
    print(f"chunk games  : {len(chunk_plan)}" + (f"  (chunk {args.chunk_index + 1}/{args.chunk_count})" if args.chunk_count else ""))
    print(f"completed    : {len(completed)}")
    print(f"remaining    : {remaining}")
    print(f"out_dir      : {out_dir}")
    print()

    if remaining == 0:
        write_outputs(out_dir, existing_rows, args.slow_threshold, {"note": "all_completed_at_start"}, selection_meta, selected_records, chunk_plan)
        print_summary(existing_rows, args.slow_threshold)
        return 0

    solver_module = None
    if args.mode == "direct":
        solver_module = load_solver_module(solver_path)

    new_rows: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()
    for game in chunk_plan:
        key = game_key(game["secret"], game["track"], game["seed"])
        if key in completed:
            continue
        if args.problem_id_mode == "full":
            problem_id = f"full_{game['secret']}"
        elif args.problem_id_mode == "secret":
            problem_id = str(game["secret"])
        else:
            problem_id = f"semi_{game['case_source']}_{game['secret']}_t{game['track']}_seed{game['seed']}"
        problem = Problem(problem_id=problem_id, secret_word=game["secret"], candidate_words=list(all_words))
        if args.mode == "direct":
            assert solver_module is not None
            result = run_direct_game(solver_module, problem, game["track"], game["seed"], args.max_turns, args.budget, args.score_cap, trace_enabled=False)
        else:
            result = run_http_game(solver_path, problem, game["track"], game["seed"], args.max_turns, args.budget, args.score_cap, trace_enabled=False, request_timeout=args.http_request_timeout)
        row = result_to_row(result)
        row["case_source"] = game["case_source"]
        row["case_rank"] = game["case_rank"]
        row["game_index"] = game["game_index"]
        row["run_id"] = out_dir.name
        append_checkpoint(checkpoint_path, row)
        new_rows.append(row)
        completed.add(key)

        done = len(existing_rows) + len(new_rows)
        elapsed = time.perf_counter() - start_wall
        rate = len(new_rows) / elapsed if elapsed > 0 else 0.0
        eta = (remaining - len(new_rows)) / rate if rate > 0 else 0.0
        mark = "OK  " if result.success else "FAIL"
        print(
            f"[{done:>5}/{len(chunk_plan)}] {mark} {game['case_source']:<10s} T{game['track']} seed={game['seed']} "
            f"secret={game['secret']} status={result.status:14s} score={result.score_turns:>3} "
            f"act={result.act_wall_s:6.2f}s ETA={eta/60:5.1f}m",
            flush=True,
        )

    finished_at = now_kst()
    elapsed_total = time.perf_counter() - start_wall
    rows = existing_rows + new_rows
    run_info = {
        "run_id": out_dir.name,
        "started_at": iso_dt(started_at),
        "finished_at": iso_dt(finished_at),
        "elapsed_s": round(elapsed_total, 3),
        "solver": str(solver_path),
        "words": str(words_path),
        "mode": args.mode,
        "tracks": tracks,
        "seeds": seeds,
        "chunk_index": args.chunk_index,
        "chunk_count": args.chunk_count,
        "chunk_games": len(chunk_plan),
        "all_games": len(full_plan),
        "problem_id_mode": args.problem_id_mode,
        "slow_threshold_turns": args.slow_threshold,
    }
    write_outputs(out_dir, rows, args.slow_threshold, run_info, selection_meta, selected_records, chunk_plan)
    print_summary(rows, args.slow_threshold)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="semi_full_benchmark runner and merger")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run one full or chunked semi_full_benchmark")
    run.add_argument("--solver", default=str(DEFAULT_SOLVER), help="solver file path; default: team00.py")
    run.add_argument("--words", default=str(DEFAULT_WORDS), help="words.txt path")
    run.add_argument("--cases-file", "--hard-cases", dest="cases_file", default=str(DEFAULT_CASES_FILE), help="debug_cases_slow_failed_top50.json path")
    run.add_argument("--hard-count", type=int, default=DEFAULT_HARD_COUNT, help="number of slow/failed words from cases file")
    run.add_argument("--random-count", type=int, default=DEFAULT_RANDOM_COUNT, help="number of random non-hard words")
    run.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="deterministic random sample seed")
    run.add_argument("--tracks", default="1,2,3", help="tracks to run, default 1,2,3")
    run.add_argument("--seeds", default="0", help="noise seeds, default 0")
    run.add_argument("--mode", choices=("direct", "http"), default="direct")
    run.add_argument("--max-turns", type=int, default=100)
    run.add_argument("--budget", type=float, default=60.0)
    run.add_argument("--score-cap", type=int, default=100)
    run.add_argument("--slow-threshold", type=int, default=DEFAULT_SLOW_THRESHOLD)
    run.add_argument("--chunk-index", type=int, default=None, help="0-based chunk index")
    run.add_argument("--chunk-count", type=int, default=None, help="total chunks")
    run.add_argument("--out-dir", default=None, help="output directory")
    run.add_argument("--http-request-timeout", type=float, default=65.0)
    run.add_argument("--problem-id-mode", choices=("semi", "full", "secret"), default="full", help="problem_id scheme; 'full' matches run_full_benchmark noise streams")
    run.set_defaults(func=run_command)

    merge = sub.add_parser("merge", help="merge chunk artifacts into one summary")
    merge.add_argument("--merge-root", required=True, help="directory containing downloaded chunk artifacts")
    merge.add_argument("--out-dir", default=None, help="merged output directory")
    merge.add_argument("--slow-threshold", type=int, default=DEFAULT_SLOW_THRESHOLD)
    merge.set_defaults(func=merge_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # Backward-compatible default: no subcommand means run.
        args = parser.parse_args(["run"] + list(argv or sys.argv[1:]))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
