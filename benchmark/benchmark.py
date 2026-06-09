#!/usr/bin/env python3
"""Local benchmark harness for the DCCP noisy Wordle project.

This script can benchmark a submitted solver in two modes:

1. direct (default): imports team00.py and calls initialize_problem/solver_act.
   This is fast and is best for algorithm tuning.
2. http: starts `python team00.py` as a subprocess and talks to /start_problem
   and /act using the official HTTP protocol. This is slower but validates server
   packaging/protocol behavior.

The grader here is intentionally self-contained and uses only the standard library
plus numpy. It implements the matching, verbal feedback, noisy effective-secret
sampling, first-three-turn force-noise rule, and score-turn accounting from the
project specification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("benchmark.py requires numpy. Install with: pip install numpy") from exc

ORDINALS = ("first", "second", "third", "fourth", "fifth")
DIGIT_TO_STATUS = {"0": "absent", "1": "misplaced", "2": "correct"}
POW3 = (81, 27, 9, 3, 1)
PAIRS = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
TRACKS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 0.0),
    2: (0.33, 0.33),
    3: (0.33, 0.66),
}


@dataclass
class Problem:
    problem_id: str
    secret_word: str
    candidate_words: List[str]


@dataclass
class GameResult:
    track: int
    seed: int
    problem_id: str
    secret: str
    success: bool
    status: str
    turns: int
    score_turns: int
    guesses: int
    submits: int
    start_wall_s: float
    act_wall_s: float
    total_wall_s: float
    last_word: str = ""
    last_action: str = ""
    error: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)


def valid_word(word: Any) -> bool:
    return isinstance(word, str) and len(word) == 5 and word.isalpha() and word.islower()


def load_problem(path: Path, problem_id: str = "bench") -> Problem:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    secret = data.get("secret_word")
    words = data.get("candidate_words")
    if not valid_word(secret):
        raise ValueError(f"Invalid secret_word in {path}: {secret!r}")
    if not isinstance(words, list) or not words:
        raise ValueError(f"Invalid candidate_words in {path}: must be a non-empty list")
    cleaned: List[str] = []
    seen = set()
    for i, w in enumerate(words):
        if not valid_word(w):
            raise ValueError(f"Invalid candidate_words[{i}] in {path}: {w!r}")
        if w not in seen:
            seen.add(w)
            cleaned.append(w)
    if secret not in seen:
        raise ValueError("secret_word must be included in candidate_words")
    return Problem(problem_id=problem_id, secret_word=secret, candidate_words=cleaned)


def parse_int_list_or_range(text: str, allowed: Optional[set] = None) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) not in (2, 3):
                raise ValueError(f"Bad range syntax: {part!r}")
            start = int(bits[0])
            stop = int(bits[1])
            step = int(bits[2]) if len(bits) == 3 else 1
            if step == 0:
                raise ValueError("Range step cannot be 0")
            out.extend(range(start, stop, step))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer")
    if allowed is not None:
        bad = [x for x in out if x not in allowed]
        if bad:
            raise ValueError(f"Unsupported value(s): {bad}; allowed={sorted(allowed)}")
    # Preserve order while removing duplicates.
    dedup: List[int] = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def stable_u64(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little", signed=False)


def feedback_code(guess: str, secret: str) -> str:
    """Duplicate-aware Wordle feedback code, matching the project spec."""
    out = ["0"] * 5
    counts: Dict[str, int] = {}
    for ch in secret:
        counts[ch] = counts.get(ch, 0) + 1
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            out[i] = "2"
            counts[g] -= 1
    for i, g in enumerate(guess):
        if out[i] == "2":
            continue
        if counts.get(g, 0) > 0:
            out[i] = "1"
            counts[g] -= 1
    return "".join(out)


def pid_from_code(code: str) -> int:
    return sum(int(d) * mul for d, mul in zip(code, POW3))


def rule_verbalize(guess: str, code: str) -> str:
    return "\n".join(
        f"The {name} letter '{ch}' is {DIGIT_TO_STATUS[d]}."
        for name, ch, d in zip(ORDINALS, guess, code)
    )


def mutate_one(secret: str, rng: np.random.Generator) -> str:
    letters = list(secret)
    pos = int(rng.integers(0, 5))
    orig = ord(letters[pos]) - 97
    repl = int(rng.integers(0, 25))
    if repl >= orig:
        repl += 1
    letters[pos] = chr(97 + repl)
    return "".join(letters)


def mutate_two(secret: str, rng: np.random.Generator) -> str:
    letters = list(secret)
    pair = PAIRS[int(rng.integers(0, len(PAIRS)))]
    for pos in pair:
        orig = ord(letters[pos]) - 97
        repl = int(rng.integers(0, 25))
        if repl >= orig:
            repl += 1
        letters[pos] = chr(97 + repl)
    return "".join(letters)


class NoiseOracle:
    """Deterministic per-game noisy feedback generator."""

    def __init__(
        self,
        secret: str,
        max_turns: int,
        p_one: float,
        p_two: float,
        force_noise_turns: int,
        seed: int,
        problem_id: str,
        track: int,
    ) -> None:
        self.secret = secret
        self.max_turns = int(max_turns)
        self.p_one = float(p_one)
        self.p_two = float(p_two)
        self.force_noise_turns = int(force_noise_turns)
        self.seed = int(seed)
        self.problem_id = str(problem_id)
        self.track = int(track)
        self.rng = np.random.default_rng(stable_u64("dccp-wordle-benchmark", self.seed, self.problem_id, self.track))
        self.types = self._sample_types()
        self.effective_secrets = self._sample_effective_secrets()

    def _sample_types(self) -> List[int]:
        types: List[int] = []
        p = max(0.0, self.p_one)
        q = max(0.0, self.p_two)
        for _ in range(self.max_turns):
            u = float(self.rng.random())
            if u < p:
                types.append(1)
            elif u < p + q:
                types.append(2)
            else:
                types.append(0)
        f = max(0, min(self.force_noise_turns, self.max_turns))
        if f > 0 and all(t == 0 for t in types[:f]):
            idx = int(self.rng.integers(0, f))
            if p + q <= 0.0:
                forced = 1
            else:
                forced = 1 if float(self.rng.random()) < p / (p + q) else 2
            types[idx] = forced
        return types

    def _sample_effective_secrets(self) -> List[str]:
        eff: List[str] = []
        for typ in self.types:
            if typ == 1:
                eff.append(mutate_one(self.secret, self.rng))
            elif typ == 2:
                eff.append(mutate_two(self.secret, self.rng))
            else:
                eff.append(self.secret)
        return eff

    def feedback_for_guess(self, guess: str, turn: int) -> Tuple[str, str, int, str]:
        idx = max(0, int(turn) - 1)
        eff_secret = self.effective_secrets[idx]
        noise_type = self.types[idx]
        code = feedback_code(guess, eff_secret)
        return rule_verbalize(guess, code), code, noise_type, eff_secret


def load_solver_module(path: Path):
    spec = importlib.util.spec_from_file_location("bench_solver_module", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_solver_module"] = module
    spec.loader.exec_module(module)
    if hasattr(module, "initialize_problem") and hasattr(module, "solver_act"):
        module.__bench_direct_kind__ = "functions"
        return module
    if hasattr(module, "NoisyWordleSolver"):
        module.__bench_direct_kind__ = "class"
        return module
    raise AttributeError(
        f"Direct mode requires {path.name} to expose either "
        "initialize_problem/solver_act or a NoisyWordleSolver class. "
        "Use --mode http for a generic submission server."
    )


def make_payload(problem_id: str, words: List[str], p_one: float, p_two: float, max_turns: int, budget: float) -> Dict[str, Any]:
    return {
        "problem_id": problem_id,
        "candidate_words": words,
        "noise_probability": p_one,
        "two_letter_noise_probability": p_two,
        "force_noise_turns": 3,
        "max_turns": max_turns,
        "guess_time_budget": budget,
    }


def validate_action(resp: Any, candidates: set) -> Tuple[str, str]:
    if not isinstance(resp, dict):
        raise ValueError(f"response is not a JSON object/dict: {resp!r}")
    action = resp.get("action")
    word = resp.get("word")
    if action not in ("guess", "submit"):
        raise ValueError(f"invalid action: {action!r}")
    if word not in candidates:
        raise ValueError(f"invalid word for action {action!r}: {word!r}")
    return str(action), str(word)


def run_direct_game(
    solver_module: Any,
    problem: Problem,
    track: int,
    seed: int,
    max_turns: int,
    budget: float,
    score_cap: int,
    trace_enabled: bool = False,
) -> GameResult:
    p_one, p_two = TRACKS[track]
    candidates = set(problem.candidate_words)
    oracle = NoiseOracle(problem.secret_word, max_turns, p_one, p_two, 3, seed, problem.problem_id, track)
    payload = make_payload(problem.problem_id, problem.candidate_words, p_one, p_two, max_turns, budget)

    direct_kind = getattr(solver_module, "__bench_direct_kind__", "functions")
    start_t = time.perf_counter()
    try:
        if direct_kind == "class":
            st = solver_module.NoisyWordleSolver(payload)
        else:
            st = solver_module.initialize_problem(payload)
    except Exception as exc:
        return GameResult(
            track=track,
            seed=seed,
            problem_id=problem.problem_id,
            secret=problem.secret_word,
            success=False,
            status="start_crash",
            turns=0,
            score_turns=score_cap,
            guesses=0,
            submits=0,
            start_wall_s=time.perf_counter() - start_t,
            act_wall_s=0.0,
            total_wall_s=time.perf_counter() - start_t,
            error=f"{type(exc).__name__}: {exc}",
        )
    start_wall = time.perf_counter() - start_t

    feedback: Optional[str] = None
    guesses = 0
    submits = 0
    act_wall = 0.0
    trace: List[Dict[str, Any]] = []
    last_word = ""
    last_action = ""
    status = "max_turns"
    success = False
    turns_used = max_turns

    for turn in range(1, max_turns + 1):
        before = time.perf_counter()
        try:
            if direct_kind == "class":
                resp = st.act(turn, feedback)
            else:
                resp = solver_module.solver_act(st, turn, feedback)
            action, word = validate_action(resp, candidates)
        except Exception as exc:
            act_wall += time.perf_counter() - before
            err = f"{type(exc).__name__}: {exc}"
            if trace_enabled:
                err += "\n" + traceback.format_exc(limit=4)
            return GameResult(
                track=track,
                seed=seed,
                problem_id=problem.problem_id,
                secret=problem.secret_word,
                success=False,
                status="act_crash_or_invalid",
                turns=turn,
                score_turns=score_cap,
                guesses=guesses,
                submits=submits,
                start_wall_s=start_wall,
                act_wall_s=act_wall,
                total_wall_s=start_wall + act_wall,
                last_word=last_word,
                last_action=last_action,
                error=err,
                trace=trace,
            )
        act_wall += time.perf_counter() - before
        last_action, last_word = action, word
        turns_used = turn

        if action == "submit":
            submits += 1
            success = word == problem.secret_word and turn < score_cap
            status = "solved" if success else "wrong_submit"
            if trace_enabled:
                trace.append({"turn": turn, "action": action, "word": word, "success": success})
            break

        guesses += 1
        feedback, code, noise_type, eff_secret = oracle.feedback_for_guess(word, turn)
        if trace_enabled:
            trace.append(
                {
                    "turn": turn,
                    "action": action,
                    "word": word,
                    "feedback_code": code,
                    "noise_type": noise_type,
                    "effective_secret": eff_secret,
                }
            )

        if act_wall > budget:
            status = "act_time_budget_exceeded"
            break
    else:
        status = "max_turns"

    score_turns = turns_used if success else score_cap
    return GameResult(
        track=track,
        seed=seed,
        problem_id=problem.problem_id,
        secret=problem.secret_word,
        success=success,
        status=status,
        turns=turns_used,
        score_turns=score_turns,
        guesses=guesses,
        submits=submits,
        start_wall_s=start_wall,
        act_wall_s=act_wall,
        total_wall_s=start_wall + act_wall,
        last_word=last_word,
        last_action=last_action,
        trace=trace,
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw or "{}")


def start_http_solver(solver_path: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PORT"] = str(port)
    return subprocess.Popen(
        [sys.executable, str(solver_path.resolve())],
        cwd=str(solver_path.resolve().parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def terminate_process(proc: subprocess.Popen) -> str:
    stderr_text = ""
    if proc.poll() is None:
        proc.terminate()
        try:
            _, stderr_text = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_text = proc.communicate(timeout=2.0)
    else:
        try:
            _, stderr_text = proc.communicate(timeout=0.2)
        except Exception:
            stderr_text = ""
    return stderr_text[-2000:] if stderr_text else ""


def run_http_game(
    solver_path: Path,
    problem: Problem,
    track: int,
    seed: int,
    max_turns: int,
    budget: float,
    score_cap: int,
    trace_enabled: bool = False,
    request_timeout: float = 65.0,
) -> GameResult:
    p_one, p_two = TRACKS[track]
    candidates = set(problem.candidate_words)
    oracle = NoiseOracle(problem.secret_word, max_turns, p_one, p_two, 3, seed, problem.problem_id, track)
    payload = make_payload(problem.problem_id, problem.candidate_words, p_one, p_two, max_turns, budget)
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_http_solver(solver_path, port)
    feedback: Optional[str] = None
    guesses = 0
    submits = 0
    trace: List[Dict[str, Any]] = []
    last_word = ""
    last_action = ""
    status = "max_turns"
    success = False
    turns_used = 0
    start_begin = time.perf_counter()
    start_wall = 0.0
    act_wall = 0.0
    error = ""

    try:
        # The official grader waits for server startup, but for convenience we poll
        # until the server is accepting connections, then send /start_problem once.
        deadline = time.perf_counter() + 5.0
        while True:
            try:
                post_json(base + "/start_problem", payload, timeout=5.0)
                break
            except urllib.error.URLError as exc:
                if time.perf_counter() >= deadline or proc.poll() is not None:
                    raise RuntimeError(f"could not start HTTP solver: {exc}") from exc
                time.sleep(0.05)
        start_wall = time.perf_counter() - start_begin

        for turn in range(1, max_turns + 1):
            turns_used = turn
            req_payload = {"problem_id": problem.problem_id, "feedback": feedback, "turn": turn}
            before = time.perf_counter()
            try:
                resp = post_json(base + "/act", req_payload, timeout=request_timeout)
                action, word = validate_action(resp, candidates)
            except Exception as exc:
                act_wall += time.perf_counter() - before
                status = "http_act_error_or_invalid"
                error = f"{type(exc).__name__}: {exc}"
                break
            act_wall += time.perf_counter() - before
            last_action, last_word = action, word

            if action == "submit":
                submits += 1
                success = word == problem.secret_word and turn < score_cap
                status = "solved" if success else "wrong_submit"
                if trace_enabled:
                    trace.append({"turn": turn, "action": action, "word": word, "success": success})
                break

            guesses += 1
            feedback, code, noise_type, eff_secret = oracle.feedback_for_guess(word, turn)
            if trace_enabled:
                trace.append(
                    {
                        "turn": turn,
                        "action": action,
                        "word": word,
                        "feedback_code": code,
                        "noise_type": noise_type,
                        "effective_secret": eff_secret,
                    }
                )
            if act_wall > budget:
                status = "act_time_budget_exceeded"
                break
    except Exception as exc:
        status = "http_start_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stderr_tail = terminate_process(proc)
        if stderr_tail and not error:
            error = stderr_tail

    score_turns = turns_used if success else score_cap
    return GameResult(
        track=track,
        seed=seed,
        problem_id=problem.problem_id,
        secret=problem.secret_word,
        success=success,
        status=status,
        turns=turns_used,
        score_turns=score_turns,
        guesses=guesses,
        submits=submits,
        start_wall_s=start_wall,
        act_wall_s=act_wall,
        total_wall_s=start_wall + act_wall,
        last_word=last_word,
        last_action=last_action,
        error=error,
        trace=trace,
    )


def choose_secret_words(args: argparse.Namespace, base_problem: Problem) -> List[str]:
    words = base_problem.candidate_words
    if args.secret:
        secrets = [args.secret]
    elif args.all_secrets:
        secrets = list(words)
    elif args.sample_secrets is not None:
        n = min(int(args.sample_secrets), len(words))
        rng = np.random.default_rng(int(args.secret_sample_seed))
        idx = rng.choice(len(words), size=n, replace=False)
        secrets = [words[int(i)] for i in idx]
    else:
        secrets = [base_problem.secret_word]
    for secret in secrets:
        if secret not in set(words):
            raise ValueError(f"Benchmark secret {secret!r} is not in candidate_words")
    return secrets


def iter_games(args: argparse.Namespace, base_problem: Problem) -> Iterable[Tuple[Problem, int, int]]:
    tracks = parse_int_list_or_range(args.tracks, allowed={1, 2, 3})
    seeds = parse_int_list_or_range(args.seeds)
    secrets = choose_secret_words(args, base_problem)
    count = 0
    for track in tracks:
        for seed in seeds:
            for i, secret in enumerate(secrets):
                if args.max_games is not None and count >= args.max_games:
                    return
                pid = f"{base_problem.problem_id}_t{track}_seed{seed}_{i:04d}_{secret}"
                yield Problem(pid, secret, base_problem.candidate_words), track, seed
                count += 1


def result_to_row(r: GameResult, run_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_info = run_info or {}
    return {
        "run_id": run_info.get("run_id", ""),
        "run_started_at": run_info.get("run_started_at", ""),
        "run_started_date": run_info.get("run_started_date", ""),
        "run_started_time": run_info.get("run_started_time", ""),
        "run_timezone": run_info.get("timezone", ""),
        "track": r.track,
        "seed": r.seed,
        "problem_id": r.problem_id,
        "secret": r.secret,
        "success": int(r.success),
        "status": r.status,
        "turns": r.turns,
        "score_turns": r.score_turns,
        "guesses": r.guesses,
        "submits": r.submits,
        "start_wall_s": f"{r.start_wall_s:.6f}",
        "act_wall_s": f"{r.act_wall_s:.6f}",
        "total_wall_s": f"{r.total_wall_s:.6f}",
        "last_action": r.last_action,
        "last_word": r.last_word,
        "error": r.error.replace("\n", "\\n")[:500],
    }


def write_outputs(out_dir: Path, results: List[GameResult], args: argparse.Namespace, run_info: Dict[str, Any]) -> Tuple[Path, Path, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_games.csv"
    jsonl_path = out_dir / "benchmark_games.jsonl"
    summary_path = out_dir / "benchmark_summary.json"
    fields = list(result_to_row(results[0], run_info).keys()) if results else ["run_id", "run_started_at", "track", "seed", "problem_id", "secret", "success", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(result_to_row(r, run_info))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            obj = result_to_row(r, run_info)
            if args.trace:
                obj["trace"] = r.trace
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    summary = summarize_results(results)
    summary["run"] = dict(run_info)
    summary["outputs"] = {"results_dir": str(out_dir)}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return csv_path, jsonl_path, summary_path


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    pos = (len(arr) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def summarize_results(results: List[GameResult]) -> Dict[str, Any]:
    by_track: Dict[int, List[GameResult]] = {}
    for r in results:
        by_track.setdefault(r.track, []).append(r)
    summary: Dict[str, Any] = {"overall": {}, "tracks": {}}
    for key, group in [("overall", results)] + [(str(track), rows) for track, rows in sorted(by_track.items())]:
        if not group:
            continue
        statuses: Dict[str, int] = {}
        for r in group:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        solved = sum(1 for r in group if r.success)
        turns_success = [r.turns for r in group if r.success]
        data = {
            "games": len(group),
            "solved": solved,
            "success_rate": solved / len(group),
            "avg_score_turns": sum(r.score_turns for r in group) / len(group),
            "avg_success_turns": (sum(turns_success) / len(turns_success)) if turns_success else None,
            "median_score_turns": percentile([r.score_turns for r in group], 0.50),
            "p90_score_turns": percentile([r.score_turns for r in group], 0.90),
            "max_score_turns": max(r.score_turns for r in group),
            "ge15_score_turns": sum(1 for r in group if r.score_turns >= 15),
            "avg_act_wall_s": sum(r.act_wall_s for r in group) / len(group),
            "p95_act_wall_s": percentile([r.act_wall_s for r in group], 0.95),
            "statuses": statuses,
        }
        if key == "overall":
            summary["overall"] = data
        else:
            summary["tracks"][key] = data
    return summary


def print_summary(results: List[GameResult]) -> None:
    summary = summarize_results(results)
    rows: List[Tuple[str, Dict[str, Any]]] = [("all", summary["overall"])]
    for track, data in summary["tracks"].items():
        rows.append((f"T{track}", data))
    headers = ["scope", "games", "solved", "succ%", "avg_score", "avg_solved", "p90_score", ">=15", "max", "avg_act_s", "p95_act_s", "top_statuses"]
    formatted: List[List[str]] = []
    for scope, data in rows:
        statuses = sorted(data["statuses"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        formatted.append(
            [
                scope,
                str(data["games"]),
                str(data["solved"]),
                f"{100.0 * data['success_rate']:.1f}",
                f"{data['avg_score_turns']:.3f}",
                "-" if data["avg_success_turns"] is None else f"{data['avg_success_turns']:.3f}",
                f"{data['p90_score_turns']:.1f}",
                str(data["ge15_score_turns"]),
                str(data["max_score_turns"]),
                f"{data['avg_act_wall_s']:.3f}",
                f"{data['p95_act_wall_s']:.3f}",
                ", ".join(f"{k}:{v}" for k, v in statuses),
            ]
        )
    widths = [len(h) for h in headers]
    for row in formatted:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in formatted:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def print_trace(result: GameResult, max_lines: int = 200) -> None:
    print(f"\nTrace: track={result.track} seed={result.seed} secret={result.secret} status={result.status} score={result.score_turns}")
    for item in result.trace[:max_lines]:
        if item.get("action") == "guess":
            print(
                f"  turn {item['turn']:>2}: guess  {item['word']}  "
                f"fb={item['feedback_code']} noise={item['noise_type']} eff={item['effective_secret']}"
            )
        else:
            print(f"  turn {item['turn']:>2}: submit {item['word']} success={item.get('success')}")
    if len(result.trace) > max_lines:
        print(f"  ... {len(result.trace) - max_lines} more trace lines omitted")
    if result.error:
        print(f"  error: {result.error}")



# ---------------------------------------------------------------------------
# Unified words.txt benchmark CLI
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SOLVER = REPO_ROOT / "team00.py"
DEFAULT_WORDS = REPO_ROOT / "words.txt"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "latest"
DEFAULT_OPENERS = ("tares", "solan", "deils", "choli", "bunny")


def now_kst() -> datetime:
    """Return the current time in Asia/Seoul when zoneinfo is available."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Seoul"))
        except Exception:
            pass
    return datetime.now().astimezone()


def iso_dt(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def make_auto_results_dir(args: argparse.Namespace, started_at: datetime) -> Path:
    """Create a deterministic-looking but collision-safe default output directory."""
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{args.bench}_{args.mode}"
    base = DEFAULT_RESULTS_ROOT / base_name
    if not base.exists():
        return base.resolve()
    for i in range(1, 1000):
        candidate = DEFAULT_RESULTS_ROOT / f"{base_name}_{i:03d}"
        if not candidate.exists():
            return candidate.resolve()
    raise SystemExit(f"Could not choose a unique results directory under {DEFAULT_RESULTS_ROOT}")


def resolve_results_dir(args: argparse.Namespace, started_at: datetime) -> Path:
    if args.results_dir is None or str(args.results_dir).strip() == "":
        return make_auto_results_dir(args, started_at)
    return resolve_output_path(args.results_dir)


def resolve_existing_path(raw: str, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(REPO_ROOT / path)
        candidates.append(SCRIPT_DIR / path)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"{label} not found: {raw!r}. Tried: {tried}")


def resolve_output_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def parse_word_csv(text: Optional[str], arg_name: str) -> List[str]:
    if text is None or str(text).strip() == "":
        return []
    out: List[str] = []
    for item in str(text).replace("\n", ",").split(","):
        word = item.strip().lower()
        if not word:
            continue
        if not valid_word(word):
            raise ValueError(f"{arg_name} contains invalid word: {word!r}")
        out.append(word)
    return dedupe_ordered(out)


def load_words_txt(path: Path) -> List[str]:
    words: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            word = raw.strip().lower()
            if not word or word.startswith("#"):
                continue
            if not valid_word(word):
                raise ValueError(f"invalid word at {path}:{line_no}: {raw.rstrip()!r}")
            if word not in seen:
                seen.add(word)
                words.append(word)
    if not words:
        raise ValueError(f"no valid words found in {path}")
    return words


def load_word_list_file(path: Path, label: str) -> List[str]:
    words: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            # Allow both newline-separated and comma-separated files.
            for token in raw.replace(",", " ").split():
                word = token.strip().lower()
                if not word or word.startswith("#"):
                    continue
                if not valid_word(word):
                    raise ValueError(f"invalid {label} word at {path}:{line_no}: {token!r}")
                if word not in seen:
                    seen.add(word)
                    words.append(word)
    if not words:
        raise ValueError(f"{label} file is empty: {path}")
    return words


def dedupe_ordered(words: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in words:
        word = str(raw).strip().lower()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def validate_words_are_from_words_txt(words: Sequence[str], words_set: set, label: str) -> None:
    bad = [w for w in words if w not in words_set]
    if bad:
        shown = ", ".join(bad[:10])
        extra = "" if len(bad) <= 10 else f" ... and {len(bad) - 10} more"
        raise ValueError(f"{label} must be present in words.txt; missing: {shown}{extra}")


def parse_candidate_size(text: str, available_count: int) -> Optional[int]:
    raw = str(text).strip().lower()
    if raw in ("all", "full", "none", "0"):
        return None
    value = int(raw)
    if value < 2:
        raise ValueError("--candidate-size must be at least 2, or use 'all'")
    if value > available_count:
        raise ValueError(f"--candidate-size={value} exceeds available word count {available_count}")
    return value


def sample_words(rng: np.random.Generator, pool: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    if k > len(pool):
        raise ValueError(f"cannot sample {k} words from pool of size {len(pool)}")
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idx]


def maybe_shuffle(words: List[str], seed: int, enabled: bool) -> List[str]:
    if not enabled or len(words) <= 1:
        return list(words)
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    perm = rng.permutation(len(words))
    return [words[int(i)] for i in perm]


def select_candidates(
    *,
    words: List[str],
    secret: str,
    candidate_size: Optional[int],
    seed: int,
    policy: str,
    include_words: Sequence[str],
    exclude_words: Sequence[str],
    exact_candidates: Sequence[str],
    strict_candidates: bool,
    shuffle_candidates: bool,
) -> List[str]:
    words_set = set(words)
    exclude_set = set(exclude_words)
    if secret not in words_set:
        raise ValueError(f"secret is not in words.txt: {secret!r}")
    if secret in exclude_set:
        raise ValueError(f"secret cannot be excluded from candidates: {secret!r}")
    validate_words_are_from_words_txt(include_words, words_set, "--include-words")
    validate_words_are_from_words_txt(exact_candidates, words_set, "--candidates/--candidate-file")
    bad_include = [w for w in include_words if w in exclude_set]
    if bad_include:
        raise ValueError(f"words cannot be both included and excluded: {bad_include[:10]}")
    bad_exact = [w for w in exact_candidates if w in exclude_set]
    if bad_exact:
        raise ValueError(f"exact candidate words cannot be excluded: {bad_exact[:10]}")

    forced = dedupe_ordered([secret, *include_words])
    rng_seed = int(stable_u64("candidate-selection", seed, secret, policy)) & ((1 << 63) - 1)
    rng = np.random.default_rng(rng_seed)

    if exact_candidates:
        candidates = dedupe_ordered(exact_candidates)
        missing_forced = [w for w in forced if w not in set(candidates)]
        if missing_forced:
            if strict_candidates:
                raise ValueError(
                    "strict candidate list does not contain required word(s): "
                    + ", ".join(missing_forced)
                )
            candidates = dedupe_ordered([*missing_forced, *candidates])
        return maybe_shuffle(candidates, rng_seed + 17, shuffle_candidates)

    eligible = [w for w in words if w not in exclude_set]
    if candidate_size is None:
        candidates = list(eligible)
        # Keep explicit forced words at the front before optional shuffle; this also
        # guarantees that they appear when the original words.txt was filtered.
        candidates = dedupe_ordered([*forced, *candidates])
        return maybe_shuffle(candidates, rng_seed + 23, shuffle_candidates)

    if len(forced) > candidate_size:
        raise ValueError(
            f"candidate_size={candidate_size} is smaller than required words "
            f"(secret + includes = {len(forced)})"
        )
    forced_set = set(forced)
    pool = [w for w in eligible if w not in forced_set]
    need = candidate_size - len(forced)
    if need > len(pool):
        raise ValueError(f"not enough eligible words to build candidate list: need {need}, have {len(pool)}")
    if policy == "random":
        extra = sample_words(rng, pool, need)
    elif policy == "prefix":
        extra = list(pool[:need])
    elif policy == "suffix":
        extra = list(pool[-need:]) if need else []
    else:
        raise ValueError(f"unknown candidate policy: {policy!r}")
    candidates = dedupe_ordered([*forced, *extra])
    return maybe_shuffle(candidates, rng_seed + 31, shuffle_candidates)


def load_problem_file(path: Path) -> List[Problem]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_items = data if isinstance(data, list) else [data]
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("problem JSON must be one object or a non-empty list of objects")
    problems: List[Problem] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"problem item {i} is not an object")
        secret = item.get("secret_word")
        candidates = item.get("candidate_words")
        if not valid_word(secret):
            raise ValueError(f"problem item {i} has invalid secret_word: {secret!r}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"problem item {i} has invalid candidate_words")
        clean = []
        seen = set()
        for j, word in enumerate(candidates):
            if not valid_word(word):
                raise ValueError(f"problem item {i} candidate {j} invalid: {word!r}")
            if word not in seen:
                seen.add(word)
                clean.append(word)
        if secret not in seen:
            raise ValueError(f"problem item {i}: secret_word must be included in candidate_words")
        problem_id = str(item.get("problem_id") or f"{path.stem}_{i:04d}_{secret}")
        problems.append(Problem(problem_id=problem_id, secret_word=str(secret), candidate_words=clean))
    return problems


def load_exact_candidates(args: argparse.Namespace) -> List[str]:
    exact = parse_word_csv(args.candidates, "--candidates")
    if args.candidate_file:
        candidate_file = resolve_existing_path(args.candidate_file, "candidate file")
        exact.extend(load_word_list_file(candidate_file, "candidate"))
    return dedupe_ordered(exact)


def build_include_exclude_lists(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    include = parse_word_csv(args.include_words, "--include-words")
    if args.include_default_openers:
        include = dedupe_ordered([*include, *DEFAULT_OPENERS])
    exclude = parse_word_csv(args.exclude_words, "--exclude-words")
    if args.exclude_file:
        exclude_file = resolve_existing_path(args.exclude_file, "exclude file")
        exclude.extend(load_word_list_file(exclude_file, "exclude"))
    return dedupe_ordered(include), dedupe_ordered(exclude)


def choose_secrets_from_words(
    args: argparse.Namespace,
    words: List[str],
    exact_candidates: Sequence[str],
    exclude_words: Sequence[str],
) -> List[str]:
    words_set = set(words)
    exclude_set = set(exclude_words)
    explicit: List[str] = []
    if args.secret:
        explicit.extend(parse_word_csv(args.secret, "--secret"))
    if args.secrets:
        explicit.extend(parse_word_csv(args.secrets, "--secrets"))
    if args.secret_file:
        secret_file = resolve_existing_path(args.secret_file, "secret file")
        explicit.extend(load_word_list_file(secret_file, "secret"))
    explicit = dedupe_ordered(explicit)
    validate_words_are_from_words_txt(explicit, words_set, "secret words")
    bad = [w for w in explicit if w in exclude_set]
    if bad:
        raise ValueError(f"secret words cannot be excluded from candidates: {bad[:10]}")

    if args.bench == "single":
        if explicit:
            return [explicit[0]]
        if exact_candidates:
            pool = [w for w in exact_candidates if w not in exclude_set]
        else:
            pool = [w for w in words if w not in exclude_set]
        if not pool:
            raise ValueError("no available words from which to choose a secret")
        rng = np.random.default_rng(int(stable_u64("single-secret", args.problem_seed)) & ((1 << 63) - 1))
        return [pool[int(rng.integers(0, len(pool)))]]

    # Bulk mode.
    if args.all_secrets:
        return [w for w in words if w not in exclude_set]
    if explicit:
        return explicit
    pool = [w for w in words if w not in exclude_set]
    if not pool:
        raise ValueError("no available words from which to sample secrets")
    n = int(args.num_problems)
    if n <= 0:
        raise ValueError("--num-problems must be positive")
    rng = np.random.default_rng(int(stable_u64("bulk-secrets", args.problem_seed)) & ((1 << 63) - 1))
    replace = n > len(pool)
    idx = rng.choice(len(pool), size=n, replace=replace)
    return [pool[int(i)] for i in idx]


def build_problems_from_words(args: argparse.Namespace) -> Tuple[List[Problem], Dict[str, Any]]:
    words_path = resolve_existing_path(args.words, "words.txt")
    words = load_words_txt(words_path)
    words_set = set(words)
    include_words, exclude_words = build_include_exclude_lists(args)
    validate_words_are_from_words_txt(exclude_words, words_set, "--exclude-words")
    exact_candidates = load_exact_candidates(args)
    available_count = len([w for w in words if w not in set(exclude_words)])
    candidate_size = None if exact_candidates else parse_candidate_size(args.candidate_size, available_count)
    secrets = choose_secrets_from_words(args, words, exact_candidates, exclude_words)
    validate_words_are_from_words_txt(secrets, words_set, "secret words")

    shared_candidates: Optional[List[str]] = None
    if args.shared_candidates and len(secrets) > 1:
        shared_include = dedupe_ordered([*secrets, *include_words])
        shared_candidates = select_candidates(
            words=words,
            secret=secrets[0],
            candidate_size=candidate_size,
            seed=int(args.problem_seed),
            policy=args.candidate_policy,
            include_words=shared_include,
            exclude_words=exclude_words,
            exact_candidates=exact_candidates,
            strict_candidates=args.strict_candidates,
            shuffle_candidates=not args.no_shuffle_candidates,
        )

    problems: List[Problem] = []
    size_label = "exact" if exact_candidates else ("all" if candidate_size is None else str(candidate_size))
    for i, secret in enumerate(secrets):
        if shared_candidates is not None:
            candidates = list(shared_candidates)
            if secret not in set(candidates):
                raise ValueError(f"shared candidate list does not contain secret {secret!r}")
        else:
            cand_seed = int(stable_u64("problem-candidates", args.problem_seed, i, secret)) & ((1 << 63) - 1)
            candidates = select_candidates(
                words=words,
                secret=secret,
                candidate_size=candidate_size,
                seed=cand_seed,
                policy=args.candidate_policy,
                include_words=include_words,
                exclude_words=exclude_words,
                exact_candidates=exact_candidates,
                strict_candidates=args.strict_candidates,
                shuffle_candidates=not args.no_shuffle_candidates,
            )
        problem_id = f"words_{args.bench}_{args.candidate_policy}_{size_label}_{int(args.problem_seed)}_{i:04d}_{secret}"
        problems.append(Problem(problem_id=problem_id, secret_word=secret, candidate_words=candidates))

    meta = {
        "source": "words",
        "words_path": str(words_path),
        "words_count": len(words),
        "bench": args.bench,
        "candidate_size": args.candidate_size,
        "candidate_size_effective": "exact" if exact_candidates else ("all" if candidate_size is None else candidate_size),
        "candidate_policy": args.candidate_policy,
        "problem_seed": int(args.problem_seed),
        "shared_candidates": bool(args.shared_candidates),
        "include_words": include_words,
        "exclude_words_count": len(exclude_words),
        "exact_candidate_count": len(exact_candidates),
        "secrets_count": len(secrets),
    }
    return problems, meta


def candidate_digest(candidates: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=10)
    for word in candidates:
        h.update(word.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def official_problem_object(problem: Problem) -> Dict[str, Any]:
    return {"secret_word": problem.secret_word, "candidate_words": problem.candidate_words}


def full_problem_object(problem: Problem) -> Dict[str, Any]:
    return {
        "problem_id": problem.problem_id,
        "secret_word": problem.secret_word,
        "candidate_words": problem.candidate_words,
    }


def save_problem_artifacts(out_dir: Path, problems: List[Problem], config: Dict[str, Any], run_info: Dict[str, Any]) -> Tuple[Path, Path, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problems_path = out_dir / "benchmark_problems.json"
    manifest_path = out_dir / "benchmark_problem_manifest.json"
    single_path: Optional[Path] = None
    with problems_path.open("w", encoding="utf-8") as f:
        json.dump([full_problem_object(p) for p in problems], f, indent=2, ensure_ascii=False)
    if len(problems) == 1:
        single_path = out_dir / "single_problem_official_format.json"
        with single_path.open("w", encoding="utf-8") as f:
            json.dump(official_problem_object(problems[0]), f, indent=2, ensure_ascii=False)
    manifest = {
        "created_at": run_info.get("run_started_at"),
        "run": dict(run_info),
        "config": config,
        "num_problems": len(problems),
        "problems": [
            {
                "problem_id": p.problem_id,
                "secret_word": p.secret_word,
                "candidate_count": len(p.candidate_words),
                "secret_in_candidates": p.secret_word in set(p.candidate_words),
                "candidate_digest": candidate_digest(p.candidate_words),
            }
            for p in problems
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return problems_path, manifest_path, single_path


def save_config(out_dir: Path, args: argparse.Namespace, meta: Dict[str, Any], solver_path: Path, run_info: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "benchmark_config.json"
    config = {
        "run": dict(run_info),
        "results_dir": str(out_dir),
        "results_dir_was_auto": run_info.get("results_dir_was_auto", False),
        "solver": str(solver_path),
        "mode": args.mode,
        "bench": args.bench,
        "tracks": args.tracks,
        "seeds": args.seeds,
        "max_turns": args.max_turns,
        "score_cap": args.score_cap,
        "budget": args.budget,
        "source_meta": meta,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def save_run_metadata(out_dir: Path, run_info: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "benchmark_run.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    return run_path


def build_problems(args: argparse.Namespace) -> Tuple[List[Problem], Dict[str, Any]]:
    if args.problem:
        problem_path = resolve_existing_path(args.problem, "problem JSON")
        problems = load_problem_file(problem_path)
        if args.bench == "single" and len(problems) > 1 and not args.use_all_problem_file_items:
            problems = [problems[0]]
        if args.bench == "bulk" and args.num_problems and not args.use_all_problem_file_items:
            problems = problems[: int(args.num_problems)]
        meta = {
            "source": "problem_json",
            "problem_path": str(problem_path),
            "loaded_problem_count": len(problems),
        }
        return problems, meta
    return build_problems_from_words(args)


def expand_games(args: argparse.Namespace, problems: List[Problem]) -> List[Tuple[Problem, int, int]]:
    tracks = parse_int_list_or_range(args.tracks, allowed=set(TRACKS))
    seeds = parse_int_list_or_range(args.seeds)
    games: List[Tuple[Problem, int, int]] = []
    for problem in problems:
        for track in tracks:
            for seed in seeds:
                games.append((problem, track, seed))
                if args.max_games is not None and len(games) >= int(args.max_games):
                    return games
    return games


def game_print_line(idx: int, total: int, result: GameResult, problem: Problem) -> str:
    mark = "OK" if result.success else "FAIL"
    return (
        f"[{idx:>4}/{total}] {mark} T{result.track} seed={result.seed} "
        f"secret={problem.secret_word} cand={len(problem.candidate_words)} "
        f"status={result.status} score={result.score_turns} turns={result.turns} act_s={result.act_wall_s:.3f}"
    )


def copy_latest_files(out_dir: Path) -> None:
    # Every run gets its own directory by default. Also mirror the current
    # run into benchmark/results/latest for quick inspection. Recreate latest
    # so files from an older run cannot remain accidentally.
    latest = DEFAULT_RESULTS_DIR
    try:
        if out_dir.resolve() == latest.resolve():
            return
    except FileNotFoundError:
        pass
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    for name in (
        "benchmark_games.csv",
        "benchmark_games.jsonl",
        "benchmark_summary.json",
        "benchmark_problems.json",
        "benchmark_problem_manifest.json",
        "single_problem_official_format.json",
        "benchmark_config.json",
        "benchmark_run.json",
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, latest / name)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Unified local benchmark for the DCCP noisy Wordle solver. "
            "It can build candidate lists and secrets from words.txt, then run "
            "single or bulk benchmarks in direct or HTTP mode."
        )
    )
    p.add_argument("--solver", default=str(DEFAULT_SOLVER), help="solver file; default: ../team00.py from benchmark/")
    p.add_argument("--mode", choices=("direct", "http"), default="direct", help="direct imports solver; http tests official server protocol")
    p.add_argument("--bench", "--benchmark", dest="bench", choices=("single", "bulk"), default="single", help="single problem or bulk problem generation")

    source = p.add_argument_group("problem source")
    source.add_argument("--words", default=str(DEFAULT_WORDS), help="newline-separated 5-letter word list")
    source.add_argument("--problem", default=None, help="existing problem JSON; if set, words.txt generation is skipped")
    source.add_argument("--use-all-problem-file-items", action="store_true", help="when --problem is a list, use every item even in single mode")

    secrets = p.add_argument_group("secret selection from words.txt")
    secrets.add_argument("--secret", default=None, help="one secret word; in single mode this is the exact secret")
    secrets.add_argument("--secrets", default=None, help="comma-separated secret list for bulk mode")
    secrets.add_argument("--secret-file", default=None, help="file with newline/comma-separated secret words")
    secrets.add_argument("--all-secrets", action="store_true", help="bulk mode: use every word in words.txt as a secret")
    secrets.add_argument("--num-problems", type=int, default=10, help="bulk mode: number of random secrets when --secrets is absent")
    secrets.add_argument("--problem-seed", type=int, default=20260609, help="seed for selecting secrets and candidate subsets")

    candidates = p.add_argument_group("candidate list selection from words.txt")
    candidates.add_argument("--candidate-size", default="all", help="number of candidates, or 'all' to use the full words.txt list; ignored for exact --candidates/--candidate-file")
    candidates.add_argument("--candidate-policy", choices=("random", "prefix", "suffix"), default="random", help="how to choose sampled candidates from words.txt")
    candidates.add_argument("--candidates", default=None, help="exact comma-separated candidate words")
    candidates.add_argument("--candidate-file", default=None, help="exact candidate word file; combined with --candidates")
    candidates.add_argument("--include-words", default=None, help="comma-separated words forced into every candidate list")
    candidates.add_argument("--include-default-openers", action="store_true", help="force tares, solan, deils, choli, bunny into every generated list")
    candidates.add_argument("--exclude-words", default=None, help="comma-separated words excluded from random/prefix/suffix candidate generation")
    candidates.add_argument("--exclude-file", default=None, help="file with words to exclude")
    candidates.add_argument("--shared-candidates", action="store_true", help="bulk mode: use one shared candidate list for all generated secrets")
    candidates.add_argument("--strict-candidates", action="store_true", help="raise an error if exact candidates omit the secret/include words")
    candidates.add_argument("--no-shuffle-candidates", action="store_true", help="preserve candidate order instead of deterministic shuffle")

    run = p.add_argument_group("benchmark run")
    run.add_argument("--tracks", default="1,2,3", help="tracks to run, e.g. 1,2,3 or 2")
    run.add_argument("--seeds", default="0", help="noise seeds, e.g. 0,1,2 or 0:20")
    run.add_argument("--max-games", type=int, default=None, help="stop after this many track/seed/problem games")
    run.add_argument("--max-turns", type=int, default=100, help="per-game max turns")
    run.add_argument("--score-cap", type=int, default=100, help="failure score-turn count")
    run.add_argument("--budget", type=float, default=60.0, help="total /act wall-clock budget per game")
    run.add_argument("--http-request-timeout", type=float, default=65.0, help="per-request timeout in HTTP mode")
    run.add_argument("--trace", action="store_true", help="store per-turn traces in JSONL and print failed traces")
    run.add_argument("--print-all-traces", action="store_true", help="print traces for every game; implies --trace")

    output = p.add_argument_group("outputs")
    output.add_argument("--results-dir", "--out-dir", dest="results_dir", default=None, help="directory for saved benchmark outputs; default: benchmark/results/<timestamp>_<bench>_<mode>")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_started_at = now_kst()
    if args.print_all_traces:
        args.trace = True
    solver_path = resolve_existing_path(args.solver, "solver")
    out_dir = resolve_results_dir(args, run_started_at)

    problems, meta = build_problems(args)
    if not problems:
        raise SystemExit("No benchmark problems selected")
    games = expand_games(args, problems)
    if not games:
        raise SystemExit("No benchmark games selected")

    print(
        f"Running {len(games)} game(s): bench={args.bench}, mode={args.mode}, "
        f"solver={solver_path}, problems={len(problems)}, tracks={args.tracks}, seeds={args.seeds}"
    )
    print(f"Run started at: {iso_dt(run_started_at)}")
    print(f"Saving outputs under: {out_dir}")

    solver_module = None
    if args.mode == "direct":
        solver_module = load_solver_module(solver_path)

    results: List[GameResult] = []
    start_all = time.perf_counter()
    for idx, (problem, track, seed) in enumerate(games, 1):
        if args.mode == "direct":
            assert solver_module is not None
            result = run_direct_game(
                solver_module,
                problem,
                track,
                seed,
                args.max_turns,
                args.budget,
                args.score_cap,
                trace_enabled=args.trace,
            )
        else:
            result = run_http_game(
                solver_path,
                problem,
                track,
                seed,
                args.max_turns,
                args.budget,
                args.score_cap,
                trace_enabled=args.trace,
                request_timeout=args.http_request_timeout,
            )
        results.append(result)
        print(game_print_line(idx, len(games), result, problem), flush=True)
        if args.trace and (args.print_all_traces or not result.success):
            print_trace(result)

    elapsed = time.perf_counter() - start_all
    run_finished_at = now_kst()
    run_info = {
        "run_id": out_dir.name,
        "run_started_at": iso_dt(run_started_at),
        "run_started_at_utc": iso_dt(run_started_at.astimezone(timezone.utc)),
        "run_started_date": run_started_at.strftime("%Y-%m-%d"),
        "run_started_time": run_started_at.strftime("%H:%M:%S"),
        "run_finished_at": iso_dt(run_finished_at),
        "run_finished_at_utc": iso_dt(run_finished_at.astimezone(timezone.utc)),
        "run_finished_date": run_finished_at.strftime("%Y-%m-%d"),
        "run_finished_time": run_finished_at.strftime("%H:%M:%S"),
        "timezone": "Asia/Seoul",
        "elapsed_wall_s": round(elapsed, 6),
        "results_dir": str(out_dir),
        "latest_results_dir": str(DEFAULT_RESULTS_DIR.resolve()),
        "results_dir_was_auto": args.results_dir is None or str(args.results_dir).strip() == "",
    }
    print(f"\nCompleted in {elapsed:.2f}s")
    print_summary(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = save_run_metadata(out_dir, run_info)
    config_path = save_config(out_dir, args, meta, solver_path, run_info)
    problems_path, manifest_path, single_path = save_problem_artifacts(out_dir, problems, meta, run_info)
    csv_path, jsonl_path, summary_path = write_outputs(out_dir, results, args, run_info)
    copy_latest_files(out_dir)

    print("\nWrote:")
    for path in (csv_path, jsonl_path, summary_path, problems_path, manifest_path, single_path, config_path, run_path):
        if path is not None:
            print(f"  {path}")

    return 0 if any(r.success for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
