#!/usr/bin/env python3
"""
Deep debug rerun benchmark for DCCP noisy Wordle slow/failed cases.

What it logs per rerun case:
  * every submitted query/action word by turn;
  * feedback pattern, oracle noise type, and effective secret for every guess;
  * posterior top-K words after processing the incoming feedback for each turn;
  * submit-gate checks from team00.py: min-turn, pmax/odds gate, T3 direct-confirm
    guard, exact top-K confirmation, veto/retry state, and force-submit;
  * guess-selection branch: prefix opener, direct top confirmation, alternate head
    probe, endgame discriminator, or entropy-pool probe.

The script intentionally runs team00.py in direct/introspection mode.  HTTP mode can
validate the protocol, but cannot expose posterior vectors or internal submit checks.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("This debug benchmark requires numpy.") from exc


# ---------------------------------------------------------------------------
# Path/import helpers
# ---------------------------------------------------------------------------


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p.resolve()
    return None


def resolve_existing(raw: Optional[str], label: str, fallbacks: Sequence[Path]) -> Path:
    if raw:
        p = Path(raw).expanduser()
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, p]
        got = _first_existing(candidates)
        if got:
            return got
        raise SystemExit(f"{label} not found: {raw!r}")
    got = _first_existing(fallbacks)
    if got:
        return got
    tried = ", ".join(str(x) for x in fallbacks)
    raise SystemExit(f"Could not locate {label}. Tried: {tried}")


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Input data loaders
# ---------------------------------------------------------------------------


def valid_word(w: Any) -> bool:
    return isinstance(w, str) and len(w) == 5 and w.isalpha() and w.islower()


def load_words_txt(path: Path) -> List[str]:
    words: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            w = line.strip().lower()
            if not w or w.startswith("#"):
                continue
            if not valid_word(w):
                raise ValueError(f"Invalid word at {path}:{lineno}: {w!r}")
            if w not in seen:
                seen.add(w)
                words.append(w)
    if not words:
        raise ValueError(f"No words loaded from {path}")
    return words


def infer_words_from_games_csv(path: Path) -> List[str]:
    words: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "secret" not in (reader.fieldnames or []):
            raise ValueError(f"games.csv has no 'secret' column: {path}")
        for row in reader:
            w = row["secret"]
            if w not in seen:
                if not valid_word(w):
                    raise ValueError(f"Invalid secret in games.csv: {w!r}")
                seen.add(w)
                words.append(w)
    if not words:
        raise ValueError(f"Could not infer words from {path}")
    return words


def _json_from_dir_or_zip(source_dir: Optional[Path], source_zip: Optional[Path], name: str) -> Dict[str, Any]:
    if source_dir is not None:
        path = source_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if source_zip is not None:
        with zipfile.ZipFile(source_zip, "r") as zf:
            with zf.open(name, "r") as f:
                return json.loads(f.read().decode("utf-8"))
    raise ValueError("source_dir or source_zip is required")


def _games_rows_from_dir_or_zip(source_dir: Optional[Path], source_zip: Optional[Path]) -> List[Dict[str, str]]:
    if source_dir is not None:
        path = source_dir / "games.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if source_zip is not None:
        with zipfile.ZipFile(source_zip, "r") as zf:
            with zf.open("games.csv", "r") as f:
                text = f.read().decode("utf-8").splitlines()
                return list(csv.DictReader(text))
    return []


def build_cases_from_result_files(source_dir: Optional[Path], source_zip: Optional[Path]) -> Dict[str, Any]:
    slow = _json_from_dir_or_zip(source_dir, source_zip, "slow_words.json")
    failed = _json_from_dir_or_zip(source_dir, source_zip, "failed_words.json")
    summary: Dict[str, Any]
    try:
        summary = _json_from_dir_or_zip(source_dir, source_zip, "summary.json")
    except Exception:
        summary = {}

    case_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    def add_case(secret: str, track: int, seed: int, reason: str, item: Dict[str, Any], word_item: Dict[str, Any]) -> None:
        key = (secret, int(track), int(seed))
        rec = case_map.setdefault(
            key,
            {"secret": secret, "track": int(track), "seed": int(seed), "reasons": [], "original": {}},
        )
        if reason not in rec["reasons"]:
            rec["reasons"].append(reason)
        rec["original"].update(
            {
                f"{reason}_status": item.get("status"),
                f"{reason}_score_turns": item.get("score_turns"),
                f"{reason}_turns": item.get("turns"),
                f"{reason}_act_wall_s": item.get("act_wall_s"),
            }
        )
        if reason == "slow":
            rec["original"]["slow_success"] = item.get("success")
            rec["original"]["max_score_turns"] = word_item.get("max_score_turns")

    for word_item in slow.get("words", []):
        secret = word_item.get("secret")
        for item in word_item.get("cases", []):
            add_case(secret, int(item.get("track", 0)), int(item.get("seed", 0)), "slow", item, word_item)
    for word_item in failed.get("words", []):
        secret = word_item.get("secret")
        for item in word_item.get("failures", []):
            add_case(secret, int(item.get("track", 0)), int(item.get("seed", 0)), "failed", item, word_item)

    for row in _games_rows_from_dir_or_zip(source_dir, source_zip):
        try:
            key = (row["secret"], int(row["track"]), int(row["seed"]))
        except Exception:
            continue
        if key not in case_map:
            continue
        case_map[key].setdefault("problem_id", row.get("problem_id") or f"full_{row['secret']}")
        case_map[key]["original"].update(
            {
                "full_status": row.get("status"),
                "full_success": int(row.get("success", 0)),
                "full_score_turns": int(row.get("score_turns", 0)),
                "full_turns": int(row.get("turns", 0)),
                "full_guesses": int(row.get("guesses", 0)),
                "full_submits": int(row.get("submits", 0)),
                "full_act_wall_s": row.get("act_wall_s"),
                "full_last_action": row.get("last_action"),
                "full_last_word": row.get("last_word"),
            }
        )

    def score_of(case: Dict[str, Any]) -> int:
        original = case.get("original", {})
        return int(original.get("full_score_turns") or original.get("slow_score_turns") or 0)

    def turns_of(case: Dict[str, Any]) -> int:
        original = case.get("original", {})
        return int(original.get("full_turns") or original.get("slow_turns") or 0)

    cases = sorted(
        case_map.values(),
        key=lambda c: (0 if "failed" in c.get("reasons", []) else 1, -score_of(c), -turns_of(c), -int(c["track"]), c["secret"]),
    )
    return {
        "metadata": {
            "description": "Built from slow_words.json and failed_words.json.",
            "summary": summary,
            "slow_threshold_turns": slow.get("threshold_turns"),
            "case_count": len(cases),
            "failed_case_count": sum(1 for c in cases if "failed" in c.get("reasons", [])),
            "slow_case_count": sum(1 for c in cases if "slow" in c.get("reasons", [])),
        },
        "cases": cases,
    }


def load_case_bundle(args: argparse.Namespace, script_dir: Path) -> Dict[str, Any]:
    if args.cases_file:
        with Path(args.cases_file).expanduser().open("r", encoding="utf-8") as f:
            return json.load(f)
    default_cases = script_dir / "debug_cases_slow_failed_full.json"
    if default_cases.exists():
        with default_cases.open("r", encoding="utf-8") as f:
            return json.load(f)
    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else None
    source_zip = Path(args.source_zip).expanduser().resolve() if args.source_zip else None
    if source_dir or source_zip:
        return build_cases_from_result_files(source_dir, source_zip)
    raise SystemExit("No cases available. Provide --cases-file, --source-dir, or --source-zip.")


def load_words(args: argparse.Namespace, script_dir: Path, source_dir: Optional[Path], source_zip: Optional[Path]) -> List[str]:
    if args.words:
        return load_words_txt(Path(args.words).expanduser().resolve())
    packaged = script_dir / "words_from_full_benchmark.txt"
    if packaged.exists():
        return load_words_txt(packaged)
    fallback = _first_existing([Path.cwd() / "words.txt", Path.cwd() / "benchmark" / "words.txt", script_dir / "words.txt"])
    if fallback:
        return load_words_txt(fallback)
    if source_dir and (source_dir / "games.csv").exists():
        return infer_words_from_games_csv(source_dir / "games.csv")
    if source_zip:
        with zipfile.ZipFile(source_zip, "r") as zf:
            with zf.open("games.csv", "r") as f:
                rows = f.read().decode("utf-8").splitlines()
        words: List[str] = []
        seen = set()
        for row in csv.DictReader(rows):
            w = row["secret"]
            if w not in seen:
                seen.add(w)
                words.append(w)
        return words
    raise SystemExit("Could not load candidate words. Provide --words or --source-dir/--source-zip with games.csv.")


# ---------------------------------------------------------------------------
# Numeric/JSON helpers
# ---------------------------------------------------------------------------


def finite_float(x: Any) -> Any:
    try:
        y = float(x)
    except Exception:
        return x
    if math.isfinite(y):
        return y
    if y == float("inf"):
        return "inf"
    if y == float("-inf"):
        return "-inf"
    return "nan"


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return finite_float(float(obj))
    if isinstance(obj, float):
        return finite_float(obj)
    return obj


def code_from_pattern_id(pid: int) -> str:
    rem = int(pid)
    out = []
    for mul in (81, 27, 9, 3, 1):
        digit = rem // mul
        out.append(str(int(digit)))
        rem -= int(digit) * mul
    return "".join(out)


def pattern_id_from_feedback(solver_mod: Any, text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(solver_mod.parse_feedback(text))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Posterior snapshots and internal check capture
# ---------------------------------------------------------------------------


def posterior_snapshot(solver: Any, solver_mod: Any, secret: str, top_k: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pi = np.asarray(solver.pi, dtype=np.float64)
    n = len(pi)
    k = min(max(1, int(top_k)), n)
    if n <= k:
        order = np.argsort(pi)[::-1]
    else:
        order = np.argpartition(pi, -k)[-k:]
        order = order[np.argsort(pi[order])[::-1]]
    rows: List[Dict[str, Any]] = []
    cum = 0.0
    for rank, idx0 in enumerate(order, 1):
        idx = int(idx0)
        prob = float(pi[idx])
        cum += prob
        rows.append(
            {
                "rank": rank,
                "word": solver.words[idx],
                "word_index": idx,
                "prob": prob,
                "cum_prob_topk": cum,
                "is_secret": solver.words[idx] == secret,
            }
        )
    secret_idx = solver.word_to_idx.get(secret)
    if secret_idx is None:
        secret_prob = 0.0
        secret_rank = None
    else:
        secret_prob = float(pi[int(secret_idx)])
        secret_rank = int(1 + np.count_nonzero(pi > secret_prob))
    positive = pi[pi > 0.0]
    entropy_bits = float(-np.sum(positive * np.log2(positive))) if len(positive) else 0.0
    top, pmax, p2, odds = solver_mod.posterior_stats(pi)
    summary = {
        "top_word": solver.words[int(top)],
        "top_index": int(top),
        "top_prob": float(pmax),
        "second_prob": float(p2),
        "odds": finite_float(odds),
        "secret_rank": secret_rank,
        "secret_prob": secret_prob,
        "topk_mass": float(sum(r["prob"] for r in rows)),
        "entropy_bits": entropy_bits,
        "positive_count": int(np.count_nonzero(pi > 0.0)),
    }
    return summary, rows


def collect_submit_checks(solver: Any, solver_mod: Any, turn: int) -> Dict[str, Any]:
    pi = np.asarray(solver.pi, dtype=np.float64)
    top, pmax, p2, odds = solver_mod.posterior_stats(pi)
    cfg = solver.cfg
    checks: Dict[str, Any] = {
        "track": int(solver.track),
        "turn": int(turn),
        "N": int(solver.N),
        "top_index": int(top),
        "top_word": solver.words[int(top)],
        "pmax": float(pmax),
        "second_prob": float(p2),
        "odds": finite_float(odds),
        "min_submit_turn": int(cfg.min_submit_turn),
        "force_submit_turn": int(cfg.force_submit_turn),
        "threshold": float(cfg.threshold),
        "odds_threshold": finite_float(cfg.odds_threshold),
        "veto_until_before": int(getattr(solver, "exact_confirm_veto_until", 0)),
    }
    if solver.N == 1:
        checks.update({"single_candidate": True, "would_submit_without_more_checks": True})
        return checks
    if solver.track == 1:
        active_count = int(np.count_nonzero(pi > 0.0))
        checks.update(
            {
                "single_candidate": False,
                "active_count": active_count,
                "min_turn_ok": bool(turn >= cfg.min_submit_turn),
                "track1_single_active_ok": bool(turn >= cfg.min_submit_turn and active_count == 1),
                "force_due": bool(turn >= cfg.force_submit_turn),
                "would_call_exact_confirmation": False,
            }
        )
        return checks

    gate_p = bool(float(pmax) >= float(cfg.threshold))
    gate_odds = bool(float(odds) >= float(cfg.odds_threshold)) if math.isfinite(float(odds)) else True
    gate = bool(gate_p and gate_odds)
    force = bool(turn >= cfg.force_submit_turn)
    min_ok = bool(turn >= cfg.min_submit_turn)
    checks.update(
        {
            "single_candidate": False,
            "min_turn_ok": min_ok,
            "below_min_submit_turn": not min_ok,
            "force_due": force,
            "gate_pmax_ok": gate_p,
            "gate_odds_ok": gate_odds,
            "posterior_gate_ok": gate,
            "veto_active": bool(turn < int(getattr(solver, "exact_confirm_veto_until", 0))),
        }
    )
    direct_top_queries = None
    if solver.track == 3:
        direct_top_queries = int(sum(1 for _, gi, _ in solver.history if int(gi) == int(top)))
        checks.update(
            {
                "direct_top_queries": direct_top_queries,
                "direct_top_queries_required": 2,
                "direct_top_queries_ok": bool(direct_top_queries >= 2),
            }
        )
    K = min(max(2, int(cfg.topk_confirm)), int(solver.N))
    if K > 0:
        topk = np.argpartition(pi, -K)[-K:]
        topk_mass = float(np.sum(pi[topk]))
    else:
        topk_mass = 0.0
    would_call_exact = bool(
        min_ok
        and (force or gate)
        and not force
        and not (solver.track == 3 and (direct_top_queries is not None and direct_top_queries < 2))
        and not bool(turn < int(getattr(solver, "exact_confirm_veto_until", 0)))
    )
    checks.update(
        {
            "topk_confirm_K": int(K),
            "topk_mass_min": float(cfg.topk_mass_min),
            "topk_mass_current": topk_mass,
            "would_call_exact_confirmation": would_call_exact,
        }
    )
    return checks


def explain_submit_result(checks: Dict[str, Any], submit_idx: Optional[int], exact_calls: List[Dict[str, Any]]) -> str:
    if submit_idx is not None:
        if checks.get("single_candidate"):
            return "submit_single_candidate"
        if checks.get("track") == 1:
            if checks.get("track1_single_active_ok"):
                return "submit_track1_single_active"
            if checks.get("force_due"):
                return "submit_track1_force_turn"
            return "submit_track1_other"
        if checks.get("force_due"):
            return "submit_force_turn"
        if exact_calls and exact_calls[-1].get("passed"):
            return "submit_posterior_gate_and_exact_topk_confirm_passed"
        return "submit_allowed"

    if checks.get("single_candidate"):
        return "no_submit_unexpected_single_candidate"
    if checks.get("track") == 1:
        if not checks.get("min_turn_ok"):
            return "no_submit_track1_before_min_turn"
        if not checks.get("track1_single_active_ok") and not checks.get("force_due"):
            return "no_submit_track1_not_single_active"
        return "no_submit_track1_other"
    if not checks.get("min_turn_ok"):
        return "no_submit_before_min_turn"
    if not checks.get("posterior_gate_ok") and not checks.get("force_due"):
        return "no_submit_posterior_gate_failed"
    if checks.get("track") == 3 and not checks.get("direct_top_queries_ok") and not checks.get("force_due"):
        return "no_submit_track3_needs_two_direct_top_queries"
    if checks.get("veto_active"):
        return "no_submit_exact_confirm_veto_active"
    if exact_calls and not exact_calls[-1].get("passed"):
        return "no_submit_exact_topk_confirm_failed_veto_set"
    return "no_submit_other"


def choose_guess_with_capture(solver: Any, solver_mod: Any, turn: int) -> Tuple[int, Dict[str, Any]]:
    top, pmax, p2, odds = solver_mod.posterior_stats(solver.pi)
    direct_top = recent_top = consecutive_top = None
    try:
        direct_top, recent_top, consecutive_top = solver._top_history_counts(int(top))
    except Exception:
        pass

    captured: Dict[str, Any] = {
        "top_word_before_choose": solver.words[int(top)],
        "top_index_before_choose": int(top),
        "top_prob_before_choose": float(pmax),
        "second_prob_before_choose": float(p2),
        "odds_before_choose": finite_float(odds),
        "direct_top_count": direct_top,
        "recent_top_count_last4": recent_top,
        "consecutive_top_count": consecutive_top,
        "alternate_head_probe_calls": [],
        "endgame_discriminator_calls": [],
        "choose_pool_calls": [],
    }

    orig_alt = getattr(solver, "_alternate_head_probe", None)
    orig_end = getattr(solver, "_endgame_discriminator", None)
    orig_pool = getattr(solver, "choose_pool", None)

    if orig_alt is not None:
        def wrapped_alt(top_arg: int, k: int = 12):
            res = orig_alt(top_arg, k=k)
            captured["alternate_head_probe_calls"].append(
                {
                    "top_word": solver.words[int(top_arg)],
                    "k": int(k),
                    "result_index": None if res is None else int(res),
                    "result_word": None if res is None else solver.words[int(res)],
                }
            )
            return res
        solver._alternate_head_probe = wrapped_alt  # type: ignore[assignment]

    if orig_end is not None:
        def wrapped_end(top_arg: int, k: int = 96):
            res = orig_end(top_arg, k=k)
            captured["endgame_discriminator_calls"].append(
                {
                    "top_word": solver.words[int(top_arg)],
                    "k": int(k),
                    "result_index": int(res),
                    "result_word": solver.words[int(res)],
                }
            )
            return res
        solver._endgame_discriminator = wrapped_end  # type: ignore[assignment]

    if orig_pool is not None:
        def wrapped_pool():
            res = orig_pool()
            captured["choose_pool_calls"].append({"pool_size": int(len(res))})
            return res
        solver.choose_pool = wrapped_pool  # type: ignore[assignment]

    try:
        guess_idx = int(solver.choose_guess(turn))
    finally:
        if orig_alt is not None:
            solver._alternate_head_probe = orig_alt  # type: ignore[assignment]
        if orig_end is not None:
            solver._endgame_discriminator = orig_end  # type: ignore[assignment]
        if orig_pool is not None:
            solver.choose_pool = orig_pool  # type: ignore[assignment]

    reason = "choose_guess_other"
    prefix_indices = list(getattr(solver, "prefix_indices", []))
    if turn <= 3 and turn <= len(prefix_indices) and float(pmax) < 0.98 and guess_idx == int(prefix_indices[turn - 1]):
        reason = "prefix_opening_query"
    elif captured["alternate_head_probe_calls"] and any(call.get("result_index") == guess_idx for call in captured["alternate_head_probe_calls"]):
        reason = "alternate_head_probe"
    elif captured["endgame_discriminator_calls"] and any(call.get("result_index") == guess_idx for call in captured["endgame_discriminator_calls"]):
        reason = "endgame_discriminator"
    elif int(getattr(solver, "track", 0)) == 3 and guess_idx == int(top):
        reason = "track3_direct_top_confirmation"
    elif int(getattr(solver, "track", 0)) == 2 and float(pmax) >= 0.82 and guess_idx == int(top):
        reason = "track2_direct_top_probe"
    elif captured["choose_pool_calls"]:
        reason = "entropy_pool_probe"

    captured.update({"chosen_index": int(guess_idx), "chosen_word": solver.words[int(guess_idx)], "reason": reason})
    return guess_idx, captured


def capture_maybe_submit(solver: Any, solver_mod: Any, turn: int) -> Tuple[Optional[int], Dict[str, Any]]:
    checks = collect_submit_checks(solver, solver_mod, turn)
    exact_calls: List[Dict[str, Any]] = []
    orig_exact = getattr(solver, "_full_exact_topk_confirmation", None)
    if orig_exact is not None:
        def wrapped_exact(topk: np.ndarray, approx_top: int, *args, **kwargs):
            info = dict(orig_exact(topk, approx_top, *args, **kwargs) or {})
            if "winner" in info and info["winner"] is not None:
                info["winner_word"] = solver.words[int(info["winner"])]
            info["approx_top"] = int(approx_top)
            info["approx_top_word"] = solver.words[int(approx_top)]
            exact_calls.append(json_safe(info))
            return info
        solver._full_exact_topk_confirmation = wrapped_exact  # type: ignore[assignment]
    try:
        submit_idx = solver.maybe_submit(turn)
        submit_idx = None if submit_idx is None else int(submit_idx)
    finally:
        if orig_exact is not None:
            solver._full_exact_topk_confirmation = orig_exact  # type: ignore[assignment]
    checks["exact_confirmation_calls"] = exact_calls
    checks["veto_until_after"] = int(getattr(solver, "exact_confirm_veto_until", 0))
    checks["decision_reason"] = explain_submit_result(checks, submit_idx, exact_calls)
    checks["submit_index"] = submit_idx
    checks["submit_word"] = None if submit_idx is None else solver.words[int(submit_idx)]
    return submit_idx, checks


# ---------------------------------------------------------------------------
# Solver check description
# ---------------------------------------------------------------------------


def make_solver_checks_doc(words: Sequence[str], bench_mod: Any, solver_mod: Any, max_turns: int, budget: float) -> Dict[str, Any]:
    tracks_doc: Dict[str, Any] = {}
    for track in (1, 2, 3):
        p_one, p_two = bench_mod.TRACKS[track]
        payload = bench_mod.make_payload("checks_doc", list(words), p_one, p_two, max_turns, budget)
        solver = solver_mod.NoisyWordleSolver(payload)
        cfg = solver.cfg
        tracks_doc[str(track)] = {
            "noise_probability": p_one,
            "two_letter_noise_probability": p_two,
            "prefix": list(cfg.prefix),
            "pool_size": int(cfg.pool_size),
            "static_pool_size": int(cfg.static_pool_size),
            "top_post_pool_size": int(cfg.top_post_pool_size),
            "min_submit_turn": int(cfg.min_submit_turn),
            "posterior_threshold": float(cfg.threshold),
            "odds_threshold": finite_float(cfg.odds_threshold),
            "topk_confirm": int(cfg.topk_confirm),
            "topk_mass_min": float(cfg.topk_mass_min),
            "retry_gap": int(cfg.retry_gap),
            "force_submit_turn": int(cfg.force_submit_turn),
            "alpha": float(cfg.alpha),
            "mix_epsilon": float(cfg.mix_epsilon),
            "prefix_schedule_count": int(len(solver.prefix_schedules)),
        }
    return {
        "input_and_protocol_checks": [
            "candidate_words are deduplicated, then every word must be five lowercase alphabetic characters in encode_words().",
            "feedback text is parsed with FEEDBACK_RE; all five ordinal positions must be present or parse_feedback() raises ValueError.",
            "benchmark.validate_action() requires a dict, action in {'guess','submit'}, and word inside the candidate set.",
        ],
        "posterior_update_checks": [
            "First three turns use an exact joint forced-noise schedule; all-clean first-three schedules are redistributed into forced one-/two-letter noise.",
            "Each observation computes clean, one-letter-mutation, and two-letter-mutation likelihood vectors L0/L1/L2.",
            "Track 1 multiplies by clean likelihood only; Tracks 2/3 use the configured type mixture after turn 3.",
            "normalize() falls back to a uniform distribution if posterior mass is non-finite or zero.",
        ],
        "submit_gate_checks": [
            "Track 1 submits only after min_submit_turn when one candidate remains active, or at force_submit_turn.",
            "Tracks 2/3 require min_submit_turn, pmax >= threshold, and top/second odds >= odds_threshold unless force_submit_turn is reached.",
            "Track 3 additionally requires the current top word to have been directly queried at least twice before non-forced submit.",
            "Tracks 2/3 replay an untempered exact posterior over top-K and require same winner, exact_pmax threshold, exact_odds threshold, and enough top-K mass.",
            "When exact top-K confirmation fails, exact_confirm_veto_until is advanced by retry_gap.",
        ],
        "guess_selection_checks": [
            "Turns 1-3 prefer fixed prefix openers while pmax < 0.98.",
            "Track 2 directly probes the posterior top when pmax >= 0.82.",
            "Track 3 endgame alternates between direct top confirmation, alternate runner-up probes, and head-cluster discriminators.",
            "Otherwise choose_guess() maximizes feedback entropy over choose_pool(), with small posterior and repeat-use adjustments.",
        ],
        "track_configs": tracks_doc,
    }


# ---------------------------------------------------------------------------
# Detailed game runner
# ---------------------------------------------------------------------------


def case_id_for(case: Dict[str, Any], ordinal: int) -> str:
    return f"{ordinal:04d}_{case['secret']}_T{int(case['track'])}_seed{int(case.get('seed', 0))}"


def run_detailed_game(
    bench_mod: Any,
    solver_mod: Any,
    words: Sequence[str],
    case: Dict[str, Any],
    ordinal: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    secret = str(case["secret"])
    track = int(case["track"])
    seed = int(case.get("seed", 0))
    if secret not in set(words):
        raise ValueError(f"Secret {secret!r} is not in candidate words")
    p_one, p_two = bench_mod.TRACKS[track]
    problem_id = str(case.get("problem_id") or f"full_{secret}")
    cid = case_id_for(case, ordinal)

    payload = bench_mod.make_payload(problem_id, list(words), p_one, p_two, int(args.max_turns), float(args.budget))
    solver = solver_mod.NoisyWordleSolver(payload)
    oracle = bench_mod.NoiseOracle(secret, int(args.max_turns), p_one, p_two, 3, seed, problem_id, track)
    candidates = set(words)

    feedback: Optional[str] = None
    feedback_code_in: Optional[str] = None
    guesses = 0
    submits = 0
    success = False
    status = "max_turns"
    turns_used = 0
    last_word = ""
    last_action = ""
    solver_decision_wall_s = 0.0
    diagnostic_wall_s = 0.0
    turn_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for turn in range(1, int(args.max_turns) + 1):
        turns_used = turn
        turn_start = time.perf_counter()
        error = ""
        observed_info: Dict[str, Any] = {}
        try:
            # Process feedback for the previous guess exactly as team00.act() does.
            observe_start = time.perf_counter()
            before_hist_len = len(solver.history)
            solver.observe(turn, feedback)
            observe_wall = time.perf_counter() - observe_start
            solver_decision_wall_s += observe_wall

            if len(solver.history) > before_hist_len:
                hist_turn, hist_guess_idx, obs_pid = solver.history[-1]
                observed_info = {
                    "observed_turn": int(hist_turn),
                    "observed_guess_word": solver.words[int(hist_guess_idx)],
                    "observed_pattern_id": int(obs_pid),
                    "observed_feedback_code": code_from_pattern_id(int(obs_pid)),
                }
                if solver.likelihood_history:
                    _, L0, L1, L2 = solver.likelihood_history[-1]
                    true_idx = int(solver.word_to_idx[secret])
                    observed_info.update(
                        {
                            "true_L0": float(L0[true_idx]),
                            "true_L1": float(L1[true_idx]),
                            "true_L2": float(L2[true_idx]),
                        }
                    )
                    if int(hist_turn) > 3:
                        p0, p1, p2 = solver.type_probs_after_prefix(int(hist_turn))
                        observed_info["true_mixture_likelihood"] = float(p0 * L0[true_idx] + p1 * L1[true_idx] + p2 * L2[true_idx])

            diag_start = time.perf_counter()
            post_summary, post_rows = posterior_snapshot(solver, solver_mod, secret, int(args.top_k))
            diagnostic_wall_s += time.perf_counter() - diag_start
            for r in post_rows:
                rr = {
                    "case_id": cid,
                    "turn": turn,
                    "secret": secret,
                    "track": track,
                    "seed": seed,
                    **r,
                }
                topk_rows.append(rr)

            maybe_start = time.perf_counter()
            submit_idx, submit_checks = capture_maybe_submit(solver, solver_mod, turn)
            solver_decision_wall_s += time.perf_counter() - maybe_start

            if submit_idx is not None:
                action = "submit"
                word = solver.words[int(submit_idx)]
                solver.last_action = {"action": action, "word": word}
                choice_info: Dict[str, Any] = {"reason": submit_checks.get("decision_reason")}
            else:
                choose_start = time.perf_counter()
                guess_idx, choice_info = choose_guess_with_capture(solver, solver_mod, turn)
                solver_decision_wall_s += time.perf_counter() - choose_start
                action = "guess"
                word = solver.words[int(guess_idx)]
                solver.guessed.add(int(guess_idx))
                solver.pending_guess_idx = int(guess_idx)
                solver.pending_guess_turn = int(turn)
                solver.last_action = {"action": action, "word": word}

            bench_mod.validate_action({"action": action, "word": word}, candidates)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "act_crash_or_invalid"
            turn_rows.append(
                json_safe(
                    {
                        "case_id": cid,
                        "turn": turn,
                        "secret": secret,
                        "track": track,
                        "seed": seed,
                        "status": status,
                        "error": error,
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
            )
            event_rows.append({"case_id": cid, "turn": turn, "event": "act_error", "detail": error})
            break

        last_action = action
        last_word = word
        row_base: Dict[str, Any] = {
            "case_id": cid,
            "turn": turn,
            "secret": secret,
            "track": track,
            "seed": seed,
            "problem_id": problem_id,
            "feedback_in_code": feedback_code_in,
            "action": action,
            "query_word": word,
            "query_is_secret": word == secret,
            "posterior_top_word": post_summary.get("top_word"),
            "posterior_top_prob": post_summary.get("top_prob"),
            "posterior_second_prob": post_summary.get("second_prob"),
            "posterior_odds": post_summary.get("odds"),
            "secret_rank": post_summary.get("secret_rank"),
            "secret_prob": post_summary.get("secret_prob"),
            "posterior_entropy_bits": post_summary.get("entropy_bits"),
            "submit_decision_reason": submit_checks.get("decision_reason"),
            "guess_choice_reason": choice_info.get("reason"),
            "observe": observed_info,
            "submit_checks": submit_checks,
            "choice_info": choice_info,
            "turn_solver_wall_s": time.perf_counter() - turn_start,
            "error": error,
        }

        if action == "submit":
            submits += 1
            success = bool(word == secret and turn < int(args.score_cap))
            status = "solved" if success else "wrong_submit"
            row_base.update({"submit_success": success, "status_after_turn": status})
            turn_rows.append(json_safe(row_base))
            event_rows.append(
                json_safe(
                    {
                        "case_id": cid,
                        "turn": turn,
                        "event": "submit",
                        "word": word,
                        "success": success,
                        "decision_reason": submit_checks.get("decision_reason"),
                        "posterior_top_word": post_summary.get("top_word"),
                        "secret_rank": post_summary.get("secret_rank"),
                        "secret_prob": post_summary.get("secret_prob"),
                    }
                )
            )
            break

        guesses += 1
        feedback, out_code, noise_type, eff_secret = oracle.feedback_for_guess(word, turn)
        feedback_code_in = out_code
        row_base.update(
            {
                "oracle_feedback_code": out_code,
                "oracle_noise_type": int(noise_type),
                "oracle_effective_secret": eff_secret,
                "status_after_turn": "running",
            }
        )
        turn_rows.append(json_safe(row_base))
        event_rows.append(
            json_safe(
                {
                    "case_id": cid,
                    "turn": turn,
                    "event": "query",
                    "action": action,
                    "word": word,
                    "choice_reason": choice_info.get("reason"),
                    "feedback_code": out_code,
                    "noise_type": int(noise_type),
                    "effective_secret": eff_secret,
                    "posterior_top_word": post_summary.get("top_word"),
                    "secret_rank": post_summary.get("secret_rank"),
                    "secret_prob": post_summary.get("secret_prob"),
                }
            )
        )
        if bool(args.enforce_budget) and solver_decision_wall_s > float(args.budget):
            status = "act_time_budget_exceeded"
            event_rows.append({"case_id": cid, "turn": turn, "event": "budget_exceeded", "budget": float(args.budget)})
            break

    else:
        status = "max_turns"

    score_turns = turns_used if success else int(args.score_cap)
    game_row = json_safe(
        {
            "case_id": cid,
            "secret": secret,
            "track": track,
            "seed": seed,
            "problem_id": problem_id,
            "reasons": case.get("reasons", []),
            "original": case.get("original", {}),
            "success": int(success),
            "status": status,
            "turns": int(turns_used),
            "score_turns": int(score_turns),
            "guesses": int(guesses),
            "submits": int(submits),
            "last_action": last_action,
            "last_word": last_word,
            "solver_decision_wall_s": solver_decision_wall_s,
            "diagnostic_wall_s": diagnostic_wall_s,
        }
    )
    return game_row, turn_rows, topk_rows, event_rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(json_safe(v), ensure_ascii=False, sort_keys=True)
        else:
            out[k] = v
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def summarize_games(game_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(game_rows)
    solved = sum(1 for r in game_rows if int(r.get("success", 0)) == 1)
    by_status: Dict[str, int] = {}
    by_track: Dict[str, Dict[str, Any]] = {}
    for row in game_rows:
        by_status[row.get("status", "")] = by_status.get(row.get("status", ""), 0) + 1
        tr = str(row.get("track"))
        slot = by_track.setdefault(tr, {"games": 0, "solved": 0, "statuses": {}})
        slot["games"] += 1
        slot["solved"] += int(row.get("success", 0))
        slot["statuses"][row.get("status", "")] = slot["statuses"].get(row.get("status", ""), 0) + 1
    for slot in by_track.values():
        slot["success_rate"] = slot["solved"] / slot["games"] if slot["games"] else 0.0
    return {
        "games": total,
        "solved": solved,
        "failed": total - solved,
        "success_rate": solved / total if total else 0.0,
        "statuses": by_status,
        "tracks": by_track,
        "avg_score_turns": (sum(int(r.get("score_turns", 0)) for r in game_rows) / total) if total else 0.0,
        "avg_solver_decision_wall_s": (sum(float(r.get("solver_decision_wall_s", 0.0)) for r in game_rows) / total) if total else 0.0,
    }


def write_case_report(out_dir: Path, game: Dict[str, Any], turns: List[Dict[str, Any]], topk: List[Dict[str, Any]]) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cid = game["case_id"]
    lines: List[str] = []
    lines.append(f"# {cid}\n")
    lines.append(f"secret={game['secret']} track={game['track']} seed={game['seed']} status={game['status']} score_turns={game['score_turns']}\n")
    lines.append("\n| turn | query | reason | feedback/noise | top posterior | secret rank/prob | submit check |\n")
    lines.append("|---:|---|---|---|---|---|---|\n")
    top_by_turn: Dict[int, List[Dict[str, Any]]] = {}
    for r in topk:
        top_by_turn.setdefault(int(r["turn"]), []).append(r)
    for row in turns:
        turn = int(row["turn"])
        top_words = ", ".join(f"{r['word']}:{float(r['prob']):.4f}" for r in sorted(top_by_turn.get(turn, []), key=lambda x: int(x["rank"]))[:5])
        query = f"{row.get('action')} {row.get('query_word')}"
        reason = row.get("guess_choice_reason") or row.get("submit_decision_reason")
        fb = row.get("oracle_feedback_code") or row.get("feedback_in_code") or ""
        if row.get("oracle_noise_type") is not None:
            fb = f"{fb} / n={row.get('oracle_noise_type')} eff={row.get('oracle_effective_secret')}"
        secret_rank = f"{row.get('secret_rank')} / {float(row.get('secret_prob') or 0.0):.5f}"
        submit_reason = row.get("submit_decision_reason") or ""
        lines.append(f"| {turn} | {query} | {reason} | {fb} | {top_words} | {secret_rank} | {submit_reason} |\n")
    (reports / f"{cid}.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rerun slow/failed noisy Wordle cases with posterior/action diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--solver", default=None, help="Path to team00.py")
    p.add_argument("--benchmark-py", default=None, help="Path to benchmark.py")
    p.add_argument("--words", default=None, help="Candidate words.txt. If omitted, packaged words or games.csv are used.")
    p.add_argument("--cases-file", default=None, help="JSON case bundle. If omitted, packaged debug_cases_slow_failed_full.json is used.")
    p.add_argument("--source-dir", default=None, help="Directory containing games.csv, slow_words.json, failed_words.json.")
    p.add_argument("--source-zip", default=None, help="ZIP containing games.csv, slow_words.json, failed_words.json.")
    p.add_argument("--out-dir", default=None, help="Output directory for detailed debug logs.")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases after filtering/sorting.")
    p.add_argument("--only-failed", action="store_true", help="Run only cases whose reasons include 'failed'.")
    p.add_argument("--only-track", type=int, choices=(1, 2, 3), default=None, help="Run only one track.")
    p.add_argument("--top-k", type=int, default=10, help="Posterior top-K rows to save for each turn.")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--score-cap", type=int, default=100)
    p.add_argument("--enforce-budget", action="store_true", help="Stop a game when accumulated solver decision time exceeds --budget.")
    p.add_argument("--write-reports", action="store_true", help="Write one Markdown report per case.")
    p.add_argument("--stop-on-error", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    benchmark_py = resolve_existing(
        args.benchmark_py,
        "benchmark.py",
        [script_dir / "benchmark.py", Path.cwd() / "benchmark.py", Path.cwd() / "benchmark" / "benchmark.py", script_dir.parent / "benchmark.py"],
    )
    solver_py = resolve_existing(
        args.solver,
        "team00.py",
        [script_dir / "team00.py", Path.cwd() / "team00.py", Path.cwd().parent / "team00.py", script_dir.parent / "team00.py"],
    )
    bench_mod = import_module_from_path(benchmark_py, "debug_benchmark_module")
    solver_mod = import_module_from_path(solver_py, "debug_solver_module")
    if not hasattr(solver_mod, "NoisyWordleSolver"):
        raise SystemExit("This deep-debug runner requires team00.py to expose NoisyWordleSolver.")

    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else None
    source_zip = Path(args.source_zip).expanduser().resolve() if args.source_zip else None
    words = load_words(args, script_dir, source_dir, source_zip)
    bundle = load_case_bundle(args, script_dir)
    cases = list(bundle.get("cases", []))
    if args.only_failed:
        cases = [c for c in cases if "failed" in c.get("reasons", [])]
    if args.only_track is not None:
        cases = [c for c in cases if int(c.get("track")) == int(args.only_track)]
    if args.max_cases is not None:
        cases = cases[: max(0, int(args.max_cases))]
    if not cases:
        raise SystemExit("No cases selected.")

    started = datetime.now().astimezone()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = script_dir / "debug_results" / started.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== deep debug benchmark ===")
    print(f"benchmark.py : {benchmark_py}")
    print(f"solver       : {solver_py}")
    print(f"words        : {len(words)}")
    print(f"cases        : {len(cases)}")
    print(f"out_dir      : {out_dir}")

    game_rows: List[Dict[str, Any]] = []
    turn_rows_all: List[Dict[str, Any]] = []
    topk_rows_all: List[Dict[str, Any]] = []
    event_rows_all: List[Dict[str, Any]] = []

    checks_doc = make_solver_checks_doc(words, bench_mod, solver_mod, int(args.max_turns), float(args.budget))
    (out_dir / "solver_checks.json").write_text(json.dumps(json_safe(checks_doc), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "cases_resolved.json").write_text(
        json.dumps(json_safe({"metadata": bundle.get("metadata", {}), "cases": cases}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for i, case in enumerate(cases, 1):
        try:
            game, turns, topk, events = run_detailed_game(bench_mod, solver_mod, words, case, i, args)
        except Exception as exc:
            if args.stop_on_error:
                raise
            cid = case_id_for(case, i)
            game = {
                "case_id": cid,
                "secret": case.get("secret"),
                "track": case.get("track"),
                "seed": case.get("seed", 0),
                "success": 0,
                "status": "debug_runner_error",
                "turns": 0,
                "score_turns": int(args.score_cap),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            turns, topk, events = [], [], [{"case_id": cid, "turn": 0, "event": "debug_runner_error", "detail": game["error"]}]
        game_rows.append(game)
        turn_rows_all.extend(turns)
        topk_rows_all.extend(topk)
        event_rows_all.extend(events)
        if args.write_reports:
            write_case_report(out_dir, game, turns, topk)
        mark = "OK" if int(game.get("success", 0)) == 1 else "FAIL"
        print(
            f"[{i:>4}/{len(cases)}] {mark:4s} T{game.get('track')} seed={game.get('seed')} "
            f"secret={game.get('secret')} status={game.get('status')} score={game.get('score_turns')} "
            f"turns={game.get('turns')}",
            flush=True,
        )

    finished = datetime.now().astimezone()
    summary = summarize_games(game_rows)
    summary["run"] = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "elapsed_s": round((finished - started).total_seconds(), 3),
        "benchmark_py": str(benchmark_py),
        "solver": str(solver_py),
        "words_count": len(words),
        "cases_count": len(cases),
        "top_k": int(args.top_k),
        "max_turns": int(args.max_turns),
        "score_cap": int(args.score_cap),
    }
    summary["outputs"] = {
        "games_csv": str(out_dir / "detailed_games.csv"),
        "games_jsonl": str(out_dir / "detailed_games.jsonl"),
        "turns_csv": str(out_dir / "turns.csv"),
        "turns_jsonl": str(out_dir / "turns.jsonl"),
        "posterior_topk_csv": str(out_dir / "posterior_topk.csv"),
        "events_jsonl": str(out_dir / "events.jsonl"),
        "solver_checks": str(out_dir / "solver_checks.json"),
        "cases_resolved": str(out_dir / "cases_resolved.json"),
    }

    write_csv(out_dir / "detailed_games.csv", game_rows)
    write_jsonl(out_dir / "detailed_games.jsonl", game_rows)
    write_csv(out_dir / "turns.csv", turn_rows_all)
    write_jsonl(out_dir / "turns.jsonl", turn_rows_all)
    write_csv(out_dir / "posterior_topk.csv", topk_rows_all)
    write_jsonl(out_dir / "events.jsonl", event_rows_all)
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nSaved outputs:")
    for key, value in summary["outputs"].items():
        print(f"  {key:20s}: {value}")
    print(f"  summary             : {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
