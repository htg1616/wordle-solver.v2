#!/usr/bin/env python3
"""
Noisy Wordle solver HTTP server for the DCCP Spring 2026 term project.

This is a self-contained production implementation derived from the junior
algorithm handoff, but modified for the actual grading protocol:
  * no external/precomputed matrix files;
  * candidate words are accepted only through /start_problem;
  * feedback rows are built lazily and cached;
  * the first-three-turn forced-noise rule is handled as an exact joint schedule;
  * Tracks 2/3 use exact collapsed one-/two-letter mutation likelihoods rather
    than a precomputed empirical confusion matrix.

Run with:
    python team00.py

If your team number is not 00, rename this file to teamXX.py before submission.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - grader environment issue
    print("NumPy is required by this solver.", exc, file=sys.stderr, flush=True)
    raise

N_PATTERNS = 243
POW3 = np.array([81, 27, 9, 3, 1], dtype=np.uint16)
ORDINALS = ("first", "second", "third", "fourth", "fifth")
DIGIT_TO_STATUS = {"0": "absent", "1": "misplaced", "2": "correct"}
STATUS_TO_DIGIT = {v: k for k, v in DIGIT_TO_STATUS.items()}
FEEDBACK_RE = re.compile(
    r"The\s+(first|second|third|fourth|fifth)\s+letter\s+'([a-z])'\s+is\s+(correct|misplaced|absent)\s*\.",
    re.IGNORECASE,
)
PAIRS: Tuple[Tuple[int, int], ...] = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
PAIR_ARR = np.array(PAIRS, dtype=np.int8)


def encode_words(words: Sequence[str]) -> np.ndarray:
    arr = np.empty((len(words), 5), dtype=np.uint8)
    for i, w in enumerate(words):
        if len(w) != 5 or (not w.isalpha()) or (not w.islower()):
            raise ValueError(f"invalid candidate at index {i}: {w!r}")
        arr[i] = [ord(c) - 97 for c in w]
    return arr


def compute_letter_counts(secrets: np.ndarray) -> np.ndarray:
    counts = np.zeros((secrets.shape[0], 26), dtype=np.uint8)
    rows = np.arange(secrets.shape[0])
    for pos in range(5):
        np.add.at(counts, (rows, secrets[:, pos]), 1)
    return counts


def feedback_ids_for_guess_with_counts(guess_vec: np.ndarray, secrets: np.ndarray, secret_counts: np.ndarray) -> np.ndarray:
    """Vectorized duplicate-aware Wordle pattern IDs for one guess vs many secrets."""
    guess_vec = np.asarray(guess_vec, dtype=np.uint8).reshape(5)
    green = secrets == guess_vec[None, :]
    marks = np.zeros((secrets.shape[0], 5), dtype=np.uint8)
    marks[green] = 2
    rem: Dict[int, np.ndarray] = {}
    # Only letters that appear in the guess can ever become yellow.
    for ch in np.unique(guess_vec):
        ch_int = int(ch)
        total = secret_counts[:, ch_int].astype(np.int16)
        green_consumed = np.count_nonzero(green[:, guess_vec == ch], axis=1).astype(np.int16)
        rem[ch_int] = total - green_consumed
    for pos in range(5):
        ch_int = int(guess_vec[pos])
        m = (~green[:, pos]) & (rem[ch_int] > 0)
        marks[m, pos] = 1
        rem[ch_int][m] -= 1
    return (marks.astype(np.uint16) @ POW3).astype(np.uint8, copy=False)


def feedback_ids_for_guess(guess_vec: np.ndarray, secrets: np.ndarray) -> np.ndarray:
    """Same as feedback_ids_for_guess_with_counts, but computes counts internally."""
    guess_vec = np.asarray(guess_vec, dtype=np.uint8).reshape(5)
    secrets = np.asarray(secrets, dtype=np.uint8)
    green = secrets == guess_vec[None, :]
    marks = np.zeros((secrets.shape[0], 5), dtype=np.uint8)
    marks[green] = 2
    rem: Dict[int, np.ndarray] = {}
    for ch in np.unique(guess_vec):
        ch_int = int(ch)
        total = np.count_nonzero(secrets == ch_int, axis=1).astype(np.int16)
        green_consumed = np.count_nonzero(green[:, guess_vec == ch], axis=1).astype(np.int16)
        rem[ch_int] = total - green_consumed
    for pos in range(5):
        ch_int = int(guess_vec[pos])
        m = (~green[:, pos]) & (rem[ch_int] > 0)
        marks[m, pos] = 1
        rem[ch_int][m] -= 1
    return (marks.astype(np.uint16) @ POW3).astype(np.uint8, copy=False)


def pattern_id_from_code(code: str) -> int:
    return int(sum((ord(code[i]) - 48) * int(POW3[i]) for i in range(5)))


def parse_feedback(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    out: List[Optional[str]] = [None] * 5
    order = {name: i for i, name in enumerate(ORDINALS)}
    for m in FEEDBACK_RE.finditer(text):
        out[order[m.group(1).lower()]] = STATUS_TO_DIGIT[m.group(3).lower()]
    if any(x is None for x in out):
        raise ValueError(f"could not parse feedback: {text!r}")
    code = "".join(x for x in out if x is not None)
    return pattern_id_from_code(code)


def normalize(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    if not math.isfinite(s) or s <= 0.0:
        return np.ones_like(v, dtype=np.float64) / max(1, len(v))
    return v / s


def entropy_from_mass(mass: np.ndarray) -> float:
    s = float(np.sum(mass))
    if s <= 0.0:
        return 0.0
    p = mass[mass > 0.0] / s
    return float(-np.sum(p * np.log2(p)))


def posterior_stats(pi: np.ndarray) -> Tuple[int, float, float, float]:
    if len(pi) == 1:
        return 0, 1.0, 0.0, float("inf")
    # argpartition avoids sorting the whole array every turn.
    two = np.argpartition(pi, -2)[-2:]
    if pi[int(two[0])] >= pi[int(two[1])]:
        top, second = int(two[0]), int(two[1])
    else:
        top, second = int(two[1]), int(two[0])
    p1 = float(pi[top])
    p2 = float(pi[second])
    return top, p1, p2, p1 / max(p2, 1e-300)


def dedupe_words(words: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for w in words:
        w = str(w)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


@dataclass(frozen=True)
class SolverConfig:
    track: int
    prefix: Tuple[str, ...]
    pool_size: int
    static_pool_size: int
    top_post_pool_size: int
    min_submit_turn: int
    threshold: float
    odds_threshold: float
    topk_confirm: int
    topk_mass_min: float
    retry_gap: int
    force_submit_turn: int
    alpha: float
    mix_epsilon: float


class NoisyWordleSolver:
    def __init__(self, data: dict):
        words_raw = data.get("candidate_words") or []
        self.words = dedupe_words(words_raw)
        if not self.words:
            raise ValueError("empty candidate_words")
        self.N = len(self.words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.enc = encode_words(self.words)
        self.counts = compute_letter_counts(self.enc)

        self.p_one = float(data.get("noise_probability", 0.0) or 0.0)
        self.p_two = float(data.get("two_letter_noise_probability", 0.0) or 0.0)
        self.force_noise_turns = int(data.get("force_noise_turns", 3) or 3)
        self.max_turns = int(data.get("max_turns", 100) or 100)
        self.track = self._infer_track(self.p_one, self.p_two)
        self.avg_near2, self.avg_near3 = self._estimate_cluster_profile()
        self.hard_candidate_mode = self._is_hard_candidate_list()
        self.cfg = self._make_config()

        self.row_cache: Dict[int, np.ndarray] = {}
        self.pending_guess_idx: Optional[int] = None
        self.pending_guess_turn: Optional[int] = None
        self.last_processed_turn = 0
        self.guessed: set[int] = set()
        self.history: List[Tuple[int, int, int]] = []
        # Each entry stores (turn, L0, L1, L2), where Lt is P(obs | secret, noise type t).
        self.likelihood_history: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self.pi = np.ones(self.N, dtype=np.float64) / self.N
        self.exact_confirm_veto_until = 0
        self.prefix_schedules = self._make_prefix_schedules()
        self.static_pool = self._make_static_pool()
        self.prefix_indices = self._make_prefix_indices()
        self.last_action: Optional[dict] = None
        self.next_forced_probe_idx: Optional[int] = None

    @staticmethod
    def _infer_track(p_one: float, p_two: float) -> int:
        if p_one <= 1e-12 and p_two <= 1e-12:
            return 1
        if p_two > 0.5:
            return 3
        return 2

    def _estimate_cluster_profile(self) -> Tuple[float, float]:
        """Estimate how clustered the candidate list is.

        Full 12k lists usually contain many broad information probes.  A custom
        list made only of hard words often has many near-neighbours, so pure
        posterior-top guessing is dangerous.  We cheaply estimate the average
        number of words within Hamming distance <=2 and <=3 on a deterministic
        sample and use it to enable hard-candidate mode.
        """
        if self.N <= 1:
            return 0.0, 0.0
        sample_n = min(self.N, 384)
        if sample_n == self.N:
            sample_idx = np.arange(self.N, dtype=np.int32)
        else:
            sample_idx = np.linspace(0, self.N - 1, sample_n, dtype=np.int32)
        near2_total = 0.0
        near3_total = 0.0
        for idx0 in sample_idx:
            dist = np.count_nonzero(self.enc != self.enc[int(idx0)], axis=1)
            near2_total += float(np.count_nonzero(dist <= 2) - 1)
            near3_total += float(np.count_nonzero(dist <= 3) - 1)
        return near2_total / float(sample_n), near3_total / float(sample_n)

    def _is_hard_candidate_list(self) -> bool:
        """Detect small or intentionally difficult candidate lists."""
        if self.N <= 1500:
            return True
        if self.N <= 3500 and self.avg_near3 >= 18.0:
            return True
        if self.avg_near2 >= 7.0 or self.avg_near3 >= 45.0:
            return True
        return False

    def _make_config(self) -> SolverConfig:
        max_submit_turn = max(1, min(self.max_turns - 1, 99))
        if self.track == 1:
            prefix = ("tares", "solan", "deils")
            return SolverConfig(
                track=1, prefix=prefix,
                pool_size=self._adapt_pool(224 if self.hard_candidate_mode else 160),
                static_pool_size=self._adapt_pool(192 if self.hard_candidate_mode else 128),
                top_post_pool_size=min(160 if self.hard_candidate_mode else 96, self.N),
                min_submit_turn=4, threshold=0.999999, odds_threshold=1e9,
                topk_confirm=0, topk_mass_min=1.0, retry_gap=0,
                force_submit_turn=min(40, max_submit_turn), alpha=1.0, mix_epsilon=0.0,
            )
        if self.track == 2:
            # Exact likelihood is better calibrated than the handoff confusion matrix;
            # keep the handoff's safety gates but use alpha=1.0 for sharper posterior.
            prefix = ("tares", "choli", "bunny")
            return SolverConfig(
                track=2, prefix=prefix,
                pool_size=self._adapt_pool(448 if self.hard_candidate_mode else 288),
                static_pool_size=self._adapt_pool(384 if self.hard_candidate_mode else 256),
                top_post_pool_size=min(288 if self.hard_candidate_mode else 160, self.N),
                min_submit_turn=5, threshold=0.996 if self.hard_candidate_mode else 0.995,
                odds_threshold=800.0 if self.hard_candidate_mode else 500.0,
                topk_confirm=(min(self.N, 256) if self.hard_candidate_mode else min(48, self.N)),
                topk_mass_min=0.96 if self.hard_candidate_mode else 0.90, retry_gap=1 if self.hard_candidate_mode else 3,
                force_submit_turn=min(70, max_submit_turn), alpha=1.0, mix_epsilon=1e-10,
            )
        # Track 3 is very noisy.  Use the handoff's conservative gate, but exact
        # early schedule and exact mutation likelihoods make alpha=0.95 safe enough
        # while avoiding the slow over-diffusion of alpha=0.75.
        prefix = ("tares", "solan", "deils")
        return SolverConfig(
            track=3, prefix=prefix,
            pool_size=self._adapt_pool(672 if self.hard_candidate_mode else 448),
            static_pool_size=self._adapt_pool(384 if self.hard_candidate_mode else 224),
            top_post_pool_size=min(384 if self.hard_candidate_mode else 288, self.N),
            min_submit_turn=5, threshold=0.988 if self.hard_candidate_mode else 0.985,
            odds_threshold=180.0 if self.hard_candidate_mode else 120.0,
            topk_confirm=(min(self.N, 384) if self.hard_candidate_mode else min(96, self.N)),
            topk_mass_min=0.94 if self.hard_candidate_mode else 0.88, retry_gap=1 if self.hard_candidate_mode else 2,
            force_submit_turn=min(60, max_submit_turn), alpha=0.95, mix_epsilon=1e-10,
        )

    def _adapt_pool(self, target: int) -> int:
        # Hard-only candidate lists are often much smaller and more clustered than
        # the full dictionary.  In that regime, evaluating more/all candidate
        # guesses is worth the extra compute because broad static probes may not
        # exist in the supplied list.
        if self.hard_candidate_mode:
            if self.N <= 1800:
                return self.N
            if self.N <= 4000:
                return min(self.N, max(target, 768))
            return min(self.N, max(target, 640))
        if self.N <= 350:
            return self.N
        if self.N <= 1000:
            return min(self.N, max(target, 224))
        return min(self.N, target)

    def _make_prefix_schedules(self) -> List[Tuple[Tuple[int, int, int], float]]:
        """Exact final type distribution for the first three forced-noise turns."""
        p, q = max(0.0, self.p_one), max(0.0, self.p_two)
        r = max(0.0, 1.0 - p - q)
        schedules: Dict[Tuple[int, int, int], float] = {}
        if self.track == 1 or (p <= 0.0 and q <= 0.0):
            for j in range(3):
                seq = [0, 0, 0]
                seq[j] = 1
                schedules[tuple(seq)] = schedules.get(tuple(seq), 0.0) + 1.0 / 3.0
            return list(schedules.items())
        base = [r, p, q]
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    seq = (a, b, c)
                    prob = base[a] * base[b] * base[c]
                    if prob > 0.0 and seq != (0, 0, 0):
                        schedules[seq] = schedules.get(seq, 0.0) + prob
        # If the initially sampled first three turns are all clean, one of them is
        # forcibly changed to one- or two-letter noise in proportion to p:q.
        all_clean = r ** 3
        denom = p + q
        if all_clean > 0.0:
            if denom <= 0.0:
                for j in range(3):
                    seq = [0, 0, 0]
                    seq[j] = 1
                    schedules[tuple(seq)] = schedules.get(tuple(seq), 0.0) + all_clean / 3.0
            else:
                for j in range(3):
                    seq1 = [0, 0, 0]
                    seq1[j] = 1
                    schedules[tuple(seq1)] = schedules.get(tuple(seq1), 0.0) + all_clean * (p / denom) / 3.0
                    seq2 = [0, 0, 0]
                    seq2[j] = 2
                    schedules[tuple(seq2)] = schedules.get(tuple(seq2), 0.0) + all_clean * (q / denom) / 3.0
        total = sum(schedules.values())
        if total <= 0.0:
            return [((1, 0, 0), 1.0 / 3.0), ((0, 1, 0), 1.0 / 3.0), ((0, 0, 1), 1.0 / 3.0)]
        return [(seq, prob / total) for seq, prob in schedules.items()]

    def _make_static_pool(self) -> np.ndarray:
        """Cheap start-time heuristic pool; exact entropy is evaluated lazily later."""
        N = self.N
        letter_freq = np.zeros(26, dtype=np.float64)
        pos_freq = np.zeros((5, 26), dtype=np.float64)
        for row in self.enc:
            seen = set()
            for pos, ch in enumerate(row):
                c = int(ch)
                pos_freq[pos, c] += 1.0
                seen.add(c)
            for c in seen:
                letter_freq[c] += 1.0
        # Smooth and sqrt-compress so very common letters do not dominate too much.
        letter_score = np.sqrt(letter_freq + 1.0)
        pos_score = np.sqrt(pos_freq + 1.0)
        scores = np.empty(N, dtype=np.float64)
        for i, row in enumerate(self.enc):
            unique = set(int(c) for c in row)
            s = 1.25 * sum(float(letter_score[c]) for c in unique)
            s += 0.45 * sum(float(pos_score[pos, int(c)]) for pos, c in enumerate(row))
            s -= 2.0 * (5 - len(unique))
            # Prefer words that cover different positions/letters and avoid exotic duplicate probes.
            scores[i] = s
        k = min(max(self.cfg.static_pool_size, self.cfg.pool_size), N)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return top.astype(np.int32)

    def _initial_entropy_probe_indices(self, count: int = 3) -> List[int]:
        """Pick robust openers from the supplied candidate list itself.

        When the candidate list contains only hard words, the usual broad opener
        words may be absent.  For small/clustered lists we spend a little start
        time selecting candidate guesses that split the provided list well under
        clean Wordle feedback.  This is intentionally capped so /start_problem
        remains fast even for medium-sized custom lists.
        """
        if self.N <= 1:
            return [0]
        pool_limit = 512 if self.hard_candidate_mode else 256
        if self.N <= pool_limit:
            pool = np.arange(self.N, dtype=np.int32)
        else:
            pool = self.static_pool[:min(len(self.static_pool), pool_limit)].astype(np.int32)
        weights = np.ones(self.N, dtype=np.float64) / float(self.N)
        selected: List[int] = []
        used = set()
        for _ in range(min(count, self.N)):
            best_idx = None
            best_score = -1e100
            for gi0 in pool:
                gi = int(gi0)
                if gi in used:
                    continue
                row = self.row(gi)
                mass = np.bincount(row, weights=weights, minlength=N_PATTERNS).astype(np.float64)
                score = entropy_from_mass(mass)
                # Avoid picking three almost identical probes from the same hard
                # cluster.  The first probe is pure entropy; later probes receive
                # a small diversity penalty if they are too close to an already
                # selected opener.
                for prev in selected:
                    h = self._hamming_idx(gi, prev)
                    if h <= 2:
                        score -= 0.20
                    elif h == 3:
                        score -= 0.06
                if score > best_score:
                    best_score = score
                    best_idx = gi
            if best_idx is None:
                break
            selected.append(int(best_idx))
            used.add(int(best_idx))
        return selected

    def _make_prefix_indices(self) -> List[int]:
        indices: List[int] = []
        used = set()
        if self.hard_candidate_mode:
            for idx in self._initial_entropy_probe_indices(count=3):
                if idx not in used:
                    indices.append(int(idx))
                    used.add(int(idx))
                if len(indices) >= 3:
                    return indices
        for w in self.cfg.prefix:
            idx = self.word_to_idx.get(w)
            if idx is not None and idx not in used:
                indices.append(int(idx))
                used.add(int(idx))
        for idx in self.static_pool:
            ii = int(idx)
            if len(indices) >= 3:
                break
            if ii not in used:
                indices.append(ii)
                used.add(ii)
        if not indices:
            indices = [0]
        return indices

    def row(self, guess_idx: int) -> np.ndarray:
        guess_idx = int(guess_idx)
        got = self.row_cache.get(guess_idx)
        if got is None:
            got = feedback_ids_for_guess_with_counts(self.enc[guess_idx], self.enc, self.counts)
            self.row_cache[guess_idx] = got
        return got

    def _category_info(self, guess_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        distinct = sorted(int(x) for x in np.unique(guess_vec))
        cats = distinct + [-1]
        other_letter = next(x for x in range(26) if x not in distinct)
        C = len(cats)
        reps = np.empty((5, C), dtype=np.uint8)
        mult = np.empty((5, C, self.N), dtype=np.uint16)
        in_guess = np.zeros(26, dtype=bool)
        in_guess[distinct] = True
        D = len(distinct)
        for pos in range(5):
            orig = self.enc[:, pos]
            for ci, cat in enumerate(cats):
                if cat >= 0:
                    reps[pos, ci] = cat
                    mult[pos, ci] = (orig != cat).astype(np.uint16)
                else:
                    reps[pos, ci] = other_letter
                    # Count all non-guess replacement letters, excluding original
                    # when original is itself a non-guess letter.
                    mult[pos, ci] = (26 - D - (~in_guess[orig]).astype(np.uint16)).astype(np.uint16)
        return reps, mult, np.array(cats, dtype=np.int16)

    def mutation_likelihoods(self, guess_idx: int, observed_pid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact L0, L1, L2 likelihood vectors for one observed feedback."""
        clean_row = self.row(guess_idx)
        obs = int(observed_pid)
        L0 = (clean_row == obs).astype(np.float64)
        guess_vec = self.enc[int(guess_idx)]
        reps, mult, _ = self._category_info(guess_vec)
        C = reps.shape[1]
        c1 = np.zeros(self.N, dtype=np.uint16)
        c2_u32 = np.zeros(self.N, dtype=np.uint32)
        # One-letter mutation counts.  Denominator: 5 positions * 25 replacement letters.
        for pos in range(5):
            for ci in range(C):
                m = mult[pos, ci]
                if not np.any(m):
                    continue
                eff = self.enc.copy()
                eff[:, pos] = reps[pos, ci]
                pids = feedback_ids_for_guess(guess_vec, eff)
                c1 += ((pids == obs).astype(np.uint16) * m).astype(np.uint16)
        # Two-letter mutation counts.  Denominator: C(5,2) position pairs * 25 * 25.
        if self.p_two > 0.0 or self.track == 3:
            for pos1, pos2 in PAIRS:
                for ci in range(C):
                    m_i = mult[pos1, ci].astype(np.uint32)
                    if not np.any(m_i):
                        continue
                    for cj in range(C):
                        prod = m_i * mult[pos2, cj].astype(np.uint32)
                        if not np.any(prod):
                            continue
                        eff = self.enc.copy()
                        eff[:, pos1] = reps[pos1, ci]
                        eff[:, pos2] = reps[pos2, cj]
                        pids = feedback_ids_for_guess(guess_vec, eff)
                        c2_u32 += (pids == obs).astype(np.uint32) * prod
        L1 = c1.astype(np.float64) / 125.0
        L2 = c2_u32.astype(np.float64) / 6250.0
        return L0, L1, L2

    def _first3_posterior_from_likelihoods(self) -> np.ndarray:
        k = min(3, len(self.likelihood_history))
        if k == 0:
            return np.ones(self.N, dtype=np.float64) / self.N
        accum = np.zeros(self.N, dtype=np.float64)
        like_by_t = [(L0, L1, L2) for _, L0, L1, L2 in self.likelihood_history[:k]]
        for seq, prior in self.prefix_schedules:
            prod = np.full(self.N, prior, dtype=np.float64)
            for t in range(k):
                prod *= like_by_t[t][seq[t]]
            accum += prod
        return normalize(accum)

    def _replay_exact_topk(self, topk: np.ndarray, skip_history_index: Optional[int] = None) -> np.ndarray:
        """Replay an untempered posterior on a supplied top-K candidate set.

        The original confirmation code replayed the current top-K set once.  For
        submit safety we also need leave-one-observation-out checks.  This helper
        keeps the exact first-three forced-noise marginalization, even when one of
        the prefix observations is skipped: each remaining prefix observation uses
        the noise type from its actual turn in the joint schedule.
        """
        K = int(len(topk))
        if K == 0:
            return np.zeros(0, dtype=np.float64)

        prefix_items: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        later_items: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for hist_i, (turn0, L0, L1, L2) in enumerate(self.likelihood_history):
            if skip_history_index is not None and hist_i == int(skip_history_index):
                continue
            if int(turn0) <= 3:
                prefix_items.append((int(turn0), L0, L1, L2))
            else:
                later_items.append((int(turn0), L0, L1, L2))

        if prefix_items:
            accum = np.zeros(K, dtype=np.float64)
            for seq, prior in self.prefix_schedules:
                prod = np.full(K, prior, dtype=np.float64)
                for turn0, L0, L1, L2 in prefix_items:
                    noise_type = int(seq[max(0, min(2, turn0 - 1))])
                    prod *= (L0, L1, L2)[noise_type][topk]
                accum += prod
            local = normalize(accum)
        else:
            local = np.ones(K, dtype=np.float64) / K

        for turn0, L0, L1, L2 in later_items:
            if self.track == 1:
                like = L0[topk]
            else:
                p0, p1, p2 = self.type_probs_after_prefix(turn0)
                like = p0 * L0[topk] + p1 * L1[topk] + p2 * L2[topk]
            local = normalize(local * np.maximum(like, 1e-15))
        return local

    def _local_stats(self, topk: np.ndarray, local: np.ndarray) -> Tuple[int, float, float, float, np.ndarray]:
        order = np.argsort(local)[::-1]
        winner = int(topk[int(order[0])])
        pmax = float(local[int(order[0])])
        p2 = float(local[int(order[1])]) if len(order) > 1 else 0.0
        odds = pmax / max(p2, 1e-300)
        return winner, pmax, p2, odds, order

    def _hamming_idx(self, a: int, b: int) -> int:
        return int(np.count_nonzero(self.enc[int(a)] != self.enc[int(b)]))

    def _letter_overlap_idx(self, a: int, b: int) -> int:
        return int(np.minimum(self.counts[int(a)], self.counts[int(b)]).sum())

    def _is_close_competitor(self, a: int, b: int) -> bool:
        """Structural closeness guard for near-neighbour submit failures."""
        aw = self.words[int(a)]
        bw = self.words[int(b)]
        h = self._hamming_idx(a, b)
        if h <= 2:
            return True
        overlap = self._letter_overlap_idx(a, b)
        if h <= 3 and overlap >= 4:
            return True
        if h <= 3 and (aw[:2] == bw[:2] or aw[-2:] == bw[-2:]):
            return True
        if self.hard_candidate_mode:
            # Hard-only lists may be intentionally made of words that share most
            # letters but differ across several positions.  Treat these as a
            # submit-risk cluster even when Hamming distance is four.
            if h <= 4 and overlap >= 4:
                return True
            if h <= 4 and (aw[:2] == bw[:2] or aw[-2:] == bw[-2:]):
                return True
        return False

    def _close_competitors_from_local(self, topk: np.ndarray, local: np.ndarray, top: int, limit: int = 16) -> List[int]:
        """Return close alternatives to top, ordered by exact local posterior."""
        if len(topk) <= 1:
            return []
        order = np.argsort(local)[::-1]
        out: List[int] = []
        # Very small probabilities can still matter after noisy evidence, but we
        # do not want to direct-query dozens of posterior dust words.
        floor = 1e-5 if self.track == 3 else 3e-5
        for loc in order:
            idx = int(topk[int(loc)])
            if idx == int(top):
                continue
            if float(local[int(loc)]) < floor and len(out) >= 3:
                break
            if self._is_close_competitor(int(top), idx):
                out.append(idx)
                if len(out) >= limit:
                    break
        return out

    def _find_local_position(self, topk: np.ndarray, idx: int) -> Optional[int]:
        hits = np.flatnonzero(topk == int(idx))
        if len(hits) == 0:
            return None
        return int(hits[0])

    def _suggest_submit_probe(self, topk: np.ndarray, local: np.ndarray, top: int, close: List[int]) -> int:
        """Choose the next direct probe after a submit veto."""
        direct_top = self._direct_query_count(int(top))
        # T2 failures were mostly early near-neighbour submits; two direct probes
        # of the proposed top are a cheap calibration step.  T3 is much noisier,
        # so use three direct probes when a close cluster is still alive.
        need_top = 3 if self.track == 3 and close else 2
        if direct_top < need_top:
            return int(top)

        best_idx = int(top)
        best_score = -1.0
        for idx in close[:8]:
            pos = self._find_local_position(topk, int(idx))
            if pos is None:
                continue
            prob = float(local[pos])
            dq = self._direct_query_count(int(idx))
            # Query the most plausible unconfirmed close competitor first.
            if dq >= (2 if self.track == 3 and prob > 0.01 else 1):
                continue
            score = prob / float(dq + 1)
            if idx not in self.guessed:
                score *= 1.15
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        if best_idx != int(top):
            return best_idx
        if self.track == 3 and close:
            return self._endgame_discriminator(int(top), k=min(128, max(32, len(topk))))
        return int(top)

    def _full_exact_topk_confirmation(self, topk: np.ndarray, approx_top: int, turn: int) -> dict:
        """Replay exact posterior on top-K and apply conservative submit safety gates."""
        K = int(len(topk))
        if K == 0:
            return {"passed": False, "reason": "empty_topk"}

        local = self._replay_exact_topk(topk)
        winner, exact_pmax, exact_p2, exact_odds, order = self._local_stats(topk, local)
        topk_mass = float(np.sum(self.pi[topk]))
        base_pass = bool(
            winner == int(approx_top)
            and exact_pmax >= self.cfg.threshold
            and exact_odds >= self.cfg.odds_threshold
            and topk_mass >= self.cfg.topk_mass_min
        )
        info = {
            "passed": base_pass,
            "winner": winner,
            "exact_pmax": exact_pmax,
            "exact_odds": exact_odds,
            "topk_mass": topk_mass,
            "K": K,
            "reason": "base_pass" if base_pass else "base_gate",
            "suggested_probe": int(winner),
        }
        if not base_pass:
            info["suggested_probe"] = int(winner if winner != int(approx_top) else approx_top)
            return info

        if self.track == 1:
            return info

        close = self._close_competitors_from_local(topk, local, int(approx_top), limit=(24 if self.hard_candidate_mode else 16))
        info["close_count"] = len(close)
        if close:
            direct_top = self._direct_query_count(int(approx_top))
            need_top = 3 if self.track == 3 else 2
            if direct_top < need_top:
                info.update({
                    "passed": False,
                    "reason": "need_more_direct_top",
                    "suggested_probe": int(approx_top),
                })
                return info

            # Do not submit while a close competitor with non-negligible exact mass
            # has never been directly queried.  This is the main fix for the 29
            # wrong-submit cases, all of which were close-cluster overconfidence.
            if self.hard_candidate_mode:
                comp_floor = 2e-5 if self.track == 2 else 5e-6
                close_scan = 10
            else:
                comp_floor = 2e-4 if self.track == 2 else 7e-5
                close_scan = 6
            for idx in close[:close_scan]:
                pos = self._find_local_position(topk, idx)
                prob = float(local[pos]) if pos is not None else 0.0
                if prob >= comp_floor and self._direct_query_count(idx) == 0:
                    info.update({
                        "passed": False,
                        "reason": "unqueried_close_competitor",
                        "suggested_probe": int(idx),
                    })
                    return info

            # Leave-one-observation-out stability.  If removing one recent piece of
            # evidence changes the winner to a close competitor, the posterior is
            # still brittle, so ask that competitor/top once more instead of
            # risking a 100-turn wrong submit.
            stable_until = (30 if self.track == 2 else 40) if self.hard_candidate_mode else (22 if self.track == 2 else 30)
            if turn < stable_until:
                start = max(0, len(self.likelihood_history) - 8)
                for hist_i in range(start, len(self.likelihood_history)):
                    loo = self._replay_exact_topk(topk, skip_history_index=hist_i)
                    loo_winner, loo_pmax, _, loo_odds, _ = self._local_stats(topk, loo)
                    if loo_winner != int(approx_top) and self._is_close_competitor(int(approx_top), loo_winner):
                        probe = loo_winner if self._direct_query_count(loo_winner) == 0 else int(approx_top)
                        info.update({
                            "passed": False,
                            "reason": "loo_unstable_close_winner",
                            "suggested_probe": int(probe),
                            "loo_winner": int(loo_winner),
                            "loo_odds": float(loo_odds),
                            "loo_pmax": float(loo_pmax),
                        })
                        return info

        return info

    def type_probs_after_prefix(self, turn: int) -> Tuple[float, float, float]:
        if self.track == 1:
            return 1.0, 0.0, 0.0
        p1 = max(0.0, self.p_one)
        p2 = max(0.0, self.p_two)
        p0 = max(0.0, 1.0 - p1 - p2)
        s = p0 + p1 + p2
        if s <= 0.0:
            return 1.0, 0.0, 0.0
        return p0 / s, p1 / s, p2 / s

    def observe(self, turn: int, feedback_text: Optional[str]) -> None:
        """Process feedback for the previous guess, if any."""
        if feedback_text is None or self.pending_guess_idx is None:
            return
        # Avoid double-processing if a client retries the same /act.
        if self.pending_guess_turn is not None and self.pending_guess_turn <= self.last_processed_turn:
            return
        obs = parse_feedback(feedback_text)
        if obs is None:
            return
        guess_idx = int(self.pending_guess_idx)
        guess_turn = int(self.pending_guess_turn or max(1, turn - 1))
        L0, L1, L2 = self.mutation_likelihoods(guess_idx, obs)
        self.history.append((guess_turn, guess_idx, int(obs)))
        self.likelihood_history.append((guess_turn, L0, L1, L2))
        self.last_processed_turn = guess_turn

        if guess_turn <= 3:
            self.pi = self._first3_posterior_from_likelihoods()
        else:
            if self.track == 1:
                like = L0
                self.pi = normalize(self.pi * like)
            else:
                p0, p1, p2 = self.type_probs_after_prefix(guess_turn)
                like = p0 * L0 + p1 * L1 + p2 * L2
                powered = np.power(np.maximum(like, 1e-15), self.cfg.alpha)
                self.pi = normalize(self.pi * powered)
                if self.cfg.mix_epsilon > 0.0:
                    self.pi = normalize((1.0 - self.cfg.mix_epsilon) * self.pi + self.cfg.mix_epsilon / self.N)
        self.pending_guess_idx = None
        self.pending_guess_turn = None

    def maybe_submit(self, turn: int) -> Optional[int]:
        if self.N == 1:
            return 0
        top, pmax, p2, odds = posterior_stats(self.pi)
        if self.track == 1:
            active = np.flatnonzero(self.pi > 0.0)
            if turn >= self.cfg.min_submit_turn and len(active) == 1:
                return int(active[0])
            if turn >= self.cfg.force_submit_turn:
                return int(top)
            return None
        if turn < self.cfg.min_submit_turn:
            return None
        force = turn >= self.cfg.force_submit_turn
        gate = pmax >= self.cfg.threshold and odds >= self.cfg.odds_threshold
        if not gate:
            # A forced wrong submit has the same 100-turn cap as not solving, so
            # only use the force path at the very end as a last-resort guess.
            if force and turn >= max(self.max_turns - 1, self.cfg.force_submit_turn):
                return int(top)
            return None
        # Track 3 safety guard: a single noisy observation of the current top
        # answer can make the exact posterior look overconfident.  Before
        # submitting in the high two-letter-noise track, require that the
        # proposed answer has been directly queried at least twice.  choose_guess()
        # already repeats the top answer in this situation, so this usually costs
        # one extra turn but prevents one-noisy-hit wrong submits.
        if self.track == 3:
            direct_top_queries = sum(1 for _, gi, _ in self.history if int(gi) == int(top))
            if direct_top_queries < 2:
                return None
        if turn < self.exact_confirm_veto_until:
            return None
        K = min(max(2, self.cfg.topk_confirm), self.N)
        topk = np.argpartition(self.pi, -K)[-K:]
        info = self._full_exact_topk_confirmation(topk.astype(np.int32), int(top), int(turn))
        if bool(info.get("passed", False)):
            return int(top)
        probe = info.get("suggested_probe")
        if probe is not None and 0 <= int(probe) < self.N:
            self.next_forced_probe_idx = int(probe)
        self.exact_confirm_veto_until = turn + self.cfg.retry_gap
        return None

    def choose_pool(self) -> np.ndarray:
        if self.N <= self.cfg.pool_size:
            return np.arange(self.N, dtype=np.int32)
        topn = min(self.cfg.top_post_pool_size, self.N)
        top_post = np.argpartition(self.pi, -topn)[-topn:].astype(np.int32)
        base_parts = [self.static_pool[: min(len(self.static_pool), self.cfg.static_pool_size)], top_post]
        if self.prefix_indices:
            base_parts.append(np.array(self.prefix_indices, dtype=np.int32))
        pool = np.unique(np.concatenate(base_parts))
        if len(pool) <= self.cfg.pool_size:
            return pool.astype(np.int32)
        # Preserve a mix of globally informative probes and current posterior heads.
        keep_post = min(len(top_post), max(48, self.cfg.pool_size // 2))
        best_post = top_post[np.argsort(self.pi[top_post])[-keep_post:]]
        remaining = self.cfg.pool_size - len(np.unique(best_post))
        static_keep = self.static_pool[: max(0, remaining)]
        pool = np.unique(np.concatenate([best_post, static_keep, np.array(self.prefix_indices, dtype=np.int32)]))
        if len(pool) > self.cfg.pool_size:
            # Drop the lowest-posterior members among the non-static overflow.
            pool = pool[np.argsort(self.pi[pool])[-self.cfg.pool_size:]]
        return pool.astype(np.int32)

    def _top_history_counts(self, top: int) -> Tuple[int, int, int]:
        direct = 0
        recent4 = 0
        consecutive = 0
        for _, gi, _ in self.history:
            if int(gi) == int(top):
                direct += 1
        for _, gi, _ in self.history[-4:]:
            if int(gi) == int(top):
                recent4 += 1
        for _, gi, _ in reversed(self.history):
            if int(gi) == int(top):
                consecutive += 1
            else:
                break
        return direct, recent4, consecutive


    def _direct_query_count(self, idx: int) -> int:
        return sum(1 for _, gi, _ in self.history if int(gi) == int(idx))

    def _alternate_head_probe(self, top: int, k: int = 12) -> Optional[int]:
        """Probe a plausible runner-up when T3 is stuck on a wrong top.

        Long T3 tails often happen when a near-neighbour word becomes the
        posterior top after two noisy direct observations.  If the top has
        already been queried twice but still cannot pass the submit gate, the
        most useful next action is often to directly query a high-probability
        runner-up, not to repeat the same top or ask a broad entropy probe.
        """
        if self.N <= 1:
            return None
        k = min(max(3, k), self.N)
        head = np.argpartition(self.pi, -k)[-k:].astype(np.int32)
        head = head[np.argsort(self.pi[head])[::-1]]
        top_p = float(self.pi[int(top)])
        best = None
        best_score = -1.0
        for idx0 in head:
            idx = int(idx0)
            if idx == int(top):
                continue
            prob = float(self.pi[idx])
            # Very tiny runner-ups are usually posterior dust; broad
            # discriminator guesses are better for those.
            if prob < 0.035 and prob < 0.12 * max(top_p, 1e-300):
                continue
            dq = self._direct_query_count(idx)
            if dq >= 2:
                continue
            # Prefer high posterior and words that have not already consumed
            # direct confirmations.
            score = prob / float(dq + 1)
            if idx not in self.guessed:
                score *= 1.10
            if score > best_score:
                best_score = score
                best = idx
        return best

    def _endgame_discriminator(self, top: int, k: int = 96) -> int:
        """Pick a probe that separates the current posterior head cluster.

        In T3, repeatedly asking the same top word can waste turns after two
        confirmations, because most observations are two-letter noisy.  This
        scorer ignores the tiny posterior tail and maximizes entropy only on the
        top candidate cluster, which is what matters in the long-tail cases.
        """
        k = min(max(8, k), self.N)
        head = np.argpartition(self.pi, -k)[-k:].astype(np.int32)
        head_w = self.pi[head].astype(np.float64)
        sw = float(np.sum(head_w))
        if sw <= 0.0:
            return int(top)
        head_w /= sw
        pool = self.choose_pool()
        if int(top) not in set(map(int, pool)):
            pool = np.concatenate([pool, np.array([int(top)], dtype=np.int32)])
        best_score = -1e100
        best_idx = int(top)
        for gi0 in pool:
            gi = int(gi0)
            row = self.row(gi)[head]
            mass = np.bincount(row, weights=head_w, minlength=N_PATTERNS).astype(np.float64)
            score = entropy_from_mass(mass)
            # When close to submitting, prefer actual answer candidates in the
            # posterior head; otherwise use globally good probes if they split
            # the head cluster better.
            score += 0.04 * float(self.pi[gi])
            if gi == int(top):
                score -= 0.06
            if gi in self.guessed:
                score -= 0.02
            if score > best_score:
                best_score = score
                best_idx = gi
        return int(best_idx)

    def choose_guess(self, turn: int) -> int:
        # Fixed robust opening for the guaranteed early-noise region.  If a fixed
        # word is absent from the candidate list, prefix_indices already falls back
        # to high-coverage candidate words.
        top, pmax, _, _ = posterior_stats(self.pi)
        if self.next_forced_probe_idx is not None and turn > 3:
            gi = int(self.next_forced_probe_idx)
            self.next_forced_probe_idx = None
            if 0 <= gi < self.N:
                return gi
        if turn <= 3 and turn <= len(self.prefix_indices) and pmax < 0.98:
            gi = int(self.prefix_indices[turn - 1])
            return gi
        # Endgame under T3: two direct top queries are useful for safety, but
        # repeatedly asking the same word after that gives little information
        # because most feedback is two-letter noisy.  Switch to a head-cluster
        # discriminator once the top has already been checked enough.
        if self.track == 3 and turn >= 5:
            direct_top, recent_top, consecutive_top = self._top_history_counts(int(top))
            if direct_top >= 2:
                # If the current top has already been directly checked twice
                # but still did not pass maybe_submit(), do not keep circling on
                # the same word.  First try a high-posterior runner-up direct
                # probe; this specifically cuts cases like footy/zooty/lofty
                # where the true answer sits at rank 2-5 for several turns.
                alt = self._alternate_head_probe(int(top), k=22 if self.hard_candidate_mode else 14)
                if alt is not None and (pmax < 0.995 or consecutive_top >= 2):
                    return int(alt)
                if pmax >= (0.975 if self.hard_candidate_mode else 0.985):
                    return self._endgame_discriminator(int(top), k=160 if self.hard_candidate_mode else 96)
            if direct_top < 2 and (pmax >= 0.72 or (pmax >= 0.42 and recent_top < 2)):
                return int(top)
            if pmax >= 0.55 and recent_top >= 2:
                alt = self._alternate_head_probe(int(top), k=18 if self.hard_candidate_mode else 10)
                if alt is not None:
                    return int(alt)
                return self._endgame_discriminator(int(top), k=192 if self.hard_candidate_mode else 128)
        if self.hard_candidate_mode and self.track == 2 and turn >= 5:
            direct_top, recent_top, consecutive_top = self._top_history_counts(int(top))
            if pmax >= 0.88 and direct_top < 2:
                return int(top)
            if pmax >= 0.74 and recent_top >= 1:
                alt = self._alternate_head_probe(int(top), k=20)
                if alt is not None and (pmax < 0.97 or consecutive_top >= 2):
                    return int(alt)
                return self._endgame_discriminator(int(top), k=128)
            if pmax >= 0.55 and len(self.history) >= 4:
                return self._endgame_discriminator(int(top), k=160)

        if self.track == 2 and pmax >= 0.82:
            # Before submitting T2 near-neighbour cases, maybe_submit() may set a
            # forced probe.  If not, one extra top query is still the fastest way
            # to calibrate a high posterior.
            return int(top)

        pool = self.choose_pool()
        weights = self.pi
        best_score = -1e100
        best_idx = int(pool[0])
        for gi0 in pool:
            gi = int(gi0)
            row = self.row(gi)
            mass = np.bincount(row, weights=weights, minlength=N_PATTERNS).astype(np.float64)
            score = entropy_from_mass(mass)
            # Prefer plausible final answers slightly; avoid wasting probes on the
            # exact same low-value repeated word, except in endgame handled above.
            score += 0.08 * float(self.pi[gi])
            if gi in self.guessed:
                score -= 0.03
            if score > best_score:
                best_score = score
                best_idx = gi
        return int(best_idx)

    def act(self, turn: int, feedback_text: Optional[str]) -> dict:
        self.observe(turn, feedback_text)
        submit_idx = self.maybe_submit(turn)
        if submit_idx is not None:
            self.last_action = {"action": "submit", "word": self.words[int(submit_idx)]}
            return self.last_action
        guess_idx = self.choose_guess(turn)
        self.guessed.add(int(guess_idx))
        self.pending_guess_idx = int(guess_idx)
        self.pending_guess_turn = int(turn)
        self.last_action = {"action": "guess", "word": self.words[int(guess_idx)]}
        return self.last_action


SOLVERS: Dict[str, NoisyWordleSolver] = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # keep grader stdout/stderr clean
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            data = self._read_json()
            if self.path == "/start_problem":
                pid = str(data.get("problem_id", "default"))
                SOLVERS.clear()  # only one active problem is required by the spec
                SOLVERS[pid] = NoisyWordleSolver(data)
                self._write_json({})
                return
            if self.path == "/act":
                pid = str(data.get("problem_id", "default"))
                solver = SOLVERS.get(pid)
                if solver is None:
                    # Unknown problem_id is a protocol error.  Returning a fake
                    # candidate word such as "error" would itself be an invalid
                    # action under the grader's candidate-list rule.
                    self._write_json({"error": "unknown_problem_id"}, status=404)
                    return
                turn = int(data.get("turn", len(solver.history) + 1) or 1)
                action = solver.act(turn, data.get("feedback"))
                self._write_json(action)
                return
            self._write_json({"error": "not_found"}, status=404)
        except Exception:
            # A malformed response is an immediate failure.  Emit a valid action if
            # possible, and print diagnostics only to stderr for local debugging.
            traceback.print_exc(file=sys.stderr)
            try:
                if SOLVERS:
                    solver = next(iter(SOLVERS.values()))
                    top, _, _, _ = posterior_stats(solver.pi)
                    self._write_json({"action": "guess", "word": solver.words[int(top)]})
                else:
                    self._write_json({"error": "internal"}, status=500)
            except Exception:
                self._write_json({"error": "internal"}, status=500)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Noisy Wordle solver listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
