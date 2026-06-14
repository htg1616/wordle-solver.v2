#!/usr/bin/env python3
"""Merge semi_full_benchmark chunk artifacts into one result directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(x) for x in values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def summarize_rows(rows: Sequence[Dict[str, Any]], slow_threshold: int) -> Dict[str, Any]:
    def group_summary(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}
        scores = [int(r.get("score_turns", 0)) for r in group]
        turns_success = [int(r.get("turns", 0)) for r in group if int(r.get("success", 0)) == 1]
        statuses: Dict[str, int] = {}
        for r in group:
            status = str(r.get("status", ""))
            statuses[status] = statuses.get(status, 0) + 1
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
            "statuses": dict(sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    by_track: Dict[str, List[Dict[str, Any]]] = {}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_track.setdefault(str(row.get("track", "")), []).append(row)
        by_source.setdefault(str(row.get("case_source", "unknown")), []).append(row)
    return {"overall": group_summary(rows), "tracks": {k: group_summary(v) for k, v in sorted(by_track.items())}, "sources": {k: group_summary(v) for k, v in sorted(by_source.items())}}


def read_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, int]] = set()
    for csv_path in sorted(root.rglob("games.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (str(row.get("secret", "")), int(row.get("track", 0)), int(row.get("seed", 0)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(row))
    return rows


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], slow_threshold: int, merge_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r.get("selected_index", 0)), int(r.get("track", 0)), int(r.get("seed", 0))))
    fields: List[str] = []
    seen_fields = set()
    for row in rows:
        for field in row:
            if field not in seen_fields:
                fields.append(field)
                seen_fields.add(field)
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
        item = {
            "track": int(row.get("track", 0)),
            "seed": int(row.get("seed", 0)),
            "case_source": row.get("case_source", ""),
            "success": bool(int(row.get("success", 0))),
            "status": row.get("status", ""),
            "score_turns": score,
            "turns": int(row.get("turns", 0)),
            "act_wall_s": row.get("act_wall_s", ""),
            "last_word": row.get("last_word", ""),
        }
        if score > slow_threshold:
            slow_by_secret.setdefault(str(row.get("secret", "")), []).append(item)
        if int(row.get("success", 1)) == 0:
            failed_by_secret.setdefault(str(row.get("secret", "")), []).append(item)
    slow_list = sorted(({"secret": secret, "max_score_turns": max(e["score_turns"] for e in entries), "cases": entries} for secret, entries in slow_by_secret.items()), key=lambda x: (-int(x["max_score_turns"]), str(x["secret"])))
    failed_list = sorted(({"secret": secret, "failures": entries} for secret, entries in failed_by_secret.items()), key=lambda x: str(x["secret"]))
    (out_dir / "slow_words.json").write_text(json.dumps({"threshold_turns": slow_threshold, "count": len(slow_list), "words": slow_list}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "failed_words.json").write_text(json.dumps({"count": len(failed_list), "words": failed_list}, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summarize_rows(rows, slow_threshold)
    summary["name"] = "semi_full_benchmark"
    summary["merge"] = {"merge_root": str(merge_root), "chunks_found": len(list(merge_root.rglob("games.csv"))), "rows_merged": len(rows), "slow_threshold_turns": slow_threshold}
    summary["outputs"] = {"games_csv": str(games_csv), "games_jsonl": str(games_jsonl), "slow_words": str(out_dir / "slow_words.json"), "failed_words": str(out_dir / "failed_words.json")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge semi_full_benchmark chunk artifacts")
    parser.add_argument("--merge-root", required=True)
    parser.add_argument("--out-dir", default="benchmark/results/semi_full_benchmark_merged")
    parser.add_argument("--slow-threshold", type=int, default=15)
    args = parser.parse_args()
    root = Path(args.merge_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows = read_rows(root)
    if not rows:
        raise SystemExit(f"No games.csv rows found under {root}")
    write_outputs(out_dir, rows, args.slow_threshold, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
