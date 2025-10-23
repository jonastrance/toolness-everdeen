#!/usr/bin/env python3
"""Personal Momentum Map CLI.

This lightweight tool keeps track of momentum scores across projects,
habits, and goals. It stores data in a local JSON file and provides a
textual interface for quick updates, momentum visualization, and
daily recalibration suggestions.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

DATA_FILE = Path(__file__).with_name("momentum_data.json")
DECAY_PER_DAY = 1.8  # points lost per day without attention
MAX_SCORE = 100.0
MIN_SCORE = 0.0
HISTORY_LIMIT = 200  # keep the last N events per area

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(ts: datetime) -> str:
    return ts.strftime(ISO_FORMAT)


def parse_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, ISO_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        # Accept older timestamps without microseconds
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None


def load_data() -> Dict:
    if DATA_FILE.exists():
        try:
            with DATA_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.setdefault("areas", {})
    return data


def save_data(data: Dict) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


@dataclass
class AreaSnapshot:
    name: str
    score: float
    trend: float
    last_update: Optional[datetime]
    energy_state: str


def apply_decay(area: Dict, now: datetime) -> bool:
    """Apply natural momentum decay since the last decay timestamp."""
    last_decay_ts = parse_ts(area.get("last_decay_ts"))
    if last_decay_ts is None:
        last_decay_ts = parse_ts(area.get("last_update_ts"))
    if last_decay_ts is None:
        area["last_decay_ts"] = fmt_ts(now)
        return False

    elapsed_days = (now - last_decay_ts).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return False

    decay_amount = elapsed_days * DECAY_PER_DAY
    if decay_amount <= 0:
        area["last_decay_ts"] = fmt_ts(now)
        return False

    original_score = area.get("score", 50.0)
    new_score = max(MIN_SCORE, original_score - decay_amount)
    area["score"] = new_score
    area["last_decay_ts"] = fmt_ts(now)
    return not math.isclose(original_score, new_score, rel_tol=1e-9, abs_tol=1e-9)


def apply_decay_all(data: Dict, now: datetime) -> None:
    for area in data.get("areas", {}).values():
        apply_decay(area, now)


def ensure_area(data: Dict, name: str) -> Dict:
    areas = data.setdefault("areas", {})
    if name not in areas:
        areas[name] = {
            "score": 50.0,
            "history": [],
            "last_update_ts": None,
            "last_decay_ts": fmt_ts(utcnow()),
        }
    return areas[name]


def clamp_score(score: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, score))


def update_area(data: Dict, name: str, delta: float, note: Optional[str], now: datetime) -> Tuple[float, float]:
    area = ensure_area(data, name)
    apply_decay(area, now)

    original_score = area.get("score", 50.0)
    # Momentum updates are weighted to reflect both progress and emotional energy
    weighted_delta = delta * 2.5
    new_score = clamp_score(original_score + weighted_delta)

    history_entry = {
        "timestamp": fmt_ts(now),
        "delta": delta,
        "weighted_delta": weighted_delta,
        "note": note,
    }
    history: List[Dict] = area.setdefault("history", [])
    history.append(history_entry)
    if len(history) > HISTORY_LIMIT:
        del history[: len(history) - HISTORY_LIMIT]

    area["score"] = new_score
    area["last_update_ts"] = fmt_ts(now)
    area["last_decay_ts"] = fmt_ts(now)

    return original_score, new_score


def parse_quick_update(raw: str) -> Tuple[str, float]:
    if not raw.strip():
        raise ValueError("Update cannot be empty.")
    tokens = raw.strip().rsplit(" ", 1)
    if len(tokens) != 2:
        raise ValueError("Use the format 'Area +/-N'.")
    area_name, delta_str = tokens
    try:
        delta = float(delta_str)
    except ValueError as exc:
        raise ValueError("Momentum delta must be a number (e.g., +3 or -1.5).") from exc
    return area_name.strip(), delta


def format_score_bar(score: float, width: int = 20) -> str:
    filled = int(round((score / 100.0) * width))
    return "█" * filled + "░" * (width - filled)


def compute_trend(area: Dict, lookback: int = 5) -> float:
    history: List[Dict] = area.get("history", [])
    if not history:
        return 0.0
    recent = history[-lookback:]
    return mean(entry.get("delta", 0.0) for entry in recent)


def energy_state_from_trend(trend: float, threshold: float = 0.35) -> str:
    if trend >= threshold:
        return "Feeding"
    if trend <= -threshold:
        return "Draining"
    return "Stable"


def gather_snapshots(data: Dict, now: datetime) -> List[AreaSnapshot]:
    snapshots: List[AreaSnapshot] = []
    for name, area in data.get("areas", {}).items():
        apply_decay(area, now)
        score = area.get("score", 0.0)
        trend = compute_trend(area)
        energy_state = energy_state_from_trend(trend)
        last_update = parse_ts(area.get("last_update_ts"))
        snapshots.append(AreaSnapshot(name=name, score=score, trend=trend, energy_state=energy_state, last_update=last_update))
    snapshots.sort(key=lambda snap: snap.score, reverse=True)
    return snapshots


def humanize_timedelta(ts: Optional[datetime], now: datetime) -> str:
    if ts is None:
        return "never"
    delta = now - ts
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = months // 12
    return f"{years}y ago"


def generate_recalibration(snapshots: List[AreaSnapshot]) -> str:
    if not snapshots:
        return "Today, sketch a new momentum target to begin tracking momentum."
    average_score = mean(snap.score for snap in snapshots)
    # Choose area with lowest score weighted by negative trend.
    def priority(snap: AreaSnapshot) -> float:
        trend_penalty = -snap.trend * 10
        return snap.score + trend_penalty

    target = min(snapshots, key=priority)
    gap = average_score - target.score
    emphasis = "renew" if target.trend < 0 else "focus"
    percent = 10 if gap < 15 else 15 if gap < 30 else 20
    return f"Today, shift {percent}% more {emphasis} to {target.name} to rebalance momentum."


def generate_momentum_whisper(data: Dict) -> Optional[str]:
    histories: List[Dict] = []
    for area in data.get("areas", {}).values():
        histories.extend(area.get("history", []))
    if len(histories) < 8:
        return None
    buckets: Dict[int, List[float]] = {i: [] for i in range(7)}
    for entry in histories:
        ts = parse_ts(entry.get("timestamp"))
        if ts is None:
            continue
        buckets[ts.weekday()].append(entry.get("delta", 0.0))
    averages = {weekday: mean(values) for weekday, values in buckets.items() if values}
    if not averages:
        return None
    best_day, best_value = max(averages.items(), key=lambda item: item[1])
    worst_day, worst_value = min(averages.items(), key=lambda item: item[1])
    if best_value <= 0.25 and worst_value >= -0.25:
        return None
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][best_day]
    contrast = best_value - worst_value
    if contrast < 0.5:
        return None
    return f"Momentum whisperer: Every {day_name} your momentum spikes by about {best_value:+.1f}—plan high-impact moves there."


def render_status(data: Dict, now: datetime) -> str:
    snapshots = gather_snapshots(data, now)
    lines = []
    lines.append(f"Momentum Map — {now.strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("=" * 72)
    if not snapshots:
        lines.append("No areas tracked yet. Use the update command to log your first momentum entry.")
    for snap in snapshots:
        bar = format_score_bar(snap.score)
        trend_arrow = "↑" if snap.trend > 0.1 else "↓" if snap.trend < -0.1 else "→"
        lines.append(f"{snap.name:<16} {snap.score:6.1f} {bar}  {trend_arrow} {snap.trend:+.2f}  {snap.energy_state:<8}  last {humanize_timedelta(snap.last_update, now)}")
    lines.append("")
    lines.append(generate_recalibration(snapshots))
    whisper = generate_momentum_whisper(data)
    if whisper:
        lines.append(whisper)
    return "\n".join(lines)


def handle_update(args: argparse.Namespace) -> None:
    data = load_data()
    now = utcnow()
    try:
        area_name, delta = parse_quick_update(args.quick_update)
    except ValueError as exc:
        raise SystemExit(str(exc))
    original, updated = update_area(data, area_name, delta, args.note, now)
    save_data(data)
    change = updated - original
    direction = "increased" if change >= 0 else "decreased"
    print(f"{area_name} momentum {direction} to {updated:.1f} (Δ{change:+.1f}).")


def handle_status(_: argparse.Namespace) -> None:
    data = load_data()
    now = utcnow()
    apply_decay_all(data, now)
    save_data(data)
    print(render_status(data, now))


def handle_reset(args: argparse.Namespace) -> None:
    if DATA_FILE.exists() and not args.force:
        raise SystemExit("Use --force to confirm reset. This will erase all momentum data.")
    save_data({"areas": {}})
    print("Momentum data reset.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Momentum Map CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="Log a quick momentum update")
    update_parser.add_argument("quick_update", help="Update string, e.g., 'ProjectX +3'")
    update_parser.add_argument("--note", help="Optional short note to capture context.")
    update_parser.set_defaults(func=handle_update)

    status_parser = subparsers.add_parser("status", help="View the current momentum map")
    status_parser.set_defaults(func=handle_status)

    reset_parser = subparsers.add_parser("reset", help="Erase all stored momentum data")
    reset_parser.add_argument("--force", action="store_true", help="Confirm the reset")
    reset_parser.set_defaults(func=handle_reset)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
