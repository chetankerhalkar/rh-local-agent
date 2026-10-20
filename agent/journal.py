"""Audit journal (append-only JSONL) and per-day state (JSON)."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .config import ROOT
from .models import DayState

ECHO = True     # print every event too. The interactive daily sessions switch this off and talk in plain words


def log(event: str, **data) -> None:
    d = ROOT / "journal"
    d.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    rec = {"ts": now.isoformat(timespec="seconds"), "event": event, **data}
    with open(d / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    if ECHO:
        print(f"[{now:%H:%M:%S}Z] {event}: {json.dumps(data, default=str)[:300]}")


def load_day(date: str) -> DayState:
    p = ROOT / "state" / f"{date}.json"
    if p.exists():
        return DayState(**json.loads(p.read_text(encoding="utf-8")))
    return DayState(date=date)


def save_day(day: DayState) -> None:
    d = ROOT / "state"
    d.mkdir(exist_ok=True)
    tmp = d / f"{day.date}.json.tmp"
    tmp.write_text(json.dumps(asdict(day), indent=2), encoding="utf-8")
    tmp.replace(d / f"{day.date}.json")     # atomic: a crash never leaves half a file
