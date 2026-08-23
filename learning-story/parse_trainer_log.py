"""Parse tmrl trainer stdout into round-level metrics.

tmrl prints one header per round, then a pandas Series. The header width
changes with the epoch number (``str.ljust``), so naive regexes miss a
band of epochs. This parser accepts every layout tmrl 0.7.x emits:

    === epoch 0/10000 = round 0/10 ========================================
    === epoch 10/10000  round 0/10 ========================================
    === epoch 100/10000  round 0/10 =======================================
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HEADER_RE = re.compile(
    r"=== epoch\s+(\d+)/(\d+)\s*=*\s*round\s+(\d+)/(\d+)",
    re.IGNORECASE,
)
METRIC_RE = re.compile(
    r"^\s*(memory_len|return_train|return_test|episode_length_train|"
    r"episode_length_test|loss_actor|loss_critic|round_time|"
    r"sampling_duration|training_step_duration|idle_time)\s+([-\d.eE+]+)\s*$"
)


def parse_trainer_log(text: str) -> list[dict[str, Any]]:
    """Return one dict per completed trainer round, in log order."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and "return_train" in current:
            rows.append(current)
        current = None

    for line in text.splitlines():
        header = HEADER_RE.search(line)
        if header:
            flush()
            current = {
                "epoch": int(header.group(1)),
                "max_epochs": int(header.group(2)),
                "round": int(header.group(3)),
                "rounds_per_epoch": int(header.group(4)),
            }
            continue
        if current is None:
            continue
        metric = METRIC_RE.match(line)
        if metric:
            current[metric.group(1)] = float(metric.group(2))
    flush()
    return rows


def parse_trainer_log_file(path: str | Path) -> list[dict[str, Any]]:
    return parse_trainer_log(Path(path).read_text(encoding="utf-8", errors="replace"))


def missing_epochs(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    seen = {int(r["epoch"]) for r in rows}
    return [e for e in range(max(seen) + 1) if e not in seen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Trainer stdout / terminal capture")
    parser.add_argument("-o", "--output", type=Path, help="Write JSON here")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    rows = parse_trainer_log_file(args.log)
    payload = {
        "run_name": args.run_name,
        "source": str(args.log),
        "missing_epochs": missing_epochs(rows),
        "rounds": rows,
    }
    text = json.dumps(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {len(rows)} rounds -> {args.output}")
    else:
        print(text)
    print(
        f"epochs {min((r['epoch'] for r in rows), default=None)}-"
        f"{max((r['epoch'] for r in rows), default=None)}, "
        f"missing {len(payload['missing_epochs'])}"
    )


if __name__ == "__main__":
    main()
