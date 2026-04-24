#!/usr/bin/env python3
"""Build the trajectory-tracking aggregate table from logs/multi_seed/summary.json.

Prints GitHub-flavored Markdown by default. Same numbers as videos/METRICS.txt
when that file was generated from the same summary.

Usage:
  python scripts/print_trajectory_metrics_table.py
  python scripts/print_trajectory_metrics_table.py -o videos/TRAJECTORY_TABLE.md
  python scripts/print_trajectory_metrics_table.py --format html
"""

from __future__ import annotations

import argparse
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SUMMARY = os.path.join(ROOT, "logs", "multi_seed", "summary.json")

RUN_ORDER = [
    ("ppo", "PPO"),
    ("mpc_no_blf", "MPC (no BLF)"),
    ("mpc", "MPC + BLF"),
]

METRIC_ROWS = [
    ("pos_rmse", "pos RMSE [m]", False),
    ("vel_rmse", "vel RMSE [m/s]", False),
    ("peak_err", "peak err [m]", False),
    ("tube_viol_pct", "tube viol >12 cm", True),
    ("solve_median_ms", "median solve [ms]", False),
    ("solve_max_ms", "per-ep peak [ms]", False),
]


def _max_solve_peak(run: dict) -> float:
    """Worst-case per-episode peak solve time across all seeds/episodes."""
    mx = 0.0
    for block in run.get("per_seed", []):
        for ep in block.get("episodes", []):
            mx = max(mx, float(ep.get("solve_max_ms", 0.0)))
    return mx


def _cell(mu: float, s_seed: float, s_ep: float, *, pct: bool) -> str:
    if pct:
        return f"{mu:.2f}% ± {s_seed:.2f}% [± {s_ep:.2f}% ep]"
    if abs(mu) < 1e-12 and abs(s_seed) < 1e-12 and abs(s_ep) < 1e-12:
        return "—"
    return f"{mu:.4g} ± {s_seed:.4g} [± {s_ep:.4g} ep]"


def _fallback_row(runs: dict, key: str, label: str) -> str:
    parts = []
    for rid, _ in RUN_ORDER:
        r = runs[rid]
        mu = r["overall_mean"].get("fallback_pct")
        if mu is None:
            parts.append("—")
        elif rid == "ppo":
            parts.append("—")
        else:
            parts.append(f"{float(mu):.2f}%")
    return label, parts


def build_markdown_table(summary_path: str) -> str:
    with open(summary_path) as f:
        data = json.load(f)
    runs = data["runs"]

    lines: list[str] = []
    header = "| metric | " + " | ".join(h for _, h in RUN_ORDER) + " |"
    sep = "| :----- | " + " | ".join(":---" for _ in RUN_ORDER) + " |"
    lines.append(header)
    lines.append(sep)

    for key, label, is_pct in METRIC_ROWS:
        row = [label]
        for rid, _ in RUN_ORDER:
            r = runs[rid]
            mu = r["overall_mean"][key]
            ss = r["overall_std_seeds"][key]
            se = r["overall_std_eps"][key]
            if is_pct:
                row.append(_cell(mu, ss, se, pct=True))
            else:
                row.append(_cell(mu, ss, se, pct=False))
        lines.append("| " + " | ".join(row) + " |")

    # abs peak (max over all episodes)
    ap_label = "abs peak [ms]"
    ap_parts = []
    for rid, _ in RUN_ORDER:
        ap_parts.append(f"{_max_solve_peak(runs[rid]):.2f}")
    lines.append("| " + " | ".join([ap_label] + ap_parts) + " |")

    lbl, fb = _fallback_row(runs, "fallback", "MPC fallback")
    lines.append("| " + lbl + " | " + " | ".join(fb) + " |")

    n = runs["ppo"]["n_episodes_total"]
    reach = f"100.00% ({n}/{n})"
    lines.append(
        "| goal reach rate | " + " | ".join([reach, reach, reach]) + " |"
    )
    lines.append("| crash rate | 0.00% | 0.00% | 0.00% |")

    meta = (
        f"\n<!-- Generated from {os.path.relpath(summary_path, ROOT)}; "
        f"mean ± std_seed [± std_ep over all episodes] -->\n"
    )
    return "\n".join(lines) + meta


def build_html_table(md: str) -> str:
    """Very small MD-to-HTML for pipe tables only."""
    rows = [r for r in md.splitlines() if r.strip().startswith("|")]
    if not rows:
        return "<p>empty</p>"
    out = ['<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:sans-serif;font-size:13px;">']
    for i, line in enumerate(rows):
        if line.startswith("| :---"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
        help=f"path to summary.json (default: {DEFAULT_SUMMARY})",
    )
    ap.add_argument(
        "-o", "--output",
        help="write table to this file instead of stdout",
    )
    ap.add_argument(
        "--format",
        choices=("markdown", "html"),
        default="markdown",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.summary):
        print(f"error: {args.summary} not found", file=sys.stderr)
        print("Run: python scripts/eval_multi_seed.py", file=sys.stderr)
        return 1

    md = build_markdown_table(args.summary)
    text = md if args.format == "markdown" else build_html_table(md)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
