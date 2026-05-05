"""dashboard.py — live TUI for the autoresearch loop.

Shows:
  - Current best run (full config)
  - Last 15 experiments as a table sorted newest-first
  - If a run is in progress: live elapsed time + tail of its output
  - Per-FOV ARI breakdown for the current best

Run with:
  .venv/bin/python phase2/autoresearch/dashboard.py
  .venv/bin/python phase2/autoresearch/dashboard.py --top 25     # show more rows
  .venv/bin/python phase2/autoresearch/dashboard.py --refresh 5  # slower refresh

Press Ctrl-C to exit. Read-only — never modifies anything.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[2]
AUTORESEARCH = Path(__file__).resolve().parent
RESULTS_DIR = AUTORESEARCH / "results"
RESULTS_LOG = AUTORESEARCH / "results.md"

LINE_RE = re.compile(
    r"-\s*\[(?P<flag>.)\]\s+(?P<id>\S+)\s+ARI=(?P<ari>[\d.]+)\s+"
    r"\((?P<val>FAST|FULL|fast|full),\s*(?P<seconds>[\d.]+)s\)\s+(?P<cfg>.*)$"
)


def parse_log() -> list[dict]:
    if not RESULTS_LOG.exists():
        return []
    rows = []
    for line in RESULTS_LOG.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "improved": d["flag"].strip() == "*",
            "id":  d["id"],
            "ari": float(d["ari"]),
            "val": d["val"].lower(),
            "secs": float(d["seconds"]),
            "cfg": d["cfg"],
        })
    return rows


def is_run_in_progress() -> dict | None:
    """Return dict with running run's pid+start if any active run_experiment.py."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "run_experiment.py"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    for line in out.strip().splitlines():
        if "run_experiment.py" not in line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        pid = parts[0]
        # Read the process start time
        try:
            etime = subprocess.check_output(
                ["ps", "-o", "etime=", "-p", pid], text=True
            ).strip()
        except subprocess.CalledProcessError:
            continue
        return {"pid": pid, "etime": etime, "cmd": parts[1] if len(parts) > 1 else ""}
    return None


def tail_running_log(pid: str | None) -> list[str]:
    """Best-effort tail of any active claude tmp output file."""
    candidates = sorted(Path("/private/tmp").rglob("*.output"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0,
                        reverse=True)[:5]
    for c in candidates:
        try:
            txt = c.read_text(errors="ignore")
        except Exception:
            continue
        if "run_experiment" in txt and "MEAN ARI" not in txt:
            return txt.splitlines()[-6:]
    return []


def best_run(rows: list[dict]) -> dict | None:
    full_rows = [r for r in rows if r["val"] == "full"]
    pool = full_rows if full_rows else rows
    return max(pool, key=lambda r: r["ari"]) if pool else None


def _fmt_cfg(cfg: str) -> Text:
    """Highlight key=val pairs."""
    t = Text()
    for tok in cfg.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            t.append(k + "=", style="dim cyan")
            t.append(v + " ", style="bright_white")
        else:
            t.append(tok + " ", style="dim")
    return t


def _summary_panel(rows: list[dict], running: dict | None) -> Panel:
    best = best_run(rows)
    n_runs = len(rows)
    n_full = sum(1 for r in rows if r["val"] == "full")
    n_imp = sum(1 for r in rows if r["improved"])

    text = Text()
    if best:
        text.append("Best ARI: ", style="dim")
        text.append(f"{best['ari']:.4f}", style="bold bright_green")
        text.append(f"  ({best['val'].upper()}, ", style="dim")
        text.append(best["id"], style="bright_white")
        text.append(")\n", style="dim")
        text.append("config:   ", style="dim")
        text.append(_fmt_cfg(best["cfg"]))
        text.append("\n")
    else:
        text.append("No runs yet.\n", style="dim italic")

    text.append("\n")
    text.append(f"Total iterations: {n_runs}", style="dim")
    text.append(f"   FULL: {n_full}", style="dim")
    text.append(f"   improvements: {n_imp}\n", style="dim")

    if running:
        text.append("\n")
        text.append("⚡ RUNNING ", style="bold yellow on black")
        text.append(f"  pid={running['pid']}  elapsed={running['etime']}\n",
                    style="dim yellow")
    return Panel(text, title="[bold]autoresearch[/bold]", border_style="green")


def _runs_table(rows: list[dict], top: int) -> Panel:
    table = Table(expand=True, show_lines=False, header_style="bold cyan",
                  border_style="dim")
    table.add_column("#", justify="right", width=4, style="dim")
    table.add_column("id",  style="dim", width=15)
    table.add_column("ARI", justify="right", width=8)
    table.add_column("Δ",   justify="right", width=8)
    table.add_column("val", width=4, style="dim")
    table.add_column("dur", justify="right", width=5, style="dim")
    table.add_column("config", overflow="ellipsis", no_wrap=True)

    full_rows = [r for r in rows if r["val"] == "full"]
    pool = full_rows if full_rows else rows
    best_ari = max((r["ari"] for r in pool), default=0.0)

    recent = list(reversed(rows[-top:]))
    for i, r in enumerate(recent):
        delta = r["ari"] - best_ari
        if r["improved"]:
            ari_style = "bold bright_green"
            id_style = "bold bright_white"
        elif delta >= -0.005:
            ari_style = "bright_yellow"
            id_style = "white"
        else:
            ari_style = "dim"
            id_style = "dim"

        delta_str = f"{delta:+.3f}"
        delta_style = "green" if delta > 0 else ("red" if delta < -0.01 else "dim")
        table.add_row(
            f"{len(rows) - i}",
            Text(r["id"], style=id_style),
            Text(f"{r['ari']:.4f}", style=ari_style),
            Text(delta_str, style=delta_style),
            Text(r["val"].upper(), style="cyan" if r["val"] == "full" else "dim"),
            f"{r['secs']:.0f}s",
            _fmt_cfg(r["cfg"]),
        )
    return Panel(table, title=f"[bold]Last {len(recent)} iterations (newest first)[/bold]",
                 border_style="cyan")


def _per_fov_panel(rows: list[dict]) -> Panel | None:
    best = best_run(rows)
    if not best:
        return None
    json_path = RESULTS_DIR / f"{best['id']}.json"
    if not json_path.exists():
        return None
    try:
        result = json.loads(json_path.read_text())
    except Exception:
        return None
    table = Table(expand=True, show_lines=False, header_style="bold magenta",
                  border_style="dim")
    table.add_column("FOV", style="dim")
    table.add_column("class", justify="right")
    table.add_column("subclass", justify="right")
    table.add_column("supertype", justify="right")
    table.add_column("cluster", justify="right")
    table.add_column("frac_pred", justify="right", style="dim")
    table.add_column("frac_gt",   justify="right", style="dim")

    fovs = result.get("per_fov", {})
    for fov in sorted(fovs):
        m = fovs[fov]
        per_lvl = m.get("per_level", {})
        def cell(v): return Text(f"{v:+.3f}",
                                  style="green" if v > 0.5 else "yellow" if v > 0.3 else "red")
        table.add_row(
            fov,
            cell(per_lvl.get("class", 0)),
            cell(per_lvl.get("subclass", 0)),
            cell(per_lvl.get("supertype", 0)),
            cell(per_lvl.get("cluster", 0)),
            f"{m.get('frac_in_cell_pred', 0):.3f}",
            f"{m.get('frac_in_cell_gt', 0):.3f}",
        )
    return Panel(table,
                 title=f"[bold]Per-FOV ARI for best run ({best['id']})[/bold]",
                 border_style="magenta")


def _running_panel(running: dict | None) -> Panel | None:
    if not running:
        return None
    tail = tail_running_log(running["pid"])
    body = Text()
    body.append(f"pid={running['pid']}  elapsed={running['etime']}\n",
                style="bright_yellow")
    if tail:
        body.append("\nrecent output:\n", style="dim")
        for line in tail:
            body.append(line + "\n", style="dim")
    else:
        body.append("\n(no log output captured yet)\n", style="dim italic")
    return Panel(body, title="[bold yellow]⚡ Active iteration[/bold yellow]",
                 border_style="yellow")


def render(top: int) -> Layout:
    rows = parse_log()
    running = is_run_in_progress()

    summary = _summary_panel(rows, running)
    runs    = _runs_table(rows, top)
    rp      = _running_panel(running) if running else None
    pf      = _per_fov_panel(rows)

    layout = Layout()
    if rp and pf:
        bottom = Layout(name="bottom")
        bottom.split_row(Layout(rp), Layout(pf))
        layout.split_column(
            Layout(summary, size=8),
            Layout(runs),
            Layout(bottom, size=14),
        )
    elif rp or pf:
        layout.split_column(
            Layout(summary, size=8),
            Layout(runs),
            Layout(rp or pf, size=14),
        )
    else:
        layout.split_column(
            Layout(summary, size=8),
            Layout(runs),
        )
    return layout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", type=float, default=2.0,
                   help="Refresh interval in seconds (default 2).")
    p.add_argument("--top", type=int, default=15,
                   help="Number of recent iterations to show (default 15).")
    args = p.parse_args()

    console = Console()
    try:
        with Live(render(args.top), console=console, refresh_per_second=4,
                  screen=True) as live:
            while True:
                time.sleep(args.refresh)
                live.update(render(args.top))
    except KeyboardInterrupt:
        console.print("[dim]exiting dashboard[/dim]")


if __name__ == "__main__":
    main()
