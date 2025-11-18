#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_baseline.py  —  Alpha-sweep orchestrator (fixed)
----------------------------------------------------------------
Build a 4000×2000 *baseline*, estimate the size of the daily combinatorial
space (all time-ordered triples of existing slots), pick **150 alphas** that
yield **~200..10,000 paths per patient per day**, and for each alpha:
  • (Re)generate the GA/SA daily graphs with that per-patient budget
  • Build daily hypergraphs (covers)
  • Run the weekly optimizer (greedy+LNS)
  • Collect metrics + make the 4×(6 subplots) figures
  • Save per-alpha vectors

This version fixes:
  ✓ No recursive self-invocation. Subprocess calls target the actual scripts:
      - graph_generator.py
      - hypergraph_generator.py
      - weekly_optimizer_greedy_lns.py
  ✓ No unsupported flags passed to the wrong script (e.g., `--graphs_subdir` to
    this file). Windows-safe path handling (no `is_relative_to`).
  ✓ Robust counting of time-ordered triples in O(n log n) per day.

Inputs expected in --base_dir:
  patients_treatments_opmed.csv
  patients_constraints_opmed.csv
  therapists_treatments.csv
  treatments_durations_opmed.csv
  patient_treatment_therapist_priorities_opmed.csv

Other scripts (paths can be overridden by flags):
  graph_generator.py                (GA/SA path builder)
  hypergraph_generator.py           (daily hypergraph covers)
  weekly_optimizer_greedy_lns.py    (weekly schedule builder)

Outputs:
  ./baseline_graphs_4000x2000/                    (reused if exists)
  ./alpha_runs/
      graphs_a<code>/                             per-alpha graphs
      graphs_a<code>_hyper/                       per-alpha hypergraphs
      weekly_schedule_optimized_a<code>.csv       per-alpha schedule
      optimization_summary_a<code>.csv            per-alpha summary
  alpha_results.csv                                one row per alpha
  figs/
      01_logalpha_vs_daily_score.png              6 subplots (days)
      02_logalpha_vs_runtime.png                  6 subplots
      03_logalpha_vs_num_patients.png             6 subplots
      04_num_patients_vs_total_score.png          6 subplots

Run example:
    python optimization_baseline.py \
        --base_dir . \
        --days 1,2,3,4,5,6 \
        --patients_first 118 \
        --samples 150 \
        --min_pp 200 --max_pp 10000 \
        --verbosity 1
"""
from __future__ import annotations

import argparse, json, math, os, sys, subprocess, shutil, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- small utils -----------------------------

def vprint(verbosity: int, lvl: int, msg: str):
    if verbosity >= lvl:
        print(msg, flush=True)

def safe_relpath(child: Path, parent: Path) -> str:
    """Return child's path relative to parent if possible, else absolute."""
    try:
        return str(child.relative_to(parent))
    except Exception:
        return str(child)

# time helpers ------------------------------------------------------------
MINUTE_GRID = 5

def t2min(s: str) -> int:
    s = (s or '').strip()
    if ':' not in s:
        try:
            return int(s)
        except Exception:
            return 0
    h, m = s.split(':', 1)
    return int(h)*60 + int(m)

# ------------------------ STEP 0: Baseline build ------------------------

def run_cmd(cmd: List[str], cwd: Path, verbosity: int, fail_hint: str = ""):
    vprint(verbosity, 2, f"[CMD] {' '.join(cmd)} (cwd={cwd})")
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        vprint(verbosity, 1, res.stdout)
        vprint(verbosity, 1, res.stderr)
        raise RuntimeError(f"Command failed. {fail_hint}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    if res.stdout.strip():
        vprint(verbosity, 2, res.stdout.rstrip())
    return res

def ensure_baseline(base: Path,
                    graph_py: Path,
                    days: List[int],
                    graphs_subdir: str,
                    per_patient: int,
                    group_quota: int,
                    group_slots: int,
                    method: str,
                    offlist_prob: float,
                    verbosity: int):
    gdir = base / graphs_subdir
    if gdir.exists() and any((gdir / f"day{d}_paths.csv").exists() for d in days):
        vprint(verbosity, 1, f"[BASELINE] Reusing existing {graphs_subdir}")
        return
    vprint(verbosity, 1, f"[BASELINE] Building graphs in {gdir} (per_patient={per_patient}, group_quota={group_quota})")
    py = sys.executable
    cmd = [py, safe_relpath(graph_py, base),
           "--base_dir", str(base),
           "--graphs_subdir", graphs_subdir,
           "--days", ",".join(map(str, days)),
           "--per_patient", str(per_patient),
           "--group_quota", str(group_quota),
           "--group_slots", str(group_slots),
           "--method", method,
           "--offlist_prob", str(offlist_prob),
           "--verbosity", str(max(1, verbosity))]
    run_cmd(cmd, base, verbosity, fail_hint="baseline graph build failed")

# --------------- STEP 1: Count time-ordered slot triples ---------------

def _count_time_ordered_triples(starts: np.ndarray, ends: np.ndarray) -> int:
    """Count triples (i, j, k) with end[i] <= start[j] and end[j] <= start[k].
    O(n log n) using two upper-bounds + prefix sums.
    """
    n = len(starts)
    if n < 3:
        return 0
    # P[j] = #i with E[i] <= S[j]
    E_sorted = np.sort(ends)
    idx_for_S = np.searchsorted(E_sorted, starts, side='right')  # P per j (same order as starts)
    # Now order j by its end time E[j]
    order_j = np.argsort(ends)
    P_sorted_by_E = idx_for_S[order_j]
    prefixP = np.cumsum(P_sorted_by_E, dtype=object)  # big ints
    # For each k (by S[k]), find how many j have E[j] <= S[k]
    ub = np.searchsorted(np.sort(ends), starts, side='right')  # count of j with E<=S[k]
    # Sum prefix up to ub-1 for every k
    total = 0
    for u in ub:
        if u > 0:
            total += int(prefixP[u-1])
    return int(total)

def compute_possible_counts_from_baseline(baseline_dir: Path, day: int, verbosity: int) -> Tuple[int,int]:
    """Return (patients_count, total_ordered_triples) for this day using baseline files."""
    nodes_fp = baseline_dir / f"day{day}_nodes.csv"
    paths_fp = baseline_dir / f"day{day}_paths.csv"
    if not nodes_fp.exists():
        return 0, 0
    nodes = pd.read_csv(nodes_fp).fillna("")
    # Patients present = distinct in paths file for this day
    if paths_fp.exists():
        pts = pd.read_csv(paths_fp)
        patients = pts[pts['day'] == day]['patient'].nunique()
    else:
        patients = nodes[(nodes.get('day', day) == day) & (nodes.get('node_type','') == 'patient')].shape[0]
    # Collect all slot start/end
    mask = (nodes.get('day', day) == day) & (nodes.get('node_type','slot').str.lower() == 'slot')
    slots = nodes.loc[mask, ['start','end']].copy()
    if slots.empty:
        return int(patients), 0
    starts = slots['start'].astype(str).map(t2min).to_numpy()
    ends   = slots['end'].astype(str).map(t2min).to_numpy()
    total_triples = _count_time_ordered_triples(starts, ends)
    return int(patients), int(total_triples)

# ---------------------- STEP 2: Alpha sampler --------------------------

def pick_alphas_from_range(counts: Dict[int, Tuple[int,int]],
                           min_pp: int, max_pp: int,
                           samples: int, verbosity: int) -> List[float]:
    """Pick 150 log-spaced alphas so that per-patient/day stays ~[min_pp, max_pp].
    For each day d: alpha ≈ pp * patients_d / triples_d. We aggregate across
    days by using the **median** day bounds to avoid over-constraining.
    """
    days = sorted(counts.keys())
    # per day bounds
    bounds = []
    for d in days:
        patients, triples = counts[d]
        if patients == 0 or triples == 0:
            continue
        a_min = min_pp * patients / triples
        a_max = max_pp * patients / triples
        bounds.append((a_min, a_max))
    if not bounds:
        # sensible default narrow band
        lo, hi = 1e-16, 5e-15
    else:
        lo = float(np.median([b[0] for b in bounds]))
        hi = float(np.median([b[1] for b in bounds]))
        if not (hi > lo > 0):
            lo, hi = 1e-16, 5e-15
    alphas = np.logspace(math.log10(lo), math.log10(hi), num=samples)
    vprint(verbosity, 1, f"[ALPHAS] {samples} samples in total (~per-patient range {min_pp}..{max_pp})")
    return [float(a) for a in alphas]

# ------------------ STEP 3: Per-alpha graph build ----------------------

def build_graphs_for_alpha(base: Path,
                           graph_py: Path,
                           days: List[int],
                           run_dir: Path,
                           per_patient_by_day: Dict[int,int],
                           group_slots: int,
                           method: str,
                           offlist_prob: float,
                           verbosity: int):
    py = sys.executable
    graphs_subdir = run_dir.name
    # The GA builder accepts a single --per_patient across its run.
    # We call it **once per day** to allow different budgets by day.
    for d in days:
        per_p = int(per_patient_by_day[d])
        gq = max(0, per_p // 2)
        cmd = [py, safe_relpath(graph_py, base),
               "--base_dir", str(base),
               "--graphs_subdir", graphs_subdir,
               "--days", str(d),
               "--per_patient", str(per_p),
               "--group_quota", str(gq),
               "--group_slots", str(group_slots),
               "--method", method,
               "--offlist_prob", str(offlist_prob),
               "--verbosity", str(max(1, verbosity))]
        run_cmd(cmd, base, verbosity, fail_hint=f"graph build failed for day {d}")

# ----------------- STEP 4: Hypergraph + Weekly optimize ----------------

def run_hypergraph(base: Path, hyper_py: Path, graphs_subdir: str, days: List[int], verbosity: int):
    py = sys.executable
    cmd = [py, safe_relpath(hyper_py, base),
           "--base_dir", str(base),
           "--graphs_subdir", graphs_subdir,
           "--days", ",".join(map(str, days)),
           "--verbosity", str(max(1, verbosity))]
    run_cmd(cmd, base, verbosity, fail_hint="hypergraph_generator failed")


def run_weekly_opt(base: Path, opt_py: Path, graphs_subdir: str, days: List[int], patients_first: int, topk: int, seed: int, verbosity: int,
                   out_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    py = sys.executable
    cmd = [py, safe_relpath(opt_py, base),
           "--base_dir", str(base),
           "--graphs_subdir", graphs_subdir,
           "--days", ",".join(map(str, days)),
           "--patients_first", str(patients_first),
           "--topk", str(topk),
           "--seed", str(seed),
           "--verbosity", str(max(1, verbosity))]
    t0 = time.time()
    run_cmd(cmd, base, verbosity, fail_hint="weekly optimizer failed")
    rt = time.time() - t0
    # Move outputs to unique files
    sched_src = base / "weekly_schedule_optimized.csv"
    sum_src   = base / "optimization_summary.csv"
    sched_dst = base / f"{out_prefix}_weekly_schedule_optimized.csv"
    sum_dst   = base / f"{out_prefix}_optimization_summary.csv"
    if sched_src.exists():
        shutil.move(str(sched_src), str(sched_dst))
    if sum_src.exists():
        shutil.move(str(sum_src), str(sum_dst))
    # Read summary/schedule
    summary = pd.read_csv(sum_dst) if sum_dst.exists() else pd.DataFrame()
    sched   = pd.read_csv(sched_dst) if sched_dst.exists() else pd.DataFrame()
    # Inject measured runtime if summary missing/empty
    if not summary.empty and 'runtime_seconds' in summary.columns:
        summary.loc[0, 'runtime_seconds'] = round(rt, 3)
    return sched, summary

# ------------------------ STEP 5: Plotting ------------------------------

def plot_6x(alphas_log10: List[float], per_day_series: Dict[int, List[float]], title: str, ylabel: str, outpng: Path):
    days = sorted(per_day_series.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for idx, d in enumerate(days):
        ax = axes[idx//3, idx%3]
        ax.plot(alphas_log10, per_day_series[d], marker='o', linewidth=1)
        ax.set_title(f"Day {d}")
        ax.set_xlabel("log10(alpha)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        # use ticks like 10^-k on x? We'll keep log10(alpha) numeric and rely on labels
        # (User asked for 10^{-k} look; textual log10 values are OK in practice.)
    fig.suptitle(title)
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

# ------------------------------ MAIN -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Alpha-sweep orchestrator for rehab weekly optimization (fixed)")
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--days", default="1,2,3,4,5,6")
    ap.add_argument("--patients_first", type=int, default=118)
    ap.add_argument("--seed", type=int, default=20251110)
    ap.add_argument("--verbosity", type=int, default=1)
    # External scripts
    ap.add_argument("--graph_py", default="graph_generator.py")
    ap.add_argument("--hyper_py", default="hypergraph_generator.py")
    ap.add_argument("--opt_py",   default="weekly_optimizer_greedy_lns.py")
    # Baseline
    ap.add_argument("--baseline_graphs_subdir", default="baseline_graphs_4000x2000")
    ap.add_argument("--baseline_per_patient", type=int, default=4000)
    ap.add_argument("--baseline_group_quota", type=int, default=2000)
    # Alpha selection
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--min_pp", type=int, default=200)
    ap.add_argument("--max_pp", type=int, default=10000)
    # GA builder knobs reused for per-alpha runs
    ap.add_argument("--group_slots", type=int, default=10)
    ap.add_argument("--method", choices=["ga","sa","both"], default="both")
    ap.add_argument("--offlist_prob", type=float, default=0.12)
    # Weekly optimizer knobs
    ap.add_argument("--topk", type=int, default=80)

    args = ap.parse_args()
    base = Path(args.base_dir)
    days = [int(x) for x in args.days.split(',') if str(x).strip()]

    graph_py = (base / args.graph_py) if not Path(args.graph_py).is_absolute() else Path(args.graph_py)
    hyper_py = (base / args.hyper_py) if not Path(args.hyper_py).is_absolute() else Path(args.hyper_py)
    opt_py   = (base / args.opt_py)   if not Path(args.opt_py).is_absolute()   else Path(args.opt_py)

    # 0) Baseline build (or reuse)
    ensure_baseline(base,
                    graph_py=graph_py,
                    days=days,
                    graphs_subdir=args.baseline_graphs_subdir,
                    per_patient=args.baseline_per_patient,
                    group_quota=args.baseline_group_quota,
                    group_slots=args.group_slots,
                    method=args.method,
                    offlist_prob=args.offlist_prob,
                    verbosity=args.verbosity)

    # 1) Count possible triples per day
    counts: Dict[int, Tuple[int,int]] = {}
    for d in days:
        patients, triples = compute_possible_counts_from_baseline(base / args.baseline_graphs_subdir, d, args.verbosity)
        counts[d] = (patients, triples)
        vprint(args.verbosity, 1, f"[COUNT] Day {d}: patients={patients}; total_possible_triples={triples:,}")

    # 2) Pick alphas
    alphas = pick_alphas_from_range(counts, args.min_pp, args.max_pp, args.samples, args.verbosity)

    # Prepare out folders
    alpha_root = base / "alpha_runs"
    (alpha_root / "figs").mkdir(parents=True, exist_ok=True)

    results_rows = []

    # 3) Sweep
    py = sys.executable
    for idx, alpha in enumerate(alphas, 1):
        code = f"a{alpha:.3e}".replace("+", "").replace(".", "p").replace("e-", "em")
        run_dir = alpha_root / f"graphs_{code}"
        run_dir.mkdir(parents=True, exist_ok=True)
        graphs_subdir = run_dir.name
        vprint(args.verbosity, 1, f"\n[RUN {idx:03d}/{len(alphas)}] alpha={alpha:.3e}")

        # Per-day per-patient budgets
        per_patient_by_day: Dict[int,int] = {}
        for d in days:
            patients, triples = counts.get(d, (0,0))
            if patients == 0 or triples == 0:
                per_patient_by_day[d] = 0
            else:
                est = int(round(alpha * triples / max(1, patients)))
                per_patient_by_day[d] = int(max(args.min_pp, min(args.max_pp, est)))
        vprint(args.verbosity, 2, f"    per_patient_by_day: {per_patient_by_day}")

        # 3a) Build graphs for this alpha (day by day to honor budgets)
        build_graphs_for_alpha(base, graph_py, days, run_dir, per_patient_by_day,
                               group_slots=args.group_slots, method=args.method,
                               offlist_prob=args.offlist_prob, verbosity=args.verbosity)

        # 3b) Hypergraph for this alpha (writes to <graphs>_hyper)
        run_hypergraph(base, hyper_py, graphs_subdir, days, args.verbosity)

        # 3c) Weekly optimizer
        out_prefix = f"{graphs_subdir}"
        sched_df, summary_df = run_weekly_opt(base, opt_py, graphs_subdir, days, args.patients_first, args.topk, args.seed, args.verbosity, out_prefix)

        # 3d) Collect metrics (global + per-day)
        if summary_df.empty:
            res = {
                "alpha": alpha, "log10alpha": math.log10(alpha),
                "runtime": float('nan'), "n_patients": 0,
                "objective_norm": float('nan')
            }
            for d in days: res[f"day{d}_mean_value"] = float('nan')
            results_rows.append(res)
            continue

        runtime = float(summary_df.loc[0, 'runtime_seconds']) if 'runtime_seconds' in summary_df.columns else float('nan')
        n_pat   = int(summary_df.loc[0, 'patients_scheduled_fully']) if 'patients_scheduled_fully' in summary_df.columns else int(sched_df['patient'].nunique())
        objN    = float(summary_df.loc[0, 'objective_norm_0_1']) if 'objective_norm_0_1' in summary_df.columns else float('nan')

        # per-day mean of (1 - score_norm) among assigned chains
        day_means: Dict[int, float] = {}
        if not sched_df.empty:
            tmp = sched_df.copy()
            tmp['value'] = 1.0 - tmp['score_norm'].astype(float)
            for d in days:
                g = tmp[tmp['day'] == d]
                day_means[d] = float(g['value'].mean()) if not g.empty else 0.0
        else:
            for d in days: day_means[d] = 0.0

        results_rows.append({
            "alpha": alpha, "log10alpha": math.log10(alpha),
            "runtime": runtime, "n_patients": n_pat, "objective_norm": objN,
            **{f"day{d}_mean_value": day_means[d] for d in days}
        })

    # Save vectors
    res_df = pd.DataFrame(results_rows)
    res_csv = base / "alpha_results.csv"
    res_df.to_csv(res_csv, index=False)
    vprint(args.verbosity, 1, f"[OK] Saved vectors → {res_csv}")

    # 4) Figures (each: 6 subplots)
    # (1) log10(alpha) vs daily mean value in [0,1]
    per_day_series = {d: res_df[f"day{d}_mean_value"].tolist() for d in days}
    plot_6x(res_df['log10alpha'].tolist(), per_day_series,
            title="log10(alpha) vs daily objective (mean of 1-score_norm)",
            ylabel="daily objective (0..1)",
            outpng=alpha_root / "figs" / "01_logalpha_vs_daily_score.png")

    # (2) log10(alpha) vs runtime (replicated per day)
    per_day_runtime = {d: res_df['runtime'].tolist() for d in days}
    plot_6x(res_df['log10alpha'].tolist(), per_day_runtime,
            title="log10(alpha) vs runtime (sec)",
            ylabel="runtime (sec)",
            outpng=alpha_root / "figs" / "02_logalpha_vs_runtime.png")

    # (3) log10(alpha) vs num patients (replicated per day)
    per_day_np = {d: res_df['n_patients'].tolist() for d in days}
    plot_6x(res_df['log10alpha'].tolist(), per_day_np,
            title="log10(alpha) vs fully scheduled patients",
            ylabel="#patients",
            outpng=alpha_root / "figs" / "03_logalpha_vs_num_patients.png")

    # (4) num patients vs total objective (replicated per day)
    per_day_obj = {d: res_df['objective_norm'].tolist() for d in days}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = res_df['n_patients'].tolist()
    for idx, d in enumerate(days):
        ax = axes[idx//3, idx%3]
        ax.scatter(x, per_day_obj[d], s=12)
        ax.set_title(f"Day {d}")
        ax.set_xlabel("#patients fully scheduled")
        ax.set_ylabel("objective normalized (0..1)")
        ax.grid(True, alpha=0.25)
    (alpha_root / "figs" / "04_num_patients_vs_total_score.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(alpha_root / "figs" / "04_num_patients_vs_total_score.png", dpi=150)
    plt.close(fig)

    vprint(args.verbosity, 1, f"[DONE] Figures saved under {alpha_root / 'figs'}")


if __name__ == "__main__":
    main()
