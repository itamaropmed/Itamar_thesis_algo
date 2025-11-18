#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph Generator — fixed & fast
----------------------------------
Build a **daily hypergraph** whose nodes are *paths* from your GA/SA graphs,
using covers (hyperedges) of three types:
  1) Patient cover  (P_<patient>)
  2) Therapist cover (TH_<therapist>) — includes a path if ANY of its 3 slots
     is with that therapist (group or personal). Therapist=0 is ignored.
  3) Group-treatment-slot cover (GTS_<treatment>_<j>): for each pre-minted
     group anchor j of treatment t on that day, includes every path that uses
     that exact anchor (match by treatment, start, end, therapist).

Compatible with GA v3 FAST output ("daily_graphs_fast"): expects per-day CSVs:
  • day{d}_paths.csv  (must include a JSON column "slots_json")
  • day{d}_group_slots.csv (pre-minted anchors for that day)

Outputs (under <graphs_subdir>_hyper):
  • hyper_nodes_day{d}.csv   : one row per path → one hypernode
      hnode_id, day, patient, path_id, method, score_norm, contains_group,
      t1,s1,e1,th1,g1, t2,s2,e2,th2,g2, t3,s3,e3,th3,g3
  • hyper_covers_meta_day{d}.csv : metadata per cover
      cover_id, day, type, key1, key2, size
        - type ∈ {patient, therapist, gslot}
        - patient → key1=patient_id; therapist → key1=therapist_id
        - gslot   → key1=treatment_id, key2=slot_index (1..N within treatment)
  • hyper_edges_day{d}.csv  : incidence (long form)
      cover_id, hnode_id

Notes
-----
• Robust JSON parsing: handles both proper JSON and python-ish lists.
• Fixes your earlier crash: we construct the **wide** per-path dataframe with
  one row per path (not 3× rows), and only then derive covers.
• Fast: builds lists in memory, then writes once per file.

Usage
-----
python hypergraph_generator_fixed.py \
  --base_dir . \
  --graphs_subdir daily_graphs_fast \
  --days 1,2,3,4,5,6 \
  --verbosity 1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---- helpers ---------------------------------------------------------

def vprint(verbosity: int, lvl: int, msg: str) -> None:
    if verbosity >= lvl:
        print(msg, flush=True)


def _canon(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = _canon(x)
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        m = re.search(r"-?\d+", s)
        return int(m.group()) if m else None


def t2min(s: Any) -> int:
    s = _canon(s)
    if not s or ":" not in s:
        try:
            return int(s)
        except Exception:
            return 0
    h, m = s.split(":", 1)
    return int(h) * 60 + int(m)


def m2t(m: int) -> str:
    m = int(m)
    return f"{m//60:02d}:{m%60:02d}"


# ---- JSON parsing for slots_json ------------------------------------

def _parse_slots_json(cell: Any) -> Optional[List[Dict[str, Any]]]:
    """Return list of 3 slot dicts with keys: treatment, therapist, start, end, group.
    Accepts both JSON and python-literal-like strings.
    """
    if isinstance(cell, list):
        arr = cell
    else:
        s = _canon(cell)
        if s == "":
            return None
        try:
            arr = json.loads(s)
        except Exception:
            # try a fast python-ish literal fix: replace single quotes → double
            try:
                s2 = s.replace("'", '"')
                arr = json.loads(s2)
            except Exception:
                try:
                    import ast
                    arr = ast.literal_eval(s)
                except Exception:
                    return None
    if not isinstance(arr, list) or len(arr) != 3:
        return None
    # normalize keys/types
    out: List[Dict[str, Any]] = []
    for d in arr:
        try:
            t = _to_int(d.get("treatment")) or 0
            th = _to_int(d.get("therapist")) or 0
            s = t2min(d.get("start"))
            e = t2min(d.get("end"))
            g = 1 if str(d.get("group", 0)).strip() == "1" else 0
            out.append({"treatment": t, "therapist": th, "start": s, "end": e, "group": g})
        except Exception:
            return None
    return out


# ---- group anchors mapping ------------------------------------------

def load_group_anchors(graphs_dir: Path, day: int) -> Tuple[pd.DataFrame, Dict[Tuple[int,int,int,int], int]]:
    """Read day{d}_group_slots.csv and return (df, signature→index) mapping.
    Each treatment has slots indexed 1..N by their order of appearance.
    Signature is (treatment, start, end, therapist).
    """
    gfp = graphs_dir / f"day{day}_group_slots.csv"
    if not gfp.exists():
        return pd.DataFrame(), {}
    df = pd.read_csv(gfp).fillna("")
    sig2idx: Dict[Tuple[int,int,int,int], int] = {}
    # build per treatment order
    counters: Dict[int, int] = {}
    for _, r in df.iterrows():
        t = _to_int(r.get("treatment")) or 0
        s = t2min(r.get("start"))
        e = t2min(r.get("end"))
        th = _to_int(r.get("therapist")) or 0
        counters[t] = counters.get(t, 0) + 1
        sig2idx[(t, s, e, th)] = counters[t]
    return df, sig2idx


# ---- build hypernodes (one per path) --------------------------------

def build_hypernodes(paths_df: pd.DataFrame, verbosity: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bad = 0
    for i, r in enumerate(paths_df.itertuples(index=False), 1):
        day = int(getattr(r, "day"))
        patient = int(getattr(r, "patient"))
        path_id = str(getattr(r, "path_id"))
        method = str(getattr(r, "method"))
        score_norm = float(getattr(r, "score_norm")) if hasattr(r, "score_norm") else 0.5
        contains_group = int(getattr(r, "contains_group")) if hasattr(r, "contains_group") else 0
        slots = _parse_slots_json(getattr(r, "slots_json"))
        if not slots:
            bad += 1
            continue
        t1, t2, t3 = slots[0]["treatment"], slots[1]["treatment"], slots[2]["treatment"]
        s1, s2, s3 = slots[0]["start"], slots[1]["start"], slots[2]["start"]
        e1, e2, e3 = slots[0]["end"],   slots[1]["end"],   slots[2]["end"]
        th1,th2,th3 = slots[0]["therapist"], slots[1]["therapist"], slots[2]["therapist"]
        g1, g2, g3 = slots[0]["group"], slots[1]["group"], slots[2]["group"]
        hnode_id = f"H{day}_{i}"
        rows.append({
            "hnode_id": hnode_id, "day": day, "patient": patient, "path_id": path_id,
            "method": method, "score_norm": score_norm, "contains_group": contains_group,
            "t1": t1, "s1": s1, "e1": e1, "th1": th1, "g1": g1,
            "t2": t2, "s2": s2, "e2": e2, "th2": th2, "g2": g2,
            "t3": t3, "s3": s3, "e3": e3, "th3": th3, "g3": g3,
        })
    if bad and verbosity:
        vprint(verbosity, 1, f"    Skipped {bad} paths with unparsable slots_json")
    return pd.DataFrame(rows)


# ---- build covers (hyperedges) --------------------------------------

def build_covers_for_day(day: int,
                         H_nodes: pd.DataFrame,
                         group_sig2idx: Dict[Tuple[int,int,int,int], int],
                         verbosity: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (covers_meta_df, covers_edges_df)."""
    if H_nodes.empty:
        return pd.DataFrame(columns=["cover_id","day","type","key1","key2","size"]), \
               pd.DataFrame(columns=["cover_id","hnode_id"])

    # Patient covers ---------------------------------------------------
    covers_meta: List[Dict[str, Any]] = []
    edges_rows: List[Dict[str, Any]] = []

    # group by patient
    for patient, g in H_nodes.groupby("patient"):
        members = g["hnode_id"].tolist()
        cid = f"P_{int(patient)}"
        covers_meta.append({"cover_id": cid, "day": day, "type": "patient", "key1": int(patient), "key2": "", "size": len(members)})
        edges_rows.extend({"cover_id": cid, "hnode_id": h} for h in members)

    # Therapist covers -------------------------------------------------
    # accumulate set of therapists per hnode (exclude 0)
    for row in H_nodes.itertuples(index=False):
        ths = set(x for x in [row.th1, row.th2, row.th3] if int(x) != 0)
        for th in ths:
            cid = f"TH_{int(th)}"
            edges_rows.append({"cover_id": cid, "hnode_id": row.hnode_id})
    # finalize therapist cover meta (compute sizes)
    th_sizes: Dict[str, int] = {}
    for e in edges_rows:
        if e["cover_id"].startswith("TH_"):
            th_sizes[e["cover_id"]] = th_sizes.get(e["cover_id"], 0) + 1
    for cid, sz in th_sizes.items():
        covers_meta.append({"cover_id": cid, "day": day, "type": "therapist", "key1": int(cid.split("_",1)[1]), "key2": "", "size": int(sz)})

    # Group-slot covers -----------------------------------------------
    if group_sig2idx:
        for row in H_nodes.itertuples(index=False):
            for (t, s, e, th, gflag) in (
                (row.t1, row.s1, row.e1, row.th1, row.g1),
                (row.t2, row.s2, row.e2, row.th2, row.g2),
                (row.t3, row.s3, row.e3, row.th3, row.g3),
            ):
                if int(gflag) == 1:
                    idx = group_sig2idx.get((int(t), int(s), int(e), int(th)))
                    if idx is None:
                        # try fallback without therapist match
                        idx = group_sig2idx.get((int(t), int(s), int(e), 0))
                    if idx is None:
                        continue
                    cid = f"GTS_{int(t)}_{int(idx)}"
                    edges_rows.append({"cover_id": cid, "hnode_id": row.hnode_id})
        # finalize gslot meta
        g_sizes: Dict[str, int] = {}
        for e in edges_rows:
            if e["cover_id"].startswith("GTS_"):
                g_sizes[e["cover_id"]] = g_sizes.get(e["cover_id"], 0) + 1
        for cid, sz in g_sizes.items():
            _, t, j = cid.split("_")
            covers_meta.append({"cover_id": cid, "day": day, "type": "gslot", "key1": int(t), "key2": int(j), "size": int(sz)})

    covers_meta_df = pd.DataFrame(covers_meta, columns=["cover_id","day","type","key1","key2","size"]).drop_duplicates(subset=["cover_id"]).reset_index(drop=True)
    covers_edges_df = pd.DataFrame(edges_rows, columns=["cover_id","hnode_id"])
    return covers_meta_df, covers_edges_df


# ---- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build daily hypergraphs from GA/SA paths (patient, therapist, and group-slot covers).")
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--graphs_subdir", default="daily_graphs_fast")
    ap.add_argument("--days", default="1,2,3,4,5,6")
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    verbosity = args.verbosity
    base = Path(args.base_dir)
    graphs_dir = base / args.graphs_subdir
    out_dir = base / f"{args.graphs_subdir}_hyper"
    out_dir.mkdir(parents=True, exist_ok=True)

    days = [int(x) for x in args.days.split(',') if str(x).strip()]

    for day in days:
        paths_fp = graphs_dir / f"day{day}_paths.csv"
        if not paths_fp.exists():
            vprint(verbosity, 1, f"[DAY {day}] missing {paths_fp.name}; skipping.")
            continue
        paths_df = pd.read_csv(paths_fp)
        vprint(verbosity, 1, f"[DAY {day}] Loading paths … count={len(paths_df):,}")

        # Build hypernodes (one per path)
        H_nodes = build_hypernodes(paths_df, verbosity)
        vprint(verbosity, 1, f"[DAY {day}] Hypernodes built: {len(H_nodes):,}")

        # Group anchors mapping for gslot covers
        group_df, sig2idx = load_group_anchors(graphs_dir, day)
        vprint(verbosity, 2, f"[DAY {day}] Group anchors: {len(group_df):,} (distinct signatures: {len(sig2idx)})")

        # Build covers
        covers_meta_df, covers_edges_df = build_covers_for_day(day, H_nodes, sig2idx, verbosity)
        vprint(verbosity, 1, f"[DAY {day}] Covers: meta={len(covers_meta_df):,} edges={len(covers_edges_df):,}")

        # Write outputs
        H_nodes.to_csv(out_dir / f"hyper_nodes_day{day}.csv", index=False)
        covers_meta_df.to_csv(out_dir / f"hyper_covers_meta_day{day}.csv", index=False)
        covers_edges_df.to_csv(out_dir / f"hyper_edges_day{day}.csv", index=False)
        vprint(verbosity, 1, f"[OK] Day {day} saved → {out_dir.name}/hyper_*_day{day}.csv")


if __name__ == "__main__":
    main()
