#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA/SA Path Builder — v3 (FAST, 3500/day, constraints-aware)
-----------------------------------------------------------
• Exactly 3,500 chains per patient per day (configurable).
• For patients with any GROUP treatments, at least 1,500 chains per day include ≥1 group slot.
• Patients considered: **only IDs 1..200** (keeps all therapists).
• Day selection:
    - If a patient has ≤ 12 DISTINCT treatments  → build ONLY on allowed days (from patients_constraints_opmed.csv).
    - If a patient has > 12 DISTINCT treatments  → build on ALL days (1..6), ignoring allowed-days.
• Work windows: 1–5 ⇒ 08:00–19:00 ; 6 ⇒ 10:00–14:00.
• Faster I/O: accumulate rows in memory, then single append-to-CSV per day. No per-row concat.

Output schema (CSV) under ./<graphs_subdir> (default: daily_graphs_fast)
  • day{d}_nodes.csv  : node_id,node_type,day,patient,treatment,therapist,start,end,group,cap_min,cap_max
  • day{d}_edges.csv  : src_id,dst_id,edge_type,day,path_id
  • day{d}_paths.csv  : path_id,day,patient,method,score_raw,score_norm,contains_group,slots_json
  • day{d}_group_slots.csv : treatment,therapist,start,end,cap_min,cap_max   (for reference)

Usage (defaults already match your request):
    python GA_SA_path_builder_v3_fast.py \
        --base_dir . \
        --graphs_subdir daily_graphs_fast \
        --days 1,2,3,4,5,6 \
        --per_patient 3500 \
        --group_quota 1500 \
        --group_slots 10 \
        --method both \
        --offlist_prob 0.12 \
        --verbosity 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd

# ----------------------------- Constants -----------------------------
GROUP_T_IDS: Set[int] = {11, 13, 17, 19, 25, 27, 14, 20, 15, 18, 24}

WAIT_MIN = 3
WAIT_GOOD_MAX = 7
WAIT_MAX = 15

# Work windows (minutes from midnight)
WIN_1_5 = (8*60, 19*60)   # 08:00–19:00  (updated per request)
WIN_6   = (10*60, 14*60)  # 10:00–14:00

DEFAULT_DUR = 45
MINUTE_GRID = 5

# ----------------------------- Helpers -------------------------------

def vprint(verbosity: int, level: int, msg: str) -> None:
    if verbosity >= level:
        print(msg, flush=True)

def _canon(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()

def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    sx = str(x).strip()
    if sx == "":
        return None
    try:
        return int(sx)
    except Exception:
        m = re.search(r"-?\d+", sx)
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
    m = (m // MINUTE_GRID) * MINUTE_GRID
    return f"{m//60:02d}:{m%60:02d}"

# ----------------------------- Data ----------------------------------

def load_patients_treatments(fp: Path) -> List[Tuple[int, List[int]]]:
    df = pd.read_csv(fp).fillna("")
    if "patient" not in df.columns:
        raise ValueError("patients_treatments_opmed.csv must contain column 'patient'")
    rows: List[Tuple[int, List[int]]] = []
    if "treatments" in df.columns:
        for _, r in df.iterrows():
            p = _to_int(r["patient"]) or 0
            if not p:
                continue
            ints = list(map(int, re.findall(r"\d+", str(r["treatments"]))))
            rows.append((p, ints))
    elif "treatment" in df.columns:
        groups: Dict[int, List[int]] = {}
        order: List[int] = []
        for _, r in df.iterrows():
            p = _to_int(r["patient"]) or 0
            t = _to_int(r["treatment"]) or 0
            if not p or not t:
                continue
            if p not in groups:
                groups[p] = []
                order.append(p)
            groups[p].append(t)
        rows = [(p, groups[p]) for p in order]
    else:
        raise ValueError("patients_treatments_opmed.csv must have either 'treatments' or 'treatment' column")
    return rows

def load_allowed_days(fp: Path) -> Dict[int, Set[int]]:
    if not fp.exists():
        return {}
    df = pd.read_csv(fp).fillna("")
    if "patient" not in df.columns:
        return {}
    out: Dict[int, Set[int]] = {}
    for _, r in df.iterrows():
        p = _to_int(r["patient"]) or 0
        if not p:
            continue
        days: Set[int] = set()
        for c in df.columns:
            if str(c).lower().startswith("day"):
                v = _to_int(r[c])
                if v in {1,2,3,4,5,6}:
                    days.add(v)
        out[p] = days
    return out

def load_durations(fp: Path) -> Dict[int, int]:
    if not fp.exists():
        return {}
    df = pd.read_csv(fp).fillna("")
    if "treatment" not in df.columns:
        return {}
    dur_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "duration" in lc or "minute" in lc or lc == "dur":
            dur_col = c
            break
    if not dur_col:
        return {}
    out: Dict[int, int] = {}
    for _, r in df.iterrows():
        t = _to_int(r["treatment"]) or 0
        if not t:
            continue
        try:
            d = int(float(r[dur_col]))
        except Exception:
            d = DEFAULT_DUR
        out[t] = d if d > 0 else DEFAULT_DUR
    return out

def load_eligibility(fp: Path) -> Dict[int, List[int]]:
    if not fp.exists():
        return {}
    df = pd.read_csv(fp).fillna("")
    if not {"therapist","treatment"}.issubset(df.columns):
        return {}
    out: Dict[int, Set[int]] = {}
    for _, r in df.iterrows():
        th = _to_int(r["therapist"]) or 0
        tr = _to_int(r["treatment"]) or 0
        if th and tr:
            out.setdefault(tr, set()).add(th)
    return {k: sorted(v) for k, v in out.items()}

def load_priorities(fp: Path) -> Dict[Tuple[int,int], List[int]]:
    if not fp.exists():
        return {}
    df = pd.read_csv(fp).fillna("")
    if "patient" not in df.columns or "treatment" not in df.columns:
        return {}
    out: Dict[Tuple[int,int], List[int]] = {}
    for _, r in df.iterrows():
        p = _to_int(r.get("patient")) or 0
        t = _to_int(r.get("treatment")) or 0
        if not p or not t:
            continue
        tops: List[int] = []
        for k in ("priority_1","priority_2","priority_3"):
            if k in df.columns:
                v = _to_int(r.get(k))
                if v:
                    tops.append(v)
        out[(p,t)] = tops
    return out

# -------------------------- Scheduling utils -------------------------

def work_window(day: int) -> Tuple[int,int]:
    return WIN_1_5 if day in {1,2,3,4,5} else WIN_6

def choose_therapist_for_personal(
    patient: int,
    treatment: int,
    priorities: Dict[Tuple[int,int], List[int]],
    eligibility: Dict[int, List[int]],
    offlist_prob: float,
    rng: random.Random,
) -> Optional[int]:
    elig = eligibility.get(treatment, [])
    if not elig:
        return None
    tops = [th for th in priorities.get((patient, treatment), []) if th in elig]
    if tops and rng.random() > offlist_prob:
        return rng.choice(tops)
    return rng.choice(elig)

def wait_penalty(w: int) -> float:
    if w < 0:
        return 1e6
    if w < WAIT_MIN:
        return (WAIT_MIN - w) * 50.0
    if w <= WAIT_GOOD_MAX:
        return 0.0
    if w <= WAIT_MAX:
        return (w - WAIT_GOOD_MAX) * 2.0
    return 1e6

def therapist_penalty(patient: int, treatment: int, therapist: Optional[int], priorities: Dict[Tuple[int,int], List[int]]) -> float:
    if therapist is None or therapist == 0:
        return 25.0
    tops = priorities.get((patient, treatment), [])
    if not tops:
        return 8.0
    if therapist == tops[0]:
        return 0.0
    if len(tops) > 1 and therapist == tops[1]:
        return 1.0
    if len(tops) > 2 and therapist == tops[2]:
        return 2.0
    return 8.0

# ---------------------------- Graph I/O ------------------------------

def ensure_out_dir(base_dir: Path, graphs_subdir: str) -> Path:
    gdir = base_dir / graphs_subdir
    gdir.mkdir(parents=True, exist_ok=True)
    return gdir

def write_headers_if_missing(gdir: Path, day: int) -> None:
    nfp = gdir / f"day{day}_nodes.csv"
    efp = gdir / f"day{day}_edges.csv"
    pfp = gdir / f"day{day}_paths.csv"
    if not nfp.exists():
        pd.DataFrame(columns=[
            "node_id","node_type","day","patient","treatment","therapist","start","end","group","cap_min","cap_max"
        ]).to_csv(nfp, index=False)
    if not efp.exists():
        pd.DataFrame(columns=["src_id","dst_id","edge_type","day","path_id"]).to_csv(efp, index=False)
    if not pfp.exists():
        pd.DataFrame(columns=[
            "path_id","day","patient","method","score_raw","score_norm","contains_group","slots_json"
        ]).to_csv(pfp, index=False)

def append_rows_csv(csv_path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    header = not csv_path.exists()
    df.to_csv(csv_path, index=False, mode="a", header=header, line_terminator="\n")

# --------------------------- Group slots -----------------------------

def premint_group_slots(
    day: int,
    durations: Dict[int,int],
    eligibility: Dict[int, List[int]],
    group_slots_per_t: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Create reference group slots for this day.
    Returns list of dicts: {treatment, therapist, start, end, cap_min, cap_max}
    """
    start_win, end_win = work_window(day)
    slots: List[Dict[str, Any]] = []

    for t in sorted(GROUP_T_IDS):
        dur = durations.get(t, DEFAULT_DUR)
        elig = eligibility.get(t, [])
        if not elig:
            continue
        # capacities: 2–3 for 11,13,17,19 and 2–8 for others
        if t in {11,13,17,19}:
            cmin, cmax = 2, 3
        else:
            cmin, cmax = 2, 8
        ths = elig if len(elig) <= 2 else rng.sample(elig, 2)

        span = end_win - start_win - dur
        if span <= 0:
            continue
        for k in range(group_slots_per_t):
            a = start_win + int((k + 0.5) * span / max(1, group_slots_per_t))
            a = (a // MINUTE_GRID) * MINUTE_GRID
            jitter = rng.choice([-10,-5,0,5,10])
            s = max(start_win, min(a + jitter, end_win - dur))
            e = s + dur
            slots.append({
                "treatment": t,
                "therapist": rng.choice(ths),
                "start": s,
                "end": e,
                "cap_min": cmin,
                "cap_max": cmax,
            })
    return slots

# ------------------------- Chain generation --------------------------

@dataclass
class Chain:
    treatment_triplet: Tuple[int,int,int]
    time_triplet: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]  # (s,e) per slot
    therapist_triplet: Tuple[Optional[int], Optional[int], Optional[int]]
    contains_group: bool
    raw_penalty: float
    method: str  # "GA" or "SA"

def make_personal_slot(
    day: int,
    treatment: int,
    patient: int,
    durations: Dict[int,int],
    priorities: Dict[Tuple[int,int], List[int]],
    eligibility: Dict[int, List[int]],
    offlist_prob: float,
    rng: random.Random,
    around: Optional[int] = None,
    before: bool = False,
    after: bool = False,
) -> Optional[Tuple[Tuple[int,int], int]]:
    dur = durations.get(treatment, DEFAULT_DUR)
    s_win, e_win = work_window(day)

    if around is None:
        s = rng.randrange(s_win, e_win - dur + 1, MINUTE_GRID)
        e = s + dur
    else:
        if before:
            wait = rng.randint(WAIT_MIN, min(WAIT_GOOD_MAX, WAIT_MAX))
            e = max(s_win + dur, around - wait)
            s = e - dur
            if s < s_win:
                return None
        elif after:
            wait = rng.randint(WAIT_MIN, min(WAIT_GOOD_MAX, WAIT_MAX))
            s = min(e_win - dur, around + wait)
            e = s + dur
            if e > e_win:
                return None
        else:
            s = rng.randrange(s_win, e_win - dur + 1, MINUTE_GRID)
            e = s + dur

    s = (s // MINUTE_GRID) * MINUTE_GRID
    e = (e // MINUTE_GRID) * MINUTE_GRID
    th = choose_therapist_for_personal(patient, treatment, priorities, eligibility, offlist_prob, rng)
    if th is None:
        return None
    return (s, e), th

def score_chain(patient: int,
                day: int,
                treatments: Tuple[int,int,int],
                times: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]],
                therapists: Tuple[Optional[int], Optional[int], Optional[int]],
                priorities: Dict[Tuple[int,int], List[int]]) -> float:
    w1 = times[1][0] - times[0][1]
    w2 = times[2][0] - times[1][1]
    if not (WAIT_MIN <= w1 <= WAIT_MAX and WAIT_MIN <= w2 <= WAIT_MAX):
        return 1e9
    wpen = wait_penalty(w1) + wait_penalty(w2)
    tpen = 0.0
    for idx, trt in enumerate(treatments):
        if trt not in GROUP_T_IDS:
            tpen += therapist_penalty(patient, trt, therapists[idx], priorities)
    return wpen + tpen

def make_chain_single_group(
    day: int,
    patient: int,
    personal_pool: List[int],
    group_pool: List[int],
    group_slots: List[Dict[str, Any]],
    durations: Dict[int,int],
    priorities: Dict[Tuple[int,int], List[int]],
    eligibility: Dict[int, List[int]],
    offlist_prob: float,
    rng: random.Random,
) -> Optional[Chain]:
    if not group_pool or len(personal_pool) == 0:
        return None
    g_t = rng.choice(group_pool)
    g_cands = [s for s in group_slots if s["treatment"] == g_t]
    if not g_cands:
        return None
    g = rng.choice(g_cands)

    per_choices = [t for t in personal_pool if t != g_t]
    if len(per_choices) < 2:
        global_personals = [t for t in eligibility.keys() if t not in GROUP_T_IDS]
        rng.shuffle(global_personals)
        for t in global_personals:
            if t not in per_choices:
                per_choices.append(t)
            if len(per_choices) >= 2:
                break
    if len(per_choices) < 2:
        return None
    p1, p2 = rng.sample(per_choices, 2)

    p1_slot = make_personal_slot(day, p1, patient, durations, priorities, eligibility, offlist_prob, rng,
                                 around=g["start"], before=True)
    p2_slot = make_personal_slot(day, p2, patient, durations, priorities, eligibility, offlist_prob, rng,
                                 around=g["end"], after=True)
    if not p1_slot or not p2_slot:
        return None
    (s1,e1), th1 = p1_slot
    (s3,e3), th3 = p2_slot
    s2, e2, th2 = g["start"], g["end"], g["therapist"]

    treatments = (p1, g_t, p2)
    times      = ((s1,e1),(s2,e2),(s3,e3))
    ths        = (th1, th2, th3)

    penalty = score_chain(patient, day, treatments, times, ths, priorities)
    if penalty >= 1e9:
        return None
    return Chain(treatments, times, ths, True, penalty, method="GA")

def make_chain_two_groups(
    day: int,
    patient: int,
    personal_pool: List[int],
    group_pool: List[int],
    group_slots: List[Dict[str, Any]],
    durations: Dict[int,int],
    priorities: Dict[Tuple[int,int], List[int]],
    eligibility: Dict[int, List[int]],
    offlist_prob: float,
    rng: random.Random,
) -> Optional[Chain]:
    if len(group_pool) < 2:
        return None
    g1_t, g2_t = rng.sample(group_pool, 2)
    c1 = [s for s in group_slots if s["treatment"] == g1_t]
    c2 = [s for s in group_slots if s["treatment"] == g2_t]
    if not c1 or not c2:
        return None
    s1 = rng.choice(c1)
    s2 = rng.choice(c2)
    first, second = (s1, s2) if s1["start"] <= s2["start"] else (s2, s1)

    between_t: Optional[int] = None
    if personal_pool:
        between_t = rng.choice([t for t in personal_pool if t not in {g1_t, g2_t}] or personal_pool)
    if between_t is None:
        return None

    dur_b = durations.get(between_t, DEFAULT_DUR)
    latest_start = second["start"] - WAIT_MIN - dur_b
    earliest_start = first["end"] + WAIT_MIN
    if earliest_start + dur_b + WAIT_MIN <= second["start"]:
        s_b = rng.randrange(earliest_start, latest_start + 1, MINUTE_GRID)
        e_b = s_b + dur_b
        th_b = choose_therapist_for_personal(patient, between_t, priorities, eligibility, offlist_prob, rng)
        if th_b is None:
            return None
        treatments = (first["treatment"], between_t, second["treatment"])
        times      = ((first["start"], first["end"]), (s_b,e_b), (second["start"], second["end"]))
        ths        = (first["therapist"], th_b, second["therapist"])
        penalty = score_chain(patient, day, treatments, times, ths, priorities)
        if penalty >= 1e9:
            return None
        return Chain(treatments, times, ths, True, penalty, method="GA")

    # fallback: before/after
    alt = rng.choice(["before","after"])
    if alt == "before":
        slot = make_personal_slot(day, between_t, patient, durations, priorities, eligibility, offlist_prob, rng,
                                  around=first["start"], before=True)
        if not slot:
            return None
        (s_b,e_b), th_b = slot
        treatments = (between_t, first["treatment"], second["treatment"])
        times      = ((s_b,e_b), (first["start"],first["end"]), (second["start"],second["end"]))
        ths        = (th_b, first["therapist"], second["therapist"])
    else:
        slot = make_personal_slot(day, between_t, patient, durations, priorities, eligibility, offlist_prob, rng,
                                  around=second["end"], after=True)
        if not slot:
            return None
        (s_b,e_b), th_b = slot
        treatments = (first["treatment"], second["treatment"], between_t)
        times      = ((first["start"],first["end"]), (second["start"],second["end"]), (s_b,e_b))
        ths        = (first["therapist"], second["therapist"], th_b)

    penalty = score_chain(patient, day, treatments, times, ths, priorities)
    if penalty >= 1e9:
        return None
    return Chain(treatments, times, ths, True, penalty, method="GA")

def make_chain_personal_only(
    day: int,
    patient: int,
    personal_pool: List[int],
    durations: Dict[int,int],
    priorities: Dict[Tuple[int,int], List[int]],
    eligibility: Dict[int, List[int]],
    offlist_prob: float,
    rng: random.Random,
) -> Optional[Chain]:
    per_choices = [t for t in personal_pool if t not in GROUP_T_IDS]
    if len(per_choices) < 3:
        global_personals = [t for t in eligibility.keys() if t not in GROUP_T_IDS]
        rng.shuffle(global_personals)
        for t in global_personals:
            if t not in per_choices:
                per_choices.append(t)
            if len(per_choices) >= 3:
                break
    if len(per_choices) < 3:
        return None
    t1, t2, t3 = rng.sample(per_choices, 3)

    d1 = durations.get(t1, DEFAULT_DUR)
    d2 = durations.get(t2, DEFAULT_DUR)
    d3 = durations.get(t3, DEFAULT_DUR)
    s_win, e_win = work_window(day)

    s1 = rng.randrange(s_win, e_win - (d1+d2+d3 + 2*WAIT_MIN) + 1, MINUTE_GRID)
    e1 = s1 + d1
    w12 = rng.randint(WAIT_MIN, min(WAIT_GOOD_MAX, WAIT_MAX))
    s2 = e1 + w12
    e2 = s2 + d2
    if e2 + WAIT_MIN + d3 > e_win:
        return None
    w23 = rng.randint(WAIT_MIN, min(WAIT_GOOD_MAX, WAIT_MAX))
    s3 = e2 + w23
    e3 = s3 + d3

    th1 = choose_therapist_for_personal(patient, t1, priorities, eligibility, offlist_prob, rng)
    th2 = choose_therapist_for_personal(patient, t2, priorities, eligibility, offlist_prob, rng)
    th3 = choose_therapist_for_personal(patient, t3, priorities, eligibility, offlist_prob, rng)
    if not (th1 and th2 and th3):
        return None

    treatments = (t1,t2,t3)
    times      = ((s1,e1),(s2,e2),(s3,e3))
    ths        = (th1, th2, th3)
    penalty = score_chain(patient, day, treatments, times, ths, priorities)
    if penalty >= 1e9:
        return None
    return Chain(treatments, times, ths, False, penalty, method="GA")

# --- Lightweight SA refine (kept small for speed) ---
def sa_refine(chain: Chain, day: int, durations: Dict[int,int], priorities: Dict[Tuple[int,int], List[int]], rng: random.Random, iters: int = 20) -> Chain:
    best = chain
    for _ in range(iters):
        treatments = list(best.treatment_triplet)
        times = [list(x) for x in best.time_triplet]
        ths = list(best.therapist_triplet)
        idxs = [i for i,t in enumerate(treatments) if t not in GROUP_T_IDS]
        if not idxs:
            break
        i = rng.choice(idxs)
        delta = rng.choice([-10,-5,5,10])
        s_new = times[i][0] + delta
        e_new = times[i][1] + delta
        s_win, e_win = work_window(day)
        if s_new < s_win or e_new > e_win:
            continue
        new_times = [tuple(x) for x in times]
        new_times[i] = (s_new, e_new)
        new_pen = score_chain(0, day, tuple(treatments), tuple(new_times), tuple(ths), priorities)
        if new_pen < best.raw_penalty:
            best = Chain(tuple(treatments), tuple(new_times), tuple(ths), best.contains_group, new_pen, method="SA")
    return best

# --------------------------- Main builder ----------------------------

def main():
    ap = argparse.ArgumentParser(description="FAST GA+SA path builder (3500/day, 1500 group when applicable), constraints-aware and optimized I/O.")
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--graphs_subdir", default="daily_graphs_fast", help="Where to write CSVs (new folder recommended for speed)")
    ap.add_argument("--days", default="1,2,3,4,5,6", help="Comma-separated list of days to build")
    ap.add_argument("--per_patient", type=int, default=3500, help="Chains per patient per day")
    ap.add_argument("--group_quota", type=int, default=1500, help="Min group-containing chains per patient-day for patients with group treatments")
    ap.add_argument("--group_slots", type=int, default=10, help="Pre-minted group slots per group treatment per day")
    ap.add_argument("--method", choices=["ga","sa","both"], default="both", help="Generation method")
    ap.add_argument("--offlist_prob", type=float, default=0.12, help="Probability to use a non-top3 eligible therapist for personal slots")
    ap.add_argument("--limit_patients", type=int, default=0, help="Extra cap AFTER enforcing 1..200 (0 = no extra limit)")
    ap.add_argument("--seed", type=int, default=20251109)
    ap.add_argument("--verbosity", type=int, default=1, help="0=silent, 1=progress per day/patient, 2=extra")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base = Path(args.base_dir)
    gdir = ensure_out_dir(base, args.graphs_subdir)

    # Load inputs
    pt_rows = load_patients_treatments(base / "patients_treatments_opmed.csv")
    durations = load_durations(base / "treatments_durations_opmed.csv")
    eligibility = load_eligibility(base / "therapists_treatments.csv")
    priorities = load_priorities(base / "patient_treatment_therapist_priorities_opmed.csv")
    allowed_days_map = load_allowed_days(base / "patients_constraints_opmed.csv")

    # Determine patient order — enforce 1..200
    patients_full = [p for p, _ in pt_rows]
    patients_1_200 = [p for p in patients_full if isinstance(p, int) and 1 <= p <= 200]
    if args.limit_patients > 0:
        patients_1_200 = patients_1_200[:args.limit_patients]

    # quick maps
    patient_treats: Dict[int, List[int]] = {p: lst for p, lst in pt_rows}
    distinct_count: Dict[int, int] = {p: len(set(lst)) for p, lst in pt_rows}

    # Parse days
    days = [int(x) for x in args.days.split(',') if str(x).strip()]

    # Per-day slot ids
    slot_counters: Dict[int,int] = {d: 0 for d in days}

    # Build per day
    for day in days:
        write_headers_if_missing(gdir, day)

        # Pre-mint group slots and save reference
        group_slots = premint_group_slots(day, durations, eligibility, args.group_slots, rng)
        pd.DataFrame([
            {"treatment": s["treatment"], "therapist": s["therapist"], "start": m2t(s["start"]), "end": m2t(s["end"]), "cap_min": s["cap_min"], "cap_max": s["cap_max"]}
            for s in group_slots
        ]).to_csv(gdir / f"day{day}_group_slots.csv", index=False)

        # Index for quick filter
        group_slots_by_t: Dict[int, List[Dict[str,Any]]] = {}
        for s in group_slots:
            group_slots_by_t.setdefault(s["treatment"], []).append(s)

        vprint(args.verbosity, 1, f"[DAY {day}] start — window {m2t(work_window(day)[0])}-{m2t(work_window(day)[1])}; pre-minted group slots: {len(group_slots)}")

        # In-memory row buffers
        nodes_rows: List[Dict[str, Any]] = []
        edges_rows: List[Dict[str, Any]] = []
        paths_rows: List[Dict[str, Any]] = []
        core_nodes_written: Set[str] = set()  # avoid duplicates within this run

        total_paths_day = 0
        total_group_paths_day = 0

        for idx_p, p in enumerate(patients_1_200, 1):
            treats = patient_treats.get(p, [])
            if not treats:
                continue

            # Decide if this patient is eligible on THIS day
            dcnt = distinct_count.get(p, 0)
            allowed_days = allowed_days_map.get(p, set())
            if dcnt <= 12:
                # strict: only if day is allowed (if allowed_days is empty, treat as not allowed)
                if day not in allowed_days:
                    continue
            else:
                # >12 distinct: generate in all days
                pass

            per_pool = sorted({t for t in treats if t not in GROUP_T_IDS})
            grp_pool = sorted({t for t in treats if t in GROUP_T_IDS and t in group_slots_by_t})

            need_total = args.per_patient
            need_group = min(args.group_quota, need_total) if grp_pool else 0
            need_personal_only = need_total - need_group

            # Ensure core logical nodes (source/target/patient) once per run
            src_id = f"S{day}"
            tgt_id = f"T{day}"
            pat_id = f"P{p}_D{day}"
            for nid, ntype in ((src_id,"source"), (tgt_id,"target"), (pat_id,"patient")):
                if nid not in core_nodes_written:
                    nodes_rows.append({
                        "node_id": nid, "node_type": ntype, "day": day, "patient": (p if ntype=="patient" else ""),
                        "treatment": "", "therapist": "", "start": "", "end": "", "group": 0, "cap_min": "", "cap_max": ""
                    })
                    core_nodes_written.add(nid)

            # Generate candidates
            rng_local = rng  # alias
            cands: List[Chain] = []

            # Stage 1: group-containing
            attempts = 0
            while len([c for c in cands if c.contains_group]) < need_group and attempts < need_group * 10:
                attempts += 1
                maker = make_chain_two_groups if (len(grp_pool) >= 2 and rng_local.random() < 0.30) else make_chain_single_group
                ch = maker(day, p, per_pool, grp_pool, group_slots, durations, priorities, eligibility, args.offlist_prob, rng_local)
                if ch is None:
                    continue
                if args.method in {"sa", "both"}:
                    ch = sa_refine(ch, day, durations, priorities, rng_local)
                cands.append(ch)

            made_group = len([c for c in cands if c.contains_group])

            # Stage 2: personal-only
            attempts2 = 0
            while len(cands) < need_total and attempts2 < need_personal_only * 10:
                attempts2 += 1
                ch = make_chain_personal_only(day, p, per_pool, durations, priorities, eligibility, args.offlist_prob, rng_local)
                if ch is None:
                    continue
                if args.method in {"sa", "both"}:
                    ch = sa_refine(ch, day, durations, priorities, rng_local)
                cands.append(ch)

            if len(cands) < need_total:
                vprint(args.verbosity, 2, f"    [P {p}] only {len(cands)}/{need_total} built (groups: {made_group}/{need_group})")

            # Normalize scores (lower is better) to 0..1 per patient-day
            if cands:
                raw_list = [c.raw_penalty for c in cands]
                mn, mx = min(raw_list), max(raw_list)
                denom = (mx - mn) if mx > mn else 1.0
            else:
                mn, denom = 0.0, 1.0

            # Emit rows
            group_count_written = 0
            for ch in cands:
                path_id = f"D{day}_P{p}_{ch.method}_{slot_counters[day]+1}"  # unique enough for new folder

                # Build 3 slot nodes
                node_ids = []
                for i in range(3):
                    t = ch.treatment_triplet[i]
                    s, e = ch.time_triplet[i]
                    th = ch.therapist_triplet[i] or 0
                    slot_counters[day] += 1
                    nid = f"U{day}_{slot_counters[day]}"
                    nodes_rows.append({
                        "node_id": nid, "node_type": "slot", "day": day, "patient": p,
                        "treatment": t, "therapist": th, "start": m2t(s), "end": m2t(e),
                        "group": 1 if t in GROUP_T_IDS else 0,
                        "cap_min": 2 if t in GROUP_T_IDS else "",
                        "cap_max": (3 if t in {11,13,17,19} else 8) if t in GROUP_T_IDS else "",
                    })
                    node_ids.append(nid)

                # Edges (patient->slot->slot->slot->target)
                edges_rows.append({"src_id": pat_id,      "dst_id": node_ids[0], "edge_type": "patient->slot", "day": day, "path_id": path_id})
                edges_rows.append({"src_id": node_ids[0], "dst_id": node_ids[1], "edge_type": "slot->slot",    "day": day, "path_id": path_id})
                edges_rows.append({"src_id": node_ids[1], "dst_id": node_ids[2], "edge_type": "slot->slot",    "day": day, "path_id": path_id})
                edges_rows.append({"src_id": node_ids[2], "dst_id": tgt_id,      "edge_type": "slot->target",  "day": day, "path_id": path_id})

                # Path row
                score_norm = 0.5 if denom == 1.0 else (ch.raw_penalty - mn) / denom
                slots_json = [
                    {"treatment": ch.treatment_triplet[i], "therapist": ch.therapist_triplet[i] or 0,
                     "start": m2t(ch.time_triplet[i][0]), "end": m2t(ch.time_triplet[i][1]),
                     "group": 1 if ch.treatment_triplet[i] in GROUP_T_IDS else 0}
                    for i in range(3)
                ]
                paths_rows.append({
                    "path_id": path_id, "day": day, "patient": p, "method": ch.method,
                    "score_raw": round(float(ch.raw_penalty), 5),
                    "score_norm": round(float(score_norm), 6),
                    "contains_group": 1 if ch.contains_group else 0,
                    "slots_json": json.dumps(slots_json, ensure_ascii=False),
                })
                if ch.contains_group:
                    group_count_written += 1

            total_paths_day += len(cands)
            total_group_paths_day += group_count_written
            vprint(args.verbosity, 1, f"[DAY {day}] patient {p}: wrote {len(cands)} paths (groups {group_count_written}/{need_group})")

        # Append to files once per day
        append_rows_csv(gdir / f"day{day}_nodes.csv", nodes_rows,
                        ["node_id","node_type","day","patient","treatment","therapist","start","end","group","cap_min","cap_max"])
        append_rows_csv(gdir / f"day{day}_edges.csv", edges_rows,
                        ["src_id","dst_id","edge_type","day","path_id"])
        append_rows_csv(gdir / f"day{day}_paths.csv", paths_rows,
                        ["path_id","day","patient","method","score_raw","score_norm","contains_group","slots_json"])

        vprint(args.verbosity, 1, f"[OK] Day {day} saved ({args.graphs_subdir}). nodes+={len(nodes_rows)} edges+={len(edges_rows)} paths+={len(paths_rows)} (group paths today: {total_group_paths_day})")


if __name__ == "__main__":
    main()
