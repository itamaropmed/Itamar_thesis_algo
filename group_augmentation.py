#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
group_augmentation.py
---------------------
Augment existing per-day DAGs with extra GROUP-containing paths.

Changes requested:
• Days to augment: ONLY 1..4 (skip 5 and 6).
• Patients: ONLY the FIRST 100 patients who have any group treatment, in the
  file order of patients_treatments_opmed.csv.
• Count: EXACTLY 200 augmented paths per patient per day.

Inputs (default ./opmed_reut_thesis):
  graphs_out/day{d}_nodes.csv
  graphs_out/day{d}_edges.csv
  graphs_out/day{d}_paths.csv
  patients_treatments_opmed.csv
  patient_treatment_therapist_priorities_opmed.csv
  therapists_treatments.csv
  treatments_durations_opmed.csv
  patients_constraints_opmed.csv

Outputs (in-place updates):
  graphs_out/day{d}_nodes.csv
  graphs_out/day{d}_edges.csv
  graphs_out/day{d}_paths.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
import re
import random
import pandas as pd

# ---------------- Config ----------------
GROUP_T_LIST = [
    "treatment 25","treatment 27","treatment 14","treatment 20","treatment 15",
    "treatment 18","treatment 11","treatment 19","treatment 24","treatment 17","treatment 13"
]
GROUP_T = {s.strip().lower() for s in GROUP_T_LIST}

# ONLY days 1..4
DAYS_TO_AUGMENT = [1, 2, 3, 4]

GRID = 5                          # 5-min grid
IDEAL_WAIT = (3,7)
MAX_WAIT = 20
WORK_START = 8*60                 # 08:00
WORK_END   = 20*60                # 20:00
DEFAULT_DUR = 45
RNG = random.Random(1337)

# Three shared anchors per day (chosen to leave room around them)
GROUP_ANCHORS = ["09:00", "11:00", "13:30"]

# ---------------- Small utils ----------------
def _canon(s: Any) -> str:
    if pd.isna(s): return ""
    return str(s).strip()

def t2min(val: Any) -> int:
    if isinstance(val, (int,float)) and not pd.isna(val):
        return int(val)
    s = _canon(val)
    if not s: return 0
    if ":" not in s:
        try: return int(float(s))
        except Exception: return 0
    h,m = s.split(":",1)
    try: return 60*int(h) + int(m)
    except Exception: return 0

def m2t(m: int) -> str:
    m = int(m)
    m = (m // GRID) * GRID
    return f"{m//60:02d}:{m%60:02d}"

def is_group(t: str) -> bool:
    return _canon(t).lower() in GROUP_T

def wait_ok(w: int) -> bool:
    return w < MAX_WAIT and w >= -1  # allow tiny negative due to grid

def wait_score(w: int) -> float:
    a,b = IDEAL_WAIT
    w_eff = max(0, w)
    if w_eff >= MAX_WAIT: return 0.0
    if w_eff <= a: return w_eff / max(1,a)
    if w_eff <= b: return 1.0
    return max(0.0, 1.0 - (w_eff - b) / max(1, (MAX_WAIT - b)))

# ---------------- Data loaders ----------------
def load_durations(fp: Path) -> Dict[str,int]:
    if not fp.exists(): return {}
    df = pd.read_csv(fp, dtype=str)
    tcol = next((c for c in df.columns if c.lower().strip()=="treatment"), None)
    dcol = None
    for c in df.columns:
        lc=c.lower()
        if "duration" in lc or "minute" in lc or lc=="dur":
            dcol=c; break
    out={}
    if not tcol or not dcol: return out
    for _,r in df.iterrows():
        t=_canon(r[tcol])
        try: d=int(float(r[dcol]))
        except Exception: d=DEFAULT_DUR
        if t: out[t]= d if d>0 else DEFAULT_DUR
    return out

def load_eligibility(fp: Path) -> Dict[str, List[str]]:
    """treatment -> list of eligible therapists"""
    if not fp.exists(): return {}
    df = pd.read_csv(fp, dtype=str)
    need = {"therapist","treatment"}
    if any(c not in df.columns for c in need): return {}
    df["therapist"]=df["therapist"].map(_canon)
    df["treatment"]=df["treatment"].map(_canon)
    out: Dict[str, Set[str]] = {}
    for _,r in df.iterrows():
        if r["therapist"] and r["treatment"]:
            out.setdefault(r["treatment"], set()).add(r["therapist"])
    return {k: sorted(v) for k,v in out.items()}

def load_priorities(fp: Path) -> Dict[Tuple[str,str], List[str]]:
    if not fp.exists(): return {}
    df = pd.read_csv(fp, dtype=str).fillna("")
    need = {"patient","treatment"}
    if any(c not in df.columns for c in need): return {}
    out={}
    for _,r in df.iterrows():
        p=_canon(r["patient"]); t=_canon(r["treatment"])
        tops=[]
        for k in ("priority_1","priority_2","priority_3"):
            if k in df.columns:
                v=_canon(r.get(k,""))
                if v: tops.append(v)
        out[(p,t)] = tops
    return out

def load_patients_list(fp: Path) -> Dict[str, List[str]]:
    """patient -> treatments list with repeats (semicolon list)"""
    if not fp.exists(): return {}
    df = pd.read_csv(fp, dtype=str).fillna("")
    need = {"patient","treatments"}
    if any(c not in df.columns for c in need): return {}
    out={}
    for _,r in df.iterrows():
        p=_canon(r["patient"])
        items=[x.strip() for x in str(r["treatments"]).split(";") if x.strip()]
        out[p]=items
    return out

def load_patient_order(fp: Path) -> List[str]:
    """Return patients in the file order of patients_treatments_opmed.csv."""
    if not fp.exists(): return []
    df = pd.read_csv(fp, dtype=str).fillna("")
    if "patient" not in df.columns: return []
    return [_canon(x) for x in df["patient"].tolist()]

def load_allowed_days(fp: Path) -> Dict[str, Set[int]]:
    if not fp.exists(): return {}
    df = pd.read_csv(fp, dtype=str).fillna("")
    if "patient" not in df.columns: return {}
    out={}
    for _,r in df.iterrows():
        p=_canon(r["patient"])
        days=set()
        for c in df.columns:
            lc=c.lower()
            if lc.startswith("day"):
                try:
                    v=int(float(_canon(r[c]) or "0"))
                except Exception:
                    v=0
                if v in {1,2,3,4,5,6}:
                    days.add(v)
        out[p]= days if days else set(DAYS_TO_AUGMENT)  # default to 1..4 now
    return out

# ---------------- Graph IO ----------------
def read_graph_day(graphs_dir: Path, day: int):
    ndf = pd.read_csv(graphs_dir / f"day{day}_nodes.csv")
    edf = pd.read_csv(graphs_dir / f"day{day}_edges.csv")
    pdf = pd.read_csv(graphs_dir / f"day{day}_paths.csv")
    return ndf, edf, pdf

def write_graph_day(graphs_dir: Path, day: int, nodes: pd.DataFrame, edges: pd.DataFrame, paths: pd.DataFrame):
    nodes.to_csv(graphs_dir / f"day{day}_nodes.csv", index=False)
    edges.to_csv(graphs_dir / f"day{day}_edges.csv", index=False)
    paths.to_csv(graphs_dir / f"day{day}_paths.csv", index=False)

# ---------------- Group slot pre-mint ----------------
def premint_group_slots_for_day(day: int,
                                durations: Dict[str,int],
                                eligibility: Dict[str, List[str]]) -> Dict[str, List[Tuple[int,int,str]]]:
    """
    Return: treat -> list of (start_min, end_min, fixed_therapist) for 3 shared slots.
    Therapist is fixed per (treat,day) to unify sessions (choose first eligible).
    """
    slots: Dict[str, List[Tuple[int,int,str]]] = {}
    for t_lc in GROUP_T:
        t_name = None
        # durations keys are case sensitive; find canonical match
        for key in durations.keys():
            if key.strip().lower()==t_lc:
                t_name = key; break
        if not t_name:
            # try eligibility keys
            for key in eligibility.keys():
                if key.strip().lower()==t_lc:
                    t_name = key; break
        if not t_name:
            continue
        dur = int(durations.get(t_name, DEFAULT_DUR))
        elig = eligibility.get(t_name, [])
        if not elig:
            continue
        fixed_th = elig[0]
        times=[]
        for hhmm in GROUP_ANCHORS:
            s = max(WORK_START, t2min(hhmm))
            e = s + dur
            if e <= WORK_END:
                times.append((s,e,fixed_th))
        if times:
            slots[t_name] = times
    return slots

# ---------------- Therapist choice for personal ----------------
def choose_personal_therapist(p: str, t: str,
                              priorities: Dict[Tuple[str,str], List[str]],
                              eligibility: Dict[str, List[str]]) -> Optional[str]:
    tops = priorities.get((p,t), [])
    elig = eligibility.get(t, [])
    # prefer prioritized among eligible
    for th in tops:
        if th in elig:
            return th
    # else any eligible
    if elig:
        return RNG.choice(elig)
    return None

# ---------------- Build one augmented path ----------------
def build_augmented_chain(patient: str,
                          day: int,
                          treats_list: List[str],
                          durations: Dict[str,int],
                          priorities: Dict[Tuple[str,str], List[str]],
                          eligibility: Dict[str, List[str]],
                          premint_group: Dict[str, List[Tuple[int,int,str]]],
                          use_two_groups: bool) -> Optional[Tuple[List[Dict[str,Any]], float]]:
    """
    Returns: (slots[3], score) or None
    Each slot dict: {treatment, start, end, therapist, origin='AUG'}
    """
    # unique personal pool (exclude group)
    personal_pool = [t for t in treats_list if not is_group(t)]
    personal_uniq = []
    seen=set()
    for t in personal_pool:
        if t not in seen:
            seen.add(t); personal_uniq.append(t)

    # available group treatments in patient's list that we preminted today
    group_pool = []
    for t in treats_list:
        if is_group(t):
            # find canonical premint key
            t_key = next((k for k in premint_group.keys() if k.strip().lower()==t.strip().lower()), None)
            if t_key and t_key not in group_pool:
                group_pool.append(t_key)

    if not group_pool:
        return None

    RNG.shuffle(group_pool)
    # favor single-group paths; allow two-group when patient has >=2 group types
    if use_two_groups and len(group_pool)>=2:
        g1, g2 = group_pool[:2]
        # pick one preminted slot for each
        s1 = RNG.choice(premint_group[g1])
        s2 = RNG.choice(premint_group[g2])
        # order by time
        ordered = sorted([(g1,s1),(g2,s2)], key=lambda x: x[1][0])
        (tg1,(a1,b1,th1)), (tg2,(a2,b2,th2)) = ordered
        # choose one personal distinct from groups and from itself
        if not personal_uniq:
            # fallback: any non-group in eligibility
            alt_personals = [k for k in eligibility.keys() if not is_group(k)]
            if not alt_personals:
                return None
            personal_uniq = RNG.sample(alt_personals, min(5, len(alt_personals)))
        tp = next((t for t in personal_uniq if t.strip().lower() not in {tg1.strip().lower(), tg2.strip().lower()}), None)
        if not tp:
            return None
        dp = int(durations.get(tp, DEFAULT_DUR))
        # place tp around the two group slots with waits ~3..7
        # try as middle first
        s2start = b1 + RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
        s2end   = s2start + dp
        if s2end <= a2 - 3:
            chain = [(tg1,a1,b1,th1), (tp,s2start,s2end, choose_personal_therapist(patient, tp, priorities, eligibility) or (eligibility.get(tp,[""])[0] if eligibility.get(tp) else "")), (tg2,a2,b2,th2)]
        else:
            # else: tp first
            s1end = a1 - RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
            s1start = s1end - dp
            if s1start >= WORK_START:
                chain = [(tp,s1start,s1end, choose_personal_therapist(patient, tp, priorities, eligibility) or (eligibility.get(tp,[""])[0] if eligibility.get(tp) else "")), (tg1,a1,b1,th1), (tg2,a2,b2,th2)]
            else:
                # tp last
                s3start = b2 + RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
                s3end   = s3start + dp
                if s3end <= WORK_END:
                    chain = [(tg1,a1,b1,th1), (tg2,a2,b2,th2), (tp,s3start,s3end, choose_personal_therapist(patient, tp, priorities, eligibility) or (eligibility.get(tp,[""])[0] if eligibility.get(tp) else ""))]
                else:
                    return None
    else:
        # single group
        tg = RNG.choice(group_pool)
        (gs,ge,thg) = RNG.choice(premint_group[tg])
        # pick two distinct personals
        p2 = [t for t in personal_uniq if t.strip().lower()!=tg.strip().lower()]
        if len(p2)<2:
            # broaden: any non-group in eligibility
            universe = [k for k in eligibility.keys() if not is_group(k)]
            RNG.shuffle(universe)
            for t in universe:
                if t not in p2 and t not in personal_uniq:
                    p2.append(t)
                if len(p2)>=2:
                    break
        if len(p2)<2:
            return None
        tp1, tp2 = p2[:2]
        d1 = int(durations.get(tp1, DEFAULT_DUR))
        d2 = int(durations.get(tp2, DEFAULT_DUR))

        # Try [tp1] → [tg] → [tp2] with waits ~3..7
        s1 = gs - d1 - RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
        e1 = s1 + d1
        s3 = ge + RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
        e3 = s3 + d2
        if s1 >= WORK_START and e3 <= WORK_END:
            chain = [
                (tp1,s1,e1, choose_personal_therapist(patient, tp1, priorities, eligibility) or (eligibility.get(tp1,[""])[0] if eligibility.get(tp1) else "")),
                (tg, gs, ge, thg),
                (tp2,s3,e3, choose_personal_therapist(patient, tp2, priorities, eligibility) or (eligibility.get(tp2,[""])[0] if eligibility.get(tp2) else ""))
            ]
        else:
            # fallback: place tp1 after tg then tp2 after tp1
            s1 = ge + RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
            e1 = s1 + d1
            s3 = e1 + RNG.randint(IDEAL_WAIT[0], IDEAL_WAIT[1])
            e3 = s3 + d2
            if e3 <= WORK_END:
                chain = [
                    (tg, gs, ge, thg),
                    (tp1,s1,e1, choose_personal_therapist(patient, tp1, priorities, eligibility) or (eligibility.get(tp1,[""])[0] if eligibility.get(tp1) else "")),
                    (tp2,s3,e3, choose_personal_therapist(patient, tp2, priorities, eligibility) or (eligibility.get(tp2,[""])[0] if eligibility.get(tp2) else ""))
                ]
            else:
                return None

    # validate waits
    chain = [(t, (s//GRID)*GRID, (e//GRID)*GRID, th) for (t,s,e,th) in chain]
    w1 = chain[1][1] - chain[0][2]
    w2 = chain[2][1] - chain[1][2]
    if not (wait_ok(w1) and wait_ok(w2)): return None

    # score (0..1): 70% wait quality, 30% priority for personals only
    wsc = 0.5*(wait_score(w1)+wait_score(w2))
    pr_bonus = 0.0
    n_per = 0
    for (t,_,_,th) in chain:
        if not is_group(t):
            n_per += 1
            tops = priorities.get((_canon(patient), _canon(t)), [])
            if th in tops: pr_bonus += 1.0
            elif tops: pr_bonus += 0.4
            else: pr_bonus += 0.2
    psc = (pr_bonus / max(1,n_per)) if n_per else 0.2
    score = 0.7*wsc + 0.3*(psc)

    slots = [dict(treatment=t, start=s, end=e, therapist=th, origin="AUG") for (t,s,e,th) in chain]
    return slots, float(max(0.0, min(1.0, score)))

# ---------------- Append to graph ----------------
def ensure_core_nodes(nodes_df: pd.DataFrame, day: int, patient: str) -> Tuple[pd.DataFrame, str, str, str]:
    src = f"D{day}_SRC"
    tgt = f"D{day}_TGT"
    pn  = f"D{day}_P|{patient}"
    def has(nid): return not nodes_df[nodes_df["node_id"]==nid].empty
    rows=[]
    if not has(src):
        rows.append(dict(node_id=src, type="SOURCE", day=day))
    if not has(tgt):
        rows.append(dict(node_id=tgt, type="TARGET", day=day))
    if not has(pn):
        rows.append(dict(node_id=pn, type="PATIENT", day=day, patient=patient))
    if rows:
        nodes_df = pd.concat([nodes_df, pd.DataFrame(rows)], ignore_index=True)
    return nodes_df, src, pn, tgt

def make_node_id(day: int, patient: str, idx: int, t: str, s: int, e: int, origin: str) -> str:
    return f"D{day}_P|{patient}_S{idx}|{t}|{m2t(s)}-{m2t(e)}|{origin}"

def append_path(graphs_dir: Path, day: int,
                nodes_df: pd.DataFrame, edges_df: pd.DataFrame, paths_df: pd.DataFrame,
                patient: str, slots: List[Dict[str,Any]], score: float,
                unique_counter: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ensure core nodes
    nodes_df, SRC, PN, TGT = ensure_core_nodes(nodes_df, day, patient)

    # build three slot nodes
    sids=[]
    new_nodes=[]
    for i, sl in enumerate(slots, start=1):
        nid = make_node_id(day, patient, i, _canon(sl["treatment"]), int(sl["start"]), int(sl["end"]), sl.get("origin","AUG"))
        sids.append(nid)
        new_nodes.append(dict(
            node_id=nid, type="SLOT", day=day, patient=patient, slot_idx=i,
            treatment=_canon(sl["treatment"]), start=m2t(sl["start"]), end=m2t(sl["end"]),
            origin=sl.get("origin","AUG"), therapist=_canon(sl.get("therapist",""))
        ))
    nodes_df = pd.concat([nodes_df, pd.DataFrame(new_nodes)], ignore_index=True)

    # new path id
    path_id = f"D{day}|{patient}|AUG|{unique_counter}"
    # edges (SRC->PN->S1->S2->S3->TGT)
    new_edges = [
        dict(from_id=SRC, to_id=PN, path_id=path_id),
        dict(from_id=PN, to_id=sids[0], path_id=path_id),
        dict(from_id=sids[0], to_id=sids[1], path_id=path_id),
        dict(from_id=sids[1], to_id=sids[2], path_id=path_id),
        dict(from_id=sids[2], to_id=TGT, path_id=path_id),
    ]
    edges_df = pd.concat([edges_df, pd.DataFrame(new_edges)], ignore_index=True)

    # path row
    new_path = dict(path_id=path_id, patient=patient, day=day,
                    score=round(float(score),4), origin="AUG",
                    node_ids=";".join([SRC,PN]+sids+[TGT]))
    paths_df = pd.concat([paths_df, pd.DataFrame([new_path])], ignore_index=True)

    return nodes_df, edges_df, paths_df

# ---------------- Main augmentation ----------------
def main():
    ap = argparse.ArgumentParser(description="Augment per-day DAGs with GROUP-containing paths (days 1..4), first 100 patients with group treatments, 200 paths per patient/day.")
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parent), help="Base dir (default: this file's folder)")
    ap.add_argument("--graphs_subdir", default="graphs_out", help="Subfolder with day{d}_*.csv")
    ap.add_argument("--add_per_patient", type=int, default=200, help="Paths to add per patient per day (default: 200)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    graphs_dir = base / args.graphs_subdir
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Load data files
    durations   = load_durations(base / "treatments_durations_opmed.csv")
    eligibility = load_eligibility(base / "therapists_treatments.csv")
    priorities  = load_priorities(base / "patient_treatment_therapist_priorities_opmed.csv")
    pt_map      = load_patients_list(base / "patients_treatments_opmed.csv")
    pt_order    = load_patient_order(base / "patients_treatments_opmed.csv")
    allowed     = load_allowed_days(base / "patients_constraints_opmed.csv")

    # Which patients have group treatments at all? (preserve file order)
    patients_in_order_with_groups: List[str] = []
    for p in pt_order:
        lst = pt_map.get(p, [])
        if any(is_group(t) for t in lst):
            patients_in_order_with_groups.append(p)

    # Limit to first 100 (or fewer if not enough)
    target_patients = patients_in_order_with_groups[:100]
    print(f"[INFO] Group-eligible patients found: {len(patients_in_order_with_groups)}; taking first {len(target_patients)} for augmentation.")

    for day in DAYS_TO_AUGMENT:
        # read graph
        try:
            nodes_df, edges_df, paths_df = read_graph_day(graphs_dir, day)
        except Exception as e:
            print(f"[WARN] Missing/invalid graph for day {day}: {e}")
            continue

        # premint 3 group slots per treatment for this day
        premint = premint_group_slots_for_day(day, durations, eligibility)
        if not premint:
            print(f"[WARN] No preminted group slots for day {day} (check durations/eligibility).")
            continue

        # list of patients allowed this day and within the first-100 subset
        pats_today = [p for p in target_patients if day in allowed.get(p, set())]
        if not pats_today:
            print(f"[INFO] Day {day}: no patients from first-100 subset allowed today.")
            continue

        # start unique counter safely past existing AUG count
        if "origin" in paths_df.columns:
            existing_aug = paths_df[paths_df["origin"].astype(str)=="AUG"]
            counter = len(existing_aug) + 1
        else:
            counter = 1

        total_added = 0

        # generate
        for p in pats_today:
            treats_list = pt_map.get(p, [])
            target = args.add_per_patient
            made = 0
            tries = 0
            # Allow up to ~5x attempts
            while made < target and tries < target * 5:
                # two-group probability lowered a bit for stability
                use_two = (len({t for t in treats_list if is_group(t)}) >= 2) and (RNG.random() < 0.30)
                res = build_augmented_chain(p, day, treats_list, durations, priorities, eligibility, premint, use_two_groups=use_two)
                tries += 1
                if res is None:
                    continue
                slots, sc = res
                # enforce distinct treatments within chain
                if len({_canon(s["treatment"]).lower() for s in slots}) < 3:
                    continue
                # append
                nodes_df, edges_df, paths_df = append_path(graphs_dir, day, nodes_df, edges_df, paths_df, p, slots, sc, counter)
                counter += 1
                made += 1
                total_added += 1
            print(f"[DAY {day}] {p}: added {made}/{target} augmented paths.")

        # write back
        write_graph_day(graphs_dir, day, nodes_df, edges_df, paths_df)
        print(f"[OK] Day {day}: graphs updated. Nodes={len(nodes_df)}, Paths={len(paths_df)}, Added={total_added}")

if __name__ == "__main__":
    main()
