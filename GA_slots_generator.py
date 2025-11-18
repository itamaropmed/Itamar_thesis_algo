# GA_slots_generator.py
# ------------------------------------------------------------
# Feasibility-first weekly scheduler with HARD-ENFORCED GROUP SLOTS.
#
# Hard rules:
#   • Exactly 3 DISTINCT treatments per patient-day
#   • Therapist calendars exclusive (no overlaps)
#   • Group sessions (11,13,14,15,17,18,19,20,24,25,27) only if attendees ∈ [2..8]
#   • Day windows: 1..5: 08:00–20:00, 6: 08:00–14:00
#
# Quality targets:
#   • The two waits inside each chain are in [5..7] (fallback ≤10 only if needed)
#   • High simultaneity across different therapists
#
# Strategy:
#   • Per day, build a tiny anchor catalog for group treatments.
#   • HARD reserve 2–3 patients per anchor and build forced-anchor chains.
#     Commit anchor ONLY if ≥2 chains succeed (so no singleton groups, ever).
#   • Top-up open anchors to ≤8 attendees, then fill with personal-only chains.
#
# Inputs (same folder):
#   patients_treatments_opmed.csv
#   patient_treatment_therapist_priorities_opmed.csv
#   therapists_treatments.csv
#   treatments_durations_opmed.csv
#
# Outputs:
#   schedule_batch_01.csv ... schedule_batch_10.csv
#   batches_ga_schedules.csv
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import pandas as pd
import random, re, math

# ---------------- Config ----------------
BATCH_SIZE = 40
NUM_BATCHES = 10
MAX_BORROWED_PER_BATCH = 60

DAYS = [1,2,3,4,5,6]
WORK_START = 8*60
WORK_END_BY_DAY = {1:20*60,2:20*60,3:20*60,4:20*60,5:20*60,6:14*60}
GRID = 5
BUCKET = 5

DEFAULT_DUR = 45
SEED = 424242

CHAINS_PER_PD_TARGET = 500
GROUP_SHARE_FOR_HAS_GROUP = 0.80   # ~400/500 include group if patient still needs group

GROUP_MIN = 2
GROUP_MAX = 8
GROUP_TARGET_INIT = 3              # try to reserve 3 per anchor first, fallback to 2

WAIT_MIN = 5
WAIT_MAX = 7
WAIT_FALLBACK_MAX = 10
BEST_OF_K = 20

HOT_STARTS = ["09:00","10:00","11:00","13:00","14:00","15:00"]
def _hm(s): h,m = map(int, s.split(":")); return h*60+m
HOT_STARTS_MINS = [_hm(s) for s in HOT_STARTS]

GROUP_T_LIST = [
    "treatment 25","treatment 27","treatment 14","treatment 20","treatment 15",
    "treatment 18","treatment 11","treatment 19","treatment 24","treatment 17","treatment 13"
]
GROUP_T = {s.strip().lower() for s in GROUP_T_LIST}
def is_group(t: str) -> bool: return str(t).strip().lower() in GROUP_T

# ---------------- Small utils ----------------
def split_semis(s: str) -> List[str]:
    if pd.isna(s) or not str(s).strip(): return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def m2t(m: int) -> str:
    m=int(m); return f"{m//60:02d}:{m%60:02d}"

def round_up_to_grid(m: int, g=GRID) -> int:
    return m if m%g==0 else m + (g - (m%g))

def precompute_starts(dur_values: List[int]) -> Dict[int, Dict[int, List[int]]]:
    uniq = sorted({max(5,int(x)) for x in dur_values})
    table = {d:{} for d in DAYS}
    for d in DAYS:
        end = WORK_END_BY_DAY[d]
        for dur in uniq:
            s = round_up_to_grid(WORK_START)
            L=[]
            while s + dur <= end:
                L.append(s); s += GRID
            table[d][dur] = L
    return table

def bucket_range_for(start_min: int, end_min: int, day: int) -> range:
    a = max(WORK_START, start_min)
    b = min(WORK_END_BY_DAY[day], end_min) - 1
    if b < a: return range(0,0)
    base = WORK_START // BUCKET
    return range(a//BUCKET - base, b//BUCKET - base + 1)

# ---------------- Robust durations reader ----------------
def read_duration_map(dur_fp: Path, default: int=DEFAULT_DUR) -> Dict[str,int]:
    df = pd.read_csv(dur_fp, dtype=str)
    norm = {c: re.sub(r"[^a-zA-Zא-ת ]+", "", str(c).strip().lower()) for c in df.columns}
    tcol = dcol = None
    for c,n in norm.items():
        if n in ("treatment","treatments","טיפול"): tcol = c
        if ("duration" in n) or ("minute" in n) or n in ("dur","time","זמן","משך"): dcol = c
    if tcol is None or dcol is None:
        raise ValueError("treatments_durations_opmed.csv must have columns for treatment and duration/minutes.")
    df[tcol] = df[tcol].astype(str).str.strip()
    dser = df[dcol].astype(str).str.extract(r"(\d+)", expand=False).fillna(str(default)).astype(int)
    out={}
    for t,d in zip(df[tcol], dser):
        if str(t).strip():
            out[str(t).strip()] = max(int(d), 5)
    return out

# ---------------- Data structures ----------------
@dataclass(frozen=True)
class Slot:
    treatment: str
    therapist: str
    start: int
    end: int
    is_group: bool

@dataclass(frozen=True)
class Chain:
    slots: Tuple[Slot, Slot, Slot]

# ---------------- Inputs ----------------
def load_inputs(base: Path) -> Dict[str,Any]:
    pt_fp  = base / "patients_treatments_opmed.csv"
    pri_fp = base / "patient_treatment_therapist_priorities_opmed.csv"
    tt_fp  = base / "therapists_treatments.csv"
    dur_fp = base / "treatments_durations_opmed.csv"

    pt = pd.read_csv(pt_fp, dtype=str)
    pt["patient"] = pt["patient"].astype(str).str.strip()
    pt["treatments"] = pt["treatments"].astype(str)

    all_patients = list(pt["patient"])
    demand: Dict[str,Counter] = {}
    for _,r in pt.iterrows():
        demand[r["patient"]] = Counter(split_semis(r["treatments"]))

    pr = pd.read_csv(pri_fp, dtype=str).fillna("")
    for c in ["patient","treatment","priority_1","priority_2","priority_3","all_eligible"]:
        if c in pr.columns: pr[c] = pr[c].astype(str).str.strip()
    prio_by_pt: Dict[Tuple[str,str],List[str]] = {}
    for (p,t), g in pr.groupby(["patient","treatment"]):
        row = g.iloc[0]
        top3 = [row.get("priority_1",""),row.get("priority_2",""),row.get("priority_3","")]
        prio_by_pt[(p,t)] = [x for x in top3 if x]

    tt = pd.read_csv(tt_fp, dtype=str)[["therapist","treatment"]].dropna()
    tt["therapist"] = tt["therapist"].astype(str).str.strip()
    tt["treatment"] = tt["treatment"].astype(str).str.strip()
    tt = tt[(tt["therapist"]!="") & (tt["treatment"]!="")].drop_duplicates()
    elig_by_treatment: Dict[str, List[str]] = tt.groupby("treatment")["therapist"].apply(lambda s: sorted(set(s))).to_dict()

    dur_map = read_duration_map(dur_fp, DEFAULT_DUR)
    starts_table = precompute_starts(list(dur_map.values()))

    return dict(
        demand=demand,
        prio_by_pt=prio_by_pt,
        elig_by_treatment=elig_by_treatment,
        dur_map=dur_map,
        starts_table=starts_table,
        all_patients=all_patients
    )

# ---------------- Round totals to /3 using NON-GROUP fillers ----------------
def round_totals_to_multiple_of3(demand: Dict[str,Counter], elig_by_t: Dict[str,List[str]]):
    for p,cnt in demand.items():
        total = sum(cnt.values())
        r = total % 3
        if r != 0:
            need = 3 - r
            pool = [t for t in cnt if not is_group(t)]
            if not pool:
                pool = [t for t in elig_by_t.keys() if not is_group(t)]
            if not pool:
                pool = list(elig_by_t.keys())
            for i in range(need):
                cnt[pool[i % len(pool)]] += 1

# ---------------- Helpers ----------------
def choose_therapist(p: str, t: str,
                     prio_by_pt: Dict[Tuple[str,str],List[str]],
                     elig_by_t: Dict[str,List[str]],
                     rng: random.Random) -> Optional[str]:
    top3 = [th for th in prio_by_pt.get((p,t), []) if th in elig_by_t.get(t, [])]
    if top3:
        return rng.choice(top3)
    elig = elig_by_t.get(t, [])
    if not elig:
        return None
    return rng.choice(elig)

def can_place_on_therapist(day_busy: Dict[int,List[Tuple[int,int]]],
                           day: int, start: int, end: int) -> bool:
    for a,b in day_busy.get(day, []):
        if not (end <= a or b <= start):
            return False
    return True

def place_on_therapist(day_busy: Dict[int,List[Tuple[int,int]]],
                       day: int, start: int, end: int):
    day_busy.setdefault(day, []).append((start,end))

# ---------------- Group anchor catalog ----------------
def build_group_anchor_catalog_for_day(d: int,
                                       elig_by_t: Dict[str,List[str]],
                                       dur_map: Dict[str,int],
                                       rng: random.Random,
                                       anchors_per_treatment: int = 2) -> Dict[str, List[Tuple[str,int,int]]]:
    anchors = {}
    endw = WORK_END_BY_DAY[d]
    for t, ths in elig_by_t.items():
        if not is_group(t) or not ths:
            continue
        dur = int(dur_map.get(t, DEFAULT_DUR))
        starts = []
        for hs in HOT_STARTS_MINS:
            s = round_up_to_grid(hs)
            if WORK_START <= s and s + dur <= endw:
                starts.append(s)
        if len(starts) < anchors_per_treatment:
            s = round_up_to_grid(WORK_START)
            pool=[]
            while s + dur <= endw:
                pool.append(s); s += GRID
            rng.shuffle(pool)
            for s in pool:
                if len(starts) >= anchors_per_treatment: break
                if s not in starts: starts.append(s)
        rng.shuffle(ths)
        picks=[]
        for i,s in enumerate(starts[:anchors_per_treatment]):
            th = ths[i % len(ths)]
            picks.append((th, s, s + dur))
        if picks:
            anchors[t] = picks
    return anchors

# ---------------- Chain builders ----------------
@dataclass(frozen=True)
class _Local:
    th_busy: Dict[str,Dict[int,List[Tuple[int,int]]]]

def _try_personal_at(p: str, t: str, th: str, d: int, start: int, dur: int,
                     th_busy_local: Dict[str,Dict[int,List[Tuple[int,int]]]]) -> Optional[Slot]:
    if start < WORK_START or start + dur > WORK_END_BY_DAY[d]:
        return None
    if not can_place_on_therapist(th_busy_local.setdefault(th, defaultdict(list)), d, start, start+dur):
        return None
    return Slot(t, th, start, start+dur, is_group=False)

def build_chain_forced_anchor(p: str, d: int,
                              anchor: Tuple[str,str,int,int],  # (t,th,st,en)
                              remaining_p: Counter,
                              prio_by_pt, elig_by_t, dur_map,
                              rng: random.Random,
                              wait_ceiling: int,
                              th_busy_local: Dict[str,Dict[int,List[Tuple[int,int]]]]
                              ) -> Optional[Chain]:
    t_g, th_g, st_g, en_g = anchor

    # Ensure the anchor fits locally
    if not can_place_on_therapist(th_busy_local.setdefault(th_g, defaultdict(list)), d, st_g, en_g):
        return None
    gslot = Slot(t_g, th_g, st_g, en_g, is_group=True)

    def pick_before(used:set) -> Optional[Slot]:
        pool = [t for t,c in remaining_p.items() if c>0 and not is_group(t) and t not in used] \
               or [t for t in elig_by_t.keys() if not is_group(t) and t not in used]
        random.shuffle(pool)
        for t in pool:
            th = choose_therapist(p, t, prio_by_pt, elig_by_t, rng)
            if not th: continue
            dur = int(dur_map.get(t, DEFAULT_DUR))
            for gap in random.sample(list(range(WAIT_MIN, wait_ceiling+1)), k=min(1, wait_ceiling-WAIT_MIN+1)):
                st = st_g - dur - gap
                sl = _try_personal_at(p,t,th,d,st,dur,th_busy_local)
                if sl: return sl
        return None

    def pick_after(used:set) -> Optional[Slot]:
        pool = [t for t,c in remaining_p.items() if c>0 and not is_group(t) and t not in used] \
               or [t for t in elig_by_t.keys() if not is_group(t) and t not in used]
        random.shuffle(pool)
        for t in pool:
            th = choose_therapist(p, t, prio_by_pt, elig_by_t, rng)
            if not th: continue
            dur = int(dur_map.get(t, DEFAULT_DUR))
            for gap in random.sample(list(range(WAIT_MIN, wait_ceiling+1)), k=min(1, wait_ceiling-WAIT_MIN+1)):
                st = en_g + gap
                sl = _try_personal_at(p,t,th,d,st,dur,th_busy_local)
                if sl: return sl
        return None

    # Try patterns with strict waits; distinct treatments guaranteed by construction
    used = {t_g}
    s1 = pick_before(used)
    if s1:
        used.add(s1.treatment)
        s3 = pick_after(used)
        if s3:
            ch = sorted([s1,gslot,s3], key=lambda x:x.start)
            if len({s.treatment for s in ch}) == 3:
                return Chain(slots=(ch[0],ch[1],ch[2]))

    used = {t_g}
    s2 = pick_after(used)
    if s2:
        used.add(s2.treatment)
        s3 = pick_after(used)
        if s3:
            ch = sorted([gslot,s2,s3], key=lambda x:x.start)
            if len({s.treatment for s in ch}) == 3:
                return Chain(slots=(ch[0],ch[1],ch[2]))

    used = {t_g}
    s1 = pick_before(used)
    if s1:
        used.add(s1.treatment)
        s2 = pick_before(used)
        if s2:
            ch = sorted([s1,s2,gslot], key=lambda x:x.start)
            if len({s.treatment for s in ch}) == 3:
                return Chain(slots=(ch[0],ch[1],ch[2]))

    return None

def build_500_chain_pool_for_pd(p: str, d: int,
                                remaining: Counter,
                                prio_by_pt, elig_by_t, dur_map, starts_table,
                                rng: random.Random,
                                want_total: int,
                                group_share: float,
                                anchors_for_day: Dict[str,List[Tuple[str,int,int]]]
                                ) -> List[Chain]:
    rem_group = [t for t,c in remaining.items() if c>0 and is_group(t)]
    rem_non   = [t for t,c in remaining.items() if c>0 and not is_group(t)]
    fillers_non = [t for t in elig_by_t.keys() if not is_group(t)]
    if not rem_non and fillers_non:
        rem_non = fillers_non[:]

    def try_personal(t, after_end: Optional[int], wait_max: int) -> Optional[Slot]:
        th = choose_therapist(p, t, prio_by_pt, elig_by_t, rng)
        if th is None: return None
        dur = int(dur_map.get(t, DEFAULT_DUR))
        endw = WORK_END_BY_DAY[d]
        if after_end is None:
            starts = starts_table[d].get(dur, [])
            if not starts: return None
            s = random.choice(starts)
            if s + dur > endw: return None
            return Slot(t, th, s, s + dur, is_group=False)
        s_low = round_up_to_grid(after_end + WAIT_MIN)
        s_high = after_end + wait_max
        if s_low + dur <= endw and s_low <= s_high:
            return Slot(t, th, s_low, s_low + dur, is_group=False)
        return None

    def try_group(after_end: Optional[int], wait_max: int) -> Optional[Slot]:
        if not anchors_for_day: return None
        pool_t = rem_group[:] if rem_group else list(anchors_for_day.keys())
        random.shuffle(pool_t)
        for t in pool_t:
            if t not in anchors_for_day: continue
            for (th,s,e) in random.sample(anchors_for_day[t], k=len(anchors_for_day[t])):
                if after_end is None:
                    return Slot(t, th, s, e, is_group=True)
                if s >= after_end + WAIT_MIN and s <= after_end + wait_max and e <= WORK_END_BY_DAY[d]:
                    return Slot(t, th, s, e, is_group=True)
        return None

    def build_one_chain(wait_max: int) -> Optional[Chain]:
        include_group = (group_share > 0) and (random.random() < group_share)
        parts=[]; used=set()

        s1=None
        if include_group:
            s1 = try_group(after_end=None, wait_max=wait_max)
        if s1 is None:
            pool = rem_non[:] if rem_non else fillers_non[:]
            if not pool: return None
            random.shuffle(pool)
            for t in pool:
                s1 = try_personal(t, after_end=None, wait_max=wait_max)
                if s1: break
        if s1 is None: return None
        parts.append(s1); used.add(s1.treatment)

        s2=None
        if include_group and not s1.is_group:
            s2 = try_group(after_end=s1.end, wait_max=wait_max)
        if s2 is None:
            pool = [t for t in rem_non if t not in used] or [t for t in fillers_non if t not in used]
            random.shuffle(pool)
            for t in pool:
                cand = try_personal(t, after_end=s1.end, wait_max=wait_max)
                if cand:
                    s2=cand; break
        if s2 is None: return None
        parts.append(s2); used.add(s2.treatment)

        s3=None
        if include_group and not any(s.is_group for s in parts):
            s3 = try_group(after_end=s2.end, wait_max=wait_max)
        if s3 is None:
            pool = [t for t in rem_non if t not in used] or [t for t in fillers_non if t not in used]
            random.shuffle(pool)
            for t in pool:
                cand = try_personal(t, after_end=s2.end, wait_max=wait_max)
                if cand:
                    s3=cand; break
        if s3 is None: return None

        chain = sorted([parts[0], s2, s3], key=lambda x:x.start)
        if len({s.treatment for s in chain}) != 3:
            return None
        return Chain(slots=(chain[0], chain[1], chain[2]))

    chains=[]
    seen=set()
    for wait_ceiling in (WAIT_MAX, WAIT_FALLBACK_MAX):
        guard = 0
        max_guard = want_total * 60
        while len(chains) < want_total and guard < max_guard:
            guard += 1
            ch = build_one_chain(wait_max=wait_ceiling)
            if not ch: continue
            key = tuple((s.treatment,s.therapist,s.start,s.end) for s in ch.slots)
            if key in seen: continue
            seen.add(key); chains.append(ch)
        if len(chains) >= want_total:
            break
    if len(chains) > want_total:
        chains = random.sample(chains, want_total)
    return chains

# ---------------- Day scheduler with HARD group enforcement ----------------
def schedule_day_greedy(d: int,
                        todays: List[str],
                        remaining_by_patient: Dict[str,Counter],
                        prio_by_pt, elig_by_t, dur_map, starts_table,
                        rng: random.Random):
    anchors_for_day = build_group_anchor_catalog_for_day(d, elig_by_t, dur_map, rng)

    th_busy: Dict[str, Dict[int, List[Tuple[int,int]]]] = defaultdict(lambda: defaultdict(list))
    grp_count = Counter()
    bucket_load = [0]*((WORK_END_BY_DAY[d] - WORK_START)//BUCKET)

    rows=[]; scheduled=set()

    # ---- Step 2: HARD reserve & build forced-anchor chains; commit if ≥2 ----
    need_by_t = defaultdict(list)
    for p in todays:
        for t,c in remaining_by_patient[p].items():
            if c>0 and is_group(t):
                need_by_t[t].append(p)

    reserved = defaultdict(list)  # anchor_key -> [patients]
    for t, plist in need_by_t.items():
        if t not in anchors_for_day or len(plist) < GROUP_MIN:
            continue
        rng.shuffle(plist)
        anchors = anchors_for_day[t][:]
        rng.shuffle(anchors)
        num = max(1, min(len(plist)//GROUP_MIN, len(anchors), 2))
        take = 0; i = 0
        while take < num and i < len(anchors):
            (th, st, en) = anchors[i]; i+=1
            k = GROUP_TARGET_INIT if len(plist) >= GROUP_TARGET_INIT else GROUP_MIN
            picked = plist[:k]; plist = plist[k:]
            reserved[(t,th,st,en)].extend(picked)
            take += 1

    def bucket_add(st,en):
        for b in bucket_range_for(st,en,d):
            bucket_load[b] += 1

    def commit_chain(p: str, ch: Chain):
        for s in ch.slots:
            place_on_therapist(th_busy[s.therapist], d, s.start, s.end)
            bucket_add(s.start,s.end)
        for i,s in enumerate(sorted(ch.slots, key=lambda x:x.start), start=1):
            rows.append(dict(patient=p, day=d, slot_idx=i,
                             treatment=s.treatment, therapist=s.therapist,
                             start=m2t(s.start), end=m2t(s.end)))
        for s in ch.slots:
            if remaining_by_patient[p].get(s.treatment,0)>0:
                remaining_by_patient[p][s.treatment] -= 1
                if remaining_by_patient[p][s.treatment]==0:
                    del remaining_by_patient[p][s.treatment]
            if s.is_group:
                grp_count[(s.treatment,s.therapist,s.start,s.end)] += 1
        scheduled.add(p)

    # Build on local calendars; commit only if >=2 succeed
    for key, plist in reserved.items():
        t, th, st, en = key
        local_busy = {k: defaultdict(list, {dd:list(iv) for dd,iv in v.items()}) for k,v in th_busy.items()}
        built = []
        for wait_cap in (WAIT_MAX, WAIT_FALLBACK_MAX):
            for p in plist:
                if p in scheduled: continue
                ch = build_chain_forced_anchor(p, d, key, remaining_by_patient[p],
                                               prio_by_pt, elig_by_t, dur_map, rng,
                                               wait_cap, local_busy)
                if not ch: continue
                ok=True
                for s in ch.slots:
                    if not can_place_on_therapist(local_busy.setdefault(s.therapist, defaultdict(list)), d, s.start, s.end):
                        ok=False; break
                    local_busy[s.therapist][d].append((s.start,s.end))
                if ok: built.append((p,ch))
            if len(built) >= GROUP_MIN:
                break
        if len(built) >= GROUP_MIN:
            for p,ch in built[:GROUP_MAX]:
                commit_chain(p, ch)

    # ---- Step 3: general chain pools (for remaining patients) ----
    chains_by_patient={}
    for p in todays:
        if p in scheduled: continue
        has_grp = any(is_group(t) and c>0 for t,c in remaining_by_patient[p].items())
        share = GROUP_SHARE_FOR_HAS_GROUP if has_grp else 0.0
        chains_by_patient[p] = build_500_chain_pool_for_pd(
            p, d, remaining_by_patient[p],
            prio_by_pt, elig_by_t, dur_map, starts_table,
            rng, want_total=CHAINS_PER_PD_TARGET, group_share=share,
            anchors_for_day=anchors_for_day
        )

    def chain_fits(ch: Chain) -> bool:
        for s in ch.slots:
            if s.start < WORK_START or s.end > WORK_END_BY_DAY[d]:
                return False
            if not can_place_on_therapist(th_busy[s.therapist], d, s.start, s.end):
                return False
            if s.is_group and grp_count[(s.treatment,s.therapist,s.start,s.end)] >= GROUP_MAX:
                return False
        return True

    def chain_score(ch: Chain) -> int:
        sc=0
        for s in ch.slots:
            for b in bucket_range_for(s.start,s.end,d):
                sc += bucket_load[b]
        return sc

    # Top-up attendees on already-open anchors
    for p, pool in list(chains_by_patient.items()):
        if p in scheduled: continue
        random.shuffle(pool)
        best=None; best_sc=-1; tried=0
        for ch in pool:
            anchors = [(s.treatment,s.therapist,s.start,s.end) for s in ch.slots if s.is_group]
            if not anchors or all(grp_count[a]==0 for a in anchors):  # only add to open anchors
                continue
            if not chain_fits(ch): continue
            sc = chain_score(ch)
            if sc > best_sc: best_sc, best = sc, ch
            tried += 1
            if tried >= BEST_OF_K: break
        if best is not None:
            commit_chain(p, best)

    # Fill remainder with personal-only
    for p, pool in chains_by_patient.items():
        if p in scheduled: continue
        personal = [ch for ch in pool if all(not s.is_group for s in ch.slots)]
        random.shuffle(personal)
        best=None; best_sc=-1; tried=0
        for ch in personal:
            if tried >= BEST_OF_K: break
            if not chain_fits(ch): continue
            sc = chain_score(ch)
            if sc > best_sc: best_sc, best = sc, ch
            tried += 1
        if best is not None:
            commit_chain(p, best)

    # Defensive: drop any accidental singletons (should not happen)
    grp_att = Counter()
    for r in rows:
        t,th,st,en = r["treatment"], r["therapist"], r["start"], r["end"]
        if is_group(t):
            grp_att[(t,th,st,en)] += 1
    bad = [k for k,c in grp_att.items() if c < GROUP_MIN or c > GROUP_MAX]
    if bad:
        bad_patients = set()
        for r in rows:
            key = (r["treatment"],r["therapist"],r["start"],r["end"])
            if key in bad:
                bad_patients.add(r["patient"])
        rows = [r for r in rows if r["patient"] not in bad_patients]

    return rows, set(r["patient"] for r in rows)

# ---------------- Batch helpers ----------------
def build_batches_in_order(all_patients: List[str], batch_size:int, num_batches:int)->List[List[str]]:
    batches=[]; i=0
    for _ in range(num_batches):
        chunk = all_patients[i:i+batch_size]
        if not chunk: break
        batches.append(list(chunk)); i += batch_size
    return batches

def extend_batch_with_group_patients(base_batch: List[str],
                                     unused_patients: List[str],
                                     demand: Dict[str,Counter],
                                     max_borrow:int,
                                     rng: random.Random) -> List[str]:
    batch = list(base_batch)
    remaining_pool = [p for p in unused_patients if p not in batch]
    counts = Counter()
    for p in batch:
        for t,k in demand[p].items():
            if is_group(t) and k>0:
                counts[t] += k
    need_more = [t for t,c in counts.items() if c==1]
    borrowed=0
    for t in need_more:
        cand = [p for p in remaining_pool if demand[p].get(t,0)>0]
        rng.shuffle(cand)
        for q in cand:
            batch.append(q); remaining_pool.remove(q)
            borrowed += 1
            counts[t] += demand[q][t]
            if counts[t] >= 2: break
        if borrowed >= max_borrow: break
    return batch

# ---------------- Solve one batch week ----------------
def solve_batch_week(batch_id:int,
                     batch_patients: List[str],
                     demand_all: Dict[str,Counter],
                     prio_by_pt, elig_by_t, dur_map, starts_table,
                     rng: random.Random) -> pd.DataFrame:
    remaining = {p: Counter(demand_all[p]) for p in batch_patients}
    days_needed = {p: max(1, min(6, math.ceil(sum(remaining[p].values())/3))) for p in batch_patients}
    days_used   = {p: 0 for p in batch_patients}

    schedule_rows=[]

    for d in DAYS:
        todays = [p for p in batch_patients if days_used[p] < days_needed[p] and sum(remaining[p].values())>0]
        if not todays: break

        rows_day, placed = schedule_day_greedy(
            d, todays, remaining,
            prio_by_pt, elig_by_t, dur_map, starts_table, rng
        )
        if not rows_day: continue

        per_p = defaultdict(list)
        for r in rows_day: per_p[r["patient"]].append(r)
        for p, L in per_p.items():
            if len(L) == 3: days_used[p] += 1
        for r in rows_day:
            schedule_rows.append(dict(
                batch=f"BATCH-{batch_id:02d}",
                solution=f"BATCH-{batch_id:02d}",
                patient=r["patient"], day=r["day"], slot_idx=r["slot_idx"],
                treatment=r["treatment"], therapist=r["therapist"],
                start=r["start"], end=r["end"]
            ))

    return pd.DataFrame(schedule_rows).sort_values(["batch","patient","day","slot_idx"]).reset_index(drop=True)

# ---------------- Main ----------------
def main():
    here = Path(__file__).resolve().parent
    random.seed(SEED); rng = random.Random(SEED)

    data = load_inputs(here)
    demand_all   = data["demand"]
    prio_by_pt   = data["prio_by_pt"]
    elig_by_t    = data["elig_by_treatment"]
    dur_map      = data["dur_map"]
    starts_table = data["starts_table"]
    all_patients = data["all_patients"]

    round_totals_to_multiple_of3(demand_all, elig_by_t)

    base_batches = build_batches_in_order(all_patients, BATCH_SIZE, NUM_BATCHES)

    used=set(); combined=[]
    for b_idx, base in enumerate(base_batches, start=1):
        base = [p for p in base if p not in used]
        unused = [p for p in all_patients if p not in used and p not in base]
        batch = extend_batch_with_group_patients(base, unused, demand_all, MAX_BORROWED_PER_BATCH, rng)
        for p in batch: used.add(p)

        print(f"[BATCH {b_idx}] Patients: {len(batch)} (base {len(base)}, borrowed {len(batch)-len(base)})")

        df = solve_batch_week(b_idx, batch, demand_all, prio_by_pt, elig_by_t, dur_map, starts_table, rng)
        out_batch = here / f"schedule_batch_{b_idx:02d}.csv"
        df.to_csv(out_batch, index=False)
        print(f"[OK] Wrote {out_batch.name}  ({len(df)} rows, patients={df['patient'].nunique()})")
        combined.append(df)

        if b_idx >= NUM_BATCHES:
            break

    if combined:
        big = pd.concat(combined, ignore_index=True)
        big_out = here / "batches_ga_schedules.csv"
        big.to_csv(big_out, index=False)
        print(f"[OK] Wrote batches_ga_schedules.csv  ({len(big)} rows, unique patients={big['patient'].nunique()})")
    else:
        print("[FAIL] No schedules produced — check inputs or relax pairing.")

if __name__ == "__main__":
    main()
