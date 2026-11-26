#!/usr/bin/env python3
"""
AT-RISK INTELLIGENCE  -  Predict • Detect • Act
A Python (Streamlit + Plotly) version of the HTML analytics dashboard.

Setup:   pip install "streamlit>=1.37" plotly pandas numpy openpyxl
Run:     streamlit run at_risk_intelligence.py

DEMONSTRATION ONLY: the risk score is a weighted heuristic on synthetic (or your uploaded)
data. It is NOT a scientifically validated prediction. Use it to support, never replace,
human judgement. Uploaded files are processed locally on the machine running this script.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import io
import json
import re
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

st = None   # streamlit, imported in run_app() so the logic below can be imported and tested without it
go = None   # plotly.graph_objects, same reason

# =====================================================================================
# 1. CONSTANTS & RISK MODEL
# =====================================================================================
LEVELS = [("Low", "#34d399"), ("Moderate", "#fbbf24"), ("High", "#fb923c"), ("Critical", "#f87171")]
DEFAULT_TH = [30, 60, 80]                       # lower bound of Moderate / High / Critical
PRIORITY = ["Monitor", "Monitor", "This week", "Immediate"]
STATUSES = ["New", "Contacted", "In progress", "Resolved"]
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
SCALE = {"edu": 1.25, "corp": 1.35}             # scales the weighted sum to span 0-100
SALARY_COST_LAKH = [4, 6, 9, 14, 20]            # illustrative replacement-cost base per salary band
STATE_FILE = Path(__file__).with_name(".ari_state.json") if "__file__" in globals() else Path(".ari_state.json")
DATA_FILE = STATE_FILE.with_name(".ari_dataset.pkl")


class Factor(NamedTuple):
    name: str
    weight: float
    fn: Callable          # DataFrame -> Series of raw severity (NaN where the inputs are missing)
    action: str


def _fall(d):
    return (d["perf_t1"] - d["perf_t6"]) * 6


def _tenure(d):
    v = np.where(d["ten"] < 1, 45.0, np.where(d["since"] >= 4, 60.0, 10.0))
    return pd.Series(np.where(d["ten"].isna(), np.nan, v), index=d.index)


FACTORS = {
    "edu": [
        Factor("Low attendance", .22, lambda d: (90 - d["att"]) * 2.2, "Attendance follow-up"),
        Factor("Low academic scores", .24, lambda d: (75 - d["perf"]) * 2.4, "Subject tutoring"),
        Factor("Low engagement", .14, lambda d: (70 - d["eng"]) * 2, "Faculty meeting"),
        Factor("Assignment backlog", .12, lambda d: d["back"] * 12, "Assignment support"),
        Factor("Failed subjects", .14, lambda d: d["fail"] * 25, "Academic counseling"),
        Factor("Falling grades", .14, _fall, "Academic counseling"),
    ],
    "corp": [
        Factor("Low performance", .22, lambda d: (78 - d["perf"]) * 2.4, "Training & coaching plan"),
        Factor("Poor attendance", .12, lambda d: (92 - d["att"]) * 2.5, "Manager check-in"),
        Factor("Low engagement", .22, lambda d: (70 - d["eng"]) * 2.2, "Engagement follow-up"),
        Factor("Overtime / workload", .16, lambda d: d["ot"] * 5, "Workload review"),
        Factor("Falling performance", .16, _fall, "Manager check-in"),
        Factor("Tenure / role factors", .12, _tenure, "Career development discussion"),
    ],
}
DEFAULT_WEIGHTS = {m: [f.weight for f in fs] for m, fs in FACTORS.items()}
DIMS = {"edu": [("grp", "Course"), ("year", "Year"), ("sem", "Semester")],
        "corp": [("grp", "Department"), ("role", "Role"), ("loc", "Location")]}
NUMERIC = {
    "edu": ["att", "perf", "eng", "gpa", "assign", "back", "fail", "late", "missed", "lms", "study", "lib",
            "subj_Mathematics", "subj_Core theory", "subj_Laboratory", "subj_Elective"],
    "corp": ["att", "perf", "eng", "prod", "hrs", "ot", "ten", "since", "promos", "band", "goal", "proj", "qual",
             "sat", "train", "tasks", "absd"],
}
TREND_COLS = [f"{p}_t{i}" for p in ("perf", "att") for i in range(1, 7)]


def level_of(risk, th=DEFAULT_TH):
    r = np.asarray(risk, dtype=float)
    return np.select([r >= th[2], r >= th[1], r >= th[0]], [3, 2, 1], 0)


def ensure_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Guarantee every column the model/UI reads exists (NaN when the data does not have it)."""
    df = df.copy()
    for c in NUMERIC[mode] + TREND_COLS:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("id", "name", "grp", "year", "sem", "role", "loc", "band_t"):
        if c not in df:
            df[c] = "N/A"
    if "crit" not in df:
        df["crit"] = False
    if "risk_in" not in df:
        df["risk_in"] = np.nan
    return df


def score_frame(df: pd.DataFrame, mode: str, weights=None, th=None) -> pd.DataFrame:
    """Compute risk (0-100), level, factor contributions, main issue and suggested action.
    Missing inputs are skipped and the remaining weights are re-balanced, never treated as zero."""
    th = th or DEFAULT_TH
    fs = FACTORS[mode]
    df = ensure_columns(df, mode)
    w = np.array(weights if weights is not None else DEFAULT_WEIGHTS[mode], dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    comp = np.column_stack([pd.Series(f.fn(df), index=df.index).clip(0, 100).to_numpy(float) for f in fs]) \
        if len(df) else np.zeros((0, len(fs)))
    have = ~np.isnan(comp)
    wa = (have * w).sum(axis=1)
    scale = np.where(wa > 0, 1.0 / np.where(wa > 0, wa, 1), 0.0)
    contrib = np.nan_to_num(comp) * w * SCALE[mode] * scale[:, None]
    risk = np.clip(contrib.sum(axis=1), 0, 100).round()
    risk_in = df["risk_in"].to_numpy(float)
    risk = np.where(np.isnan(risk_in), risk, np.clip(risk_in, 0, 100).round())
    out = df.copy()
    for i in range(len(fs)):
        out[f"c{i}"] = contrib[:, i]
    out["risk"] = risk
    out["lv"] = level_of(risk, th)
    top = contrib.argmax(axis=1) if len(df) else np.array([], dtype=int)
    topv = contrib.max(axis=1) if len(df) else np.array([])
    names = np.array([f.name for f in fs])
    acts = np.array([f.action for f in fs])
    out["main"] = np.where(topv >= 2, names[top] if len(df) else "", "No significant factors")
    out["action"] = np.where((risk < th[0]) | (topv < 2), "Routine monitoring", acts[top] if len(df) else "")
    out["priority"] = np.array(PRIORITY)[out["lv"].to_numpy(int)] if len(df) else []
    return out


# =====================================================================================
# 2. SYNTHETIC DATA
# =====================================================================================
FIRST = "Aarav Diya Rohan Meera Kabir Ananya Vikram Isha Arjun Naina Dev Sara Kunal Tara Neil Riya Yash Zoya Farhan Pooja Sameer Leela Ishaan Mira".split()
LAST = "Sharma Verma Iyer Khan Reddy Das Mehta Nair Gupta Singh Bose Kapoor Joshi Rao Malik Sinha Pillai Chopra Menon Saxena Bhatt Jain Ghosh Patel".split()
EDU_COURSES = ["Engineering", "Computer Science", "Business", "Biology", "Arts", "Mathematics"]
DEPTS = ["Engineering", "Sales", "Operations", "Finance", "Customer Support", "Marketing", "HR"]
ROLES = ["Associate", "Analyst", "Specialist", "Senior", "Manager"]
LOCATIONS = ["Mumbai", "Bengaluru", "Delhi", "Remote"]


def person_name(i: int) -> str:
    return f"{FIRST[i % 24]} {LAST[(i * 7 + i // 24) % 24]}"


def _trend(rng, base, slope, noise):
    k = (5 - np.arange(6))[None, :]
    return np.clip(base[:, None] + k * slope[:, None] + rng.uniform(-noise, noise, (len(base), 6)), 0, 100).round()


def generate_demo(seed: int = 42, weights=None, th=None) -> dict:
    """Realistic synthetic data: hidden 'struggle' level drives attendance, scores, engagement, etc."""
    rng = np.random.RandomState(seed)
    n = 120

    def u(k):
        return rng.uniform(-k, k, n)

    # ---- students ----
    course = rng.choice(EDU_COURSES, n)
    yr = rng.randint(1, 5, n)
    s = np.clip(rng.rand(n) ** 1.6 + (course == "Engineering") * .12 + (yr == 1) * .05, 0, 1)
    att = np.clip(96 - s * 45 + u(5), 0, 100).round()
    perf = np.clip(88 - s * 50 + u(6), 0, 100).round()
    eng = np.clip(85 - s * 50 + u(8), 0, 100).round()
    back = np.clip(np.round(s * 5 + u(.8)), 0, 6)
    fail = np.where(s > .55, np.clip(np.round((s - .45) * 6), 0, 4), 0)
    edu = pd.DataFrame({
        "id": [f"S-{1001 + i}" for i in range(n)], "name": [person_name(i) for i in range(n)], "grp": course,
        "year": [f"Year {y}" for y in yr], "sem": [f"Sem {y * 2 - int(rng.rand() < .5)}" for y in yr],
        "att": att, "perf": perf, "eng": eng, "back": back, "fail": fail, "gpa": (perf / 25).round(2),
        "late": np.round(s * 9 + rng.rand(n) * 3), "missed": np.round((100 - att) / 100 * 80),
        "assign": np.clip(100 - back * 14 + u(4), 0, 100).round(), "lms": np.clip(eng + u(8), 0, 100).round(),
        "study": np.round(1 + (1 - s) * 3 + u(.5), 1), "lib": np.round((1 - s) * 6 + rng.rand(n) * 2)})
    for sub in ("Mathematics", "Core theory", "Laboratory", "Elective"):
        edu[f"subj_{sub}"] = np.clip(perf + u(10), 0, 100).round()
    pt, at = _trend(rng, perf, (s - .3) * 3, 2), _trend(rng, att, s * 1.5, 1.5)
    for i in range(6):
        edu[f"perf_t{i + 1}"], edu[f"att_t{i + 1}"] = pt[:, i], at[:, i]

    # ---- employees ----
    dept = rng.choice(DEPTS, n)
    s = np.clip(rng.rand(n) ** 1.6 + (dept == "Customer Support") * .1 + (dept == "Engineering") * .06, 0, 1)
    ri = rng.randint(0, 5, n)
    ten = np.round(rng.rand(n) * 8 + .3, 1)
    perf = np.clip(86 - s * 45 + u(5), 0, 100).round()
    att = np.clip(97 - s * 22 + u(3), 0, 100).round()
    eng = np.clip(82 - s * 50 + u(7), 0, 100).round()
    hrs = np.round(40 + s * 14 + u(2), 1)
    corp = pd.DataFrame({
        "id": [f"E-{2001 + i}" for i in range(n)], "name": [person_name(i + 31) for i in range(n)], "grp": dept,
        "role": [ROLES[i] for i in ri], "loc": rng.choice(LOCATIONS, n), "ten": ten, "band": ri + 1, "crit": ri >= 3,
        "since": np.round(rng.rand(n) * np.minimum(ten, 6), 1), "promos": np.floor(ten / 2.5),
        "band_t": np.select([ten < 1, ten < 3, ten < 5], ["Under 1 yr", "1–3 yrs", "3–5 yrs"], "5+ yrs"),
        "perf": perf, "att": att, "eng": eng, "hrs": hrs, "ot": np.maximum(0, hrs - 40),
        "prod": np.clip(perf - s * 8 + u(4), 0, 100).round(), "goal": np.clip(perf + u(8), 0, 100).round(),
        "proj": np.clip(perf + u(8), 0, 100).round(), "qual": np.clip(perf + u(6), 0, 100).round(),
        "sat": np.clip(eng + u(6), 0, 100).round(), "train": np.clip(75 - s * 40 + u(10), 0, 100).round(),
        "tasks": np.clip(100 - s * 35 + u(5), 0, 100).round(), "absd": np.round((100 - att) / 100 * 220)})
    pt, at = _trend(rng, perf, (s - .3) * 3, 2), _trend(rng, att, s * .8, 1.2)
    for i in range(6):
        corp[f"perf_t{i + 1}"], corp[f"att_t{i + 1}"] = pt[:, i], at[:, i]
    w = weights or DEFAULT_WEIGHTS
    return {"edu": score_frame(edu, "edu", w["edu"], th), "corp": score_frame(corp, "corp", w["corp"], th)}


# =====================================================================================
# 3. CSV / EXCEL IMPORT: alias mapping, validation, missing-value handling
# =====================================================================================
_A = {
    "att": ["attendance", "attendancepercentage", "attendancepercent", "presentpercentage", "attendancerate"],
    "perf": ["performance", "performancescore", "academicscore", "examscore", "overallscore", "score", "marks"],
    "eng": ["engagement", "engagementscore", "participation", "engagementpercentage"],
    "risk": ["riskscore", "risk"], "lvl": ["risklevel", "riskcategory"],
}
FIELDS = {
    "edu": [("name", "Name", ["name", "studentname", "fullname", "personname", "student"]),
            ("id", "Student ID", ["id", "studentid", "rollno", "rollnumber", "enrollmentno"]),
            ("grp", "Course", ["course", "program", "programme", "branch", "stream", "department", "dept", "division"]),
            ("year", "Year", ["year", "academicyear", "studyyear"]), ("sem", "Semester", ["semester", "sem"]),
            ("att", "Attendance", _A["att"]), ("perf", "Performance", _A["perf"]),
            ("gpa", "GPA", ["gpa", "cgpa", "gradepointaverage"]), ("eng", "Engagement", _A["eng"]),
            ("assign", "Assignment completion", ["assignmentcompletion", "assignments", "assignmentscore", "assignmentcompletionrate"]),
            ("back", "Assignment backlog", ["assignmentbacklog", "pendingassignments"]),
            ("fail", "Failed subjects", ["failedsubjects", "failures", "backlogs", "arrears"]),
            ("late", "Late arrivals", ["latearrivals", "lates"]), ("missed", "Classes missed", ["classesmissed", "absences"]),
            ("risk", "Risk score", _A["risk"]), ("lvl", "Risk level", _A["lvl"])],
    "corp": [("name", "Name", ["name", "employeename", "fullname", "personname", "employee"]),
             ("id", "Employee ID", ["id", "employeeid", "empid", "employeenumber"]),
             ("grp", "Department", ["department", "dept", "division", "team"]),
             ("role", "Role", ["role", "jobtitle", "designation", "position", "jobrole"]),
             ("loc", "Location", ["location", "city", "office"]), ("att", "Attendance", _A["att"]),
             ("perf", "Performance", _A["perf"]), ("eng", "Engagement", _A["eng"]),
             ("prod", "Productivity", ["productivity", "productivityscore", "productivityindex"]),
             ("hrs", "Working hours / week", ["workinghours", "hoursperweek", "weeklyhours", "workload", "hours"]),
             ("ot", "Overtime hours", ["overtime", "overtimehours"]),
             ("ten", "Tenure (years)", ["tenure", "tenureyears", "yearsatcompany", "yearsofservice"]),
             ("band", "Salary band (1-5)", ["salaryband", "band", "paygrade"]),
             ("goal", "Goal completion", ["goalcompletion", "goals"]),
             ("risk", "Risk score", _A["risk"]), ("lvl", "Risk level", _A["lvl"])],
}
METRICS = ["att", "perf", "eng", "gpa", "prod"]
PCT_FIELDS = ["att", "perf", "eng", "assign", "prod", "goal", "risk"]
LEVEL_TEXT = {"low": 15, "moderate": 45, "medium": 45, "high": 70, "critical": 90}
MAX_BYTES, MAX_ROWS = 15 * 1024 * 1024, 20000
GENERIC_ERR = "Unable to read this file. Please check that it contains valid column headers and rows."


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def auto_map(headers, mode) -> dict:
    """Case/space/underscore/hyphen-tolerant matching of CSV headers to dashboard fields."""
    used, mp = set(), {}
    for field, _label, aliases in FIELDS[mode]:
        hit = next((h for a in aliases for h in headers if h not in used and norm(h) == a), None)
        if hit is None:
            hit = next((h for a in aliases if len(a) >= 6 for h in headers if h not in used and a in norm(h)), None)
        if hit is not None:
            used.add(hit)
            mp[field] = hit
    return mp


def detect_mode(headers) -> tuple[str, bool]:
    n = [norm(h) for h in headers]
    cnt = lambda ks: sum(any(k in x for k in ks) for x in n)  # noqa: E731
    e = cnt(["gpa", "cgpa", "course", "semester", "student", "exam", "assignment", "program", "branch"])
    c = cnt(["employee", "role", "salary", "tenure", "productivity", "attrition", "workload", "overtime", "jobtitle", "designation", "empid"])
    return ("edu" if e > c else "corp"), (abs(e - c) >= 2 or (min(e, c) == 0 and max(e, c) > 0))


def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Read CSV/XLSX bytes into an all-text DataFrame. Raises ValueError with a user-friendly message."""
    if len(data) == 0:
        raise ValueError("This file contains no usable records.")
    if len(data) > MAX_BYTES:
        raise ValueError("This file is too large for this demo (limit 15 MB).")
    low = name.lower()
    try:
        if low.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str, keep_default_na=False)
        elif low.endswith(".csv") or low.endswith(".txt"):
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
            try:                                          # only comma, semicolon, tab or pipe are valid delimiters
                sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
            df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, sep=sep)
        else:
            raise ValueError("Please upload a valid CSV or Excel (.xlsx) file.")
    except ValueError:
        raise
    except ImportError:
        raise ValueError("Reading Excel files needs the 'openpyxl' package. Install it with:  pip install openpyxl")
    except Exception:
        raise ValueError(GENERIC_ERR)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [not (c.startswith("Unnamed") and (df[c].astype(str).str.strip() == "").all()) for c in df.columns]]
    df = df[(df.apply(lambda r: r.astype(str).str.strip().ne("").any(), axis=1))] if len(df) else df
    if df.shape[1] < 2:
        raise ValueError(GENERIC_ERR)
    if len(df) == 0:
        raise ValueError("This file contains no usable records.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"This file has too many rows for this demo (limit {MAX_ROWS:,}).")
    return df.reset_index(drop=True)


def _text(raw, col):
    if not col or col not in raw:
        return pd.Series([""] * len(raw), index=raw.index)
    return raw[col].astype(str).str.strip()


def build_records(raw: pd.DataFrame, mp: dict, mode: str, weights=None, th=None):
    """Turn a raw uploaded table into a scored dataset. Returns (df, issues[list of (row, column, problem)])."""
    ed = mode == "edu"
    issues = []
    scale = {}
    for f in PCT_FIELDS:                                   # 0-1 columns are treated as fractions (x100)
        t = _text(raw, mp.get(f)).str.replace(r"[%,\s]", "", regex=True)
        mx = pd.to_numeric(t, errors="coerce").max()
        scale[f] = 100.0 if (pd.notna(mx) and 0 < mx <= 1) else 1.0

    def num(f):
        if not mp.get(f):
            return pd.Series(np.nan, index=raw.index)
        t = _text(raw, mp[f]).str.replace(r"[%,\s]", "", regex=True)
        v = pd.to_numeric(t.replace("", np.nan), errors="coerce") * scale.get(f, 1.0)
        bad = (t != "") & (v.isna() | ((v < 0) | (v > 100) if f in PCT_FIELDS else False))
        for i in raw.index[bad.fillna(True).to_numpy(dtype=bool)]:
            issues.append((int(i) + 2, mp[f], f"Invalid or out-of-range value: {raw.at[i, mp[f]]}"))
        return v.where(~bad.fillna(True))

    def label(f, default="Unspecified", fmt=None):
        t = _text(raw, mp.get(f))
        if fmt:
            n = pd.to_numeric(t.str.extract(r"(\d+)")[0], errors="coerce")
            return n.map(lambda x: fmt.format(int(x)) if pd.notna(x) else "N/A")
        return t.replace("", default)

    name_t, id_t = _text(raw, mp.get("name")), _text(raw, mp.get("id"))
    prefix = "S-" if ed else "E-"
    ids = id_t.where(id_t != "", pd.Series([f"{prefix}{1001 + i}" for i in range(len(raw))], index=raw.index))
    dup = ids.duplicated(keep="first")
    for i in raw.index[dup.to_numpy(dtype=bool)]:
        issues.append((int(i) + 2, mp.get("id") or "(row)", f"Duplicate ID: {ids[i]}"))
    ids = ids.where(~dup, ids + "-" + pd.Series(raw.index.astype(str), index=raw.index))
    out = pd.DataFrame({"id": ids, "name": name_t.where(name_t != "", ids)}, index=raw.index)
    out["att"], out["eng"] = num("att"), num("eng")
    perf = num("perf")
    if ed:
        gpa = num("gpa")
        gmax = gpa.max()
        k = 25 if (pd.notna(gmax) and gmax <= 4.5) else 10 if (pd.notna(gmax) and gmax <= 10) else np.nan
        perf = perf.where(perf.notna(), gpa * k)
        out["gpa"] = gpa
    out["perf"] = perf.round()
    out["grp"] = label("grp")
    if ed:
        assign = num("assign")
        back = num("back")
        out["back"] = back.where(back.notna(), (100 - assign) / 100 * 6)   # derive backlog from completion if absent
        out["assign"] = assign
        out["fail"], out["late"], out["missed"] = num("fail"), num("late"), num("missed")
        out["year"], out["sem"] = label("year", fmt="Year {}"), label("sem", fmt="Sem {}")
    else:
        ten, hrs = num("ten"), num("hrs")
        out["role"], out["loc"] = label("role"), label("loc")
        out["ten"], out["hrs"], out["prod"], out["goal"] = ten, hrs, num("prod"), num("goal")
        ot = num("ot")
        out["ot"] = ot.where(ot.notna(), (hrs - 40).clip(lower=0))
        band = num("band").round()
        out["band"] = band.where((band >= 1) & (band <= 5))
        out["crit"] = _text(raw, mp.get("role")).str.contains(r"senior|manager|lead|director|head|vp|chief", case=False, regex=True)
        out["band_t"] = np.select([ten.isna(), ten < 1, ten < 3, ten < 5], ["N/A", "Under 1 yr", "1–3 yrs", "3–5 yrs"], "5+ yrs")
        out["absd"] = ((100 - out["att"]) / 100 * 220).round()
    rk = num("risk")
    lv = _text(raw, mp.get("lvl")).str.lower().map(LEVEL_TEXT)
    out["risk_in"] = rk.where(rk.notna(), lv)
    keep = (name_t != "") | (id_t != "") | out[["att", "perf", "eng"]].notna().any(axis=1)
    out = out[keep].reset_index(drop=True)
    out["has_provided_risk"] = out["risk_in"].notna()
    return score_frame(out, mode, weights, th), issues


def apply_snapshot(cur: pd.DataFrame, prev: pd.DataFrame, use_id: bool, mode: str, weights=None, th=None):
    """Join an earlier file to the current one so the trend charts hold real before/after values."""
    th = th or DEFAULT_TH
    key = "id" if use_id else "name"
    p = prev.drop_duplicates(key).set_index(key)
    m = cur[key].map(lambda k: k in p.index)
    if not m.any():
        raise ValueError("No people could be matched between the two files. Check that IDs or names are the same.")
    out = cur.copy()
    for c in TREND_COLS:
        out[c] = np.nan
    for pre in ("perf", "att"):
        out.loc[m, f"{pre}_t1"] = cur.loc[m, key].map(p[pre]).to_numpy(float)
        out.loc[m, f"{pre}_t6"] = cur.loc[m, pre].to_numpy(float)
    out["prev_risk"] = np.where(m, cur[key].map(p["risk"]).to_numpy(float), np.nan)
    out = score_frame(out, mode, weights, th)
    alerts = out[(out["prev_risk"] < th[1]) & (out["risk"] >= th[1])]
    return out, int(m.sum()), alerts[["id", "name", "prev_risk", "risk"]]


# =====================================================================================
# 4. ANALYTICS (all numbers are computed from the active dataset)
# =====================================================================================
def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def groups(df, col):
    return [(k, df[df[col] == k]) for k in sorted(df[col].dropna().unique(), key=nat_key)]


def fmt_pct(v, nf="pct"):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v / 100:.2f}" if nf == "dec" else f"{round(v)}%"


def fmt_num(v, d=0):
    return "N/A" if v is None or pd.isna(v) else f"{v:.{d}f}"


def factor_totals(df, mode) -> pd.DataFrame:
    fs = FACTORS[mode]
    t = pd.Series({f.name: df[f"c{i}"].sum() for i, f in enumerate(fs)})
    share = (t / (t.sum() or 1) * 100).round(1)
    return share.sort_values(ascending=False).rename("share").rename_axis("factor").reset_index()


def kpi_items(df, mode, th, nf="pct"):
    hi = int((df["risk"] >= th[1]).sum())
    pc = lambda n: fmt_pct(n / len(df) * 100, "pct") if len(df) else "0%"  # noqa: E731
    dec = int(((df["perf_t1"] - df["perf_t6"]) >= 5).sum())
    if mode == "edu":
        return [("Total Students", str(len(df)), "in current selection", "#4f8cff"),
                ("Students At Risk", str(hi), f"{pc(hi)} High or Critical", "#f87171"),
                ("Average Attendance", fmt_pct(df["att"].mean(), nf), "across selection", "#22d3ee"),
                ("Avg Academic Score", fmt_pct(df["perf"].mean(), nf), "mean exam performance", "#9b6bff"),
                ("Declining Performance", str(dec), "score drop of 5+ pts", "#fbbf24"),
                ("Intervention Required", str(int((df["lv"] == 3).sum())), "Critical risk level", "#f472b6")]
    at = int(((df["risk"] >= 50) & (df["eng"] < 55)).sum())
    return [("Total Employees", str(len(df)), "in current selection", "#4f8cff"),
            ("Employees At Risk", str(hi), f"{pc(hi)} High or Critical", "#f87171"),
            ("Average Performance", fmt_pct(df["perf"].mean(), nf), "mean performance score", "#9b6bff"),
            ("Average Attendance", fmt_pct(df["att"].mean(), nf), "across selection", "#22d3ee"),
            ("Attrition Risk", pc(at), f"{at} flagged (risk 50+, low engagement)", "#fbbf24"),
            ("Productivity Index", fmt_num(df["prod"].mean()), "0-100 composite", "#34d399")]


def build_insights(df, mode, th) -> list[str]:
    out = []
    many = "students" if mode == "edu" else "employees"
    hi = df[df["risk"] >= th[1]]
    xk = "att" if mode == "edu" else "eng"
    pair = df[[xk, "perf"]].dropna()
    if len(pair) >= 3:
        r = pair[xk].corr(pair["perf"])
        if np.isfinite(r):
            strength = "strong" if abs(r) >= .7 else "moderate" if abs(r) >= .4 else "weak"
            out.append(f"{'Attendance' if mode == 'edu' else 'Engagement'} shows a {strength} "
                       f"{'positive' if r >= 0 else 'negative'} association with {'academic' if mode == 'edu' else 'job'} "
                       f"performance in the current sample (r = {r:.2f}).")
    g = sorted(((k, int((s["risk"] >= th[1]).sum()), len(s)) for k, s in groups(df, "grp")), key=lambda x: (-x[1] / x[2], -x[1]))
    if g:
        out.append(f"{g[0][0]} has the highest concentration of high-risk {many}: {g[0][1]} of {g[0][2]} ({round(g[0][1] / g[0][2] * 100)}%).")
    ft = factor_totals(df, mode)
    out.append(f"{ft.iloc[0]['factor']} is the largest contributor to risk scores, at {ft.iloc[0]['share']}% of total contribution.")
    if mode == "edu":
        s = int((hi["att"] < 75).sum())
        out.append(f"{round(s / len(hi) * 100) if len(hi) else 0}% of high-risk students ({s} of {len(hi)}) have attendance below 75%.")
    else:
        s = int((((hi["perf_t1"] - hi["perf_t6"]) >= 5) & (hi["eng"] < 60)).sum())
        out.append(f"Employees with declining performance and low engagement account for {round(s / len(hi) * 100) if len(hi) else 0}% "
                   f"of the high-risk group ({s} of {len(hi)}).")
    t = df["perf_t6"].mean() - df["perf_t1"].mean()
    if np.isfinite(t):
        out.append(f"Average performance {'rose' if t >= 0 else 'fell'} by {abs(t):.1f} points over the period shown.")
    out.append(f"{int((df['lv'] == 3).sum())} {many} are at Critical level; the average risk score is {round(df['risk'].mean())}.")
    return out


def impact_items(df, mode, th) -> list[tuple[str, str, str]]:
    hi = df[df["risk"] >= th[1]]
    if mode == "edu":
        return [("Classes missed", f"{int(df['missed'].sum()):,}", "total, estimated"),
                ("With failed subjects", str(int((df["fail"] > 0).sum())), "students"),
                ("Possible repeat risk", str(int((df["fail"] >= 2).sum())), "2+ failed subjects"),
                ("Avg assignment backlog", fmt_num(df["back"].mean(), 1), "per student"),
                ("Late arrivals", str(int(df["late"].sum())), "this term"),
                ("Low engagement", str(int((df["eng"] < 50).sum())), "under 50% engagement")]
    top = sorted(((k, s["risk"].mean()) for k, s in groups(df, "grp")), key=lambda x: -x[1])[0]
    band = hi["band"].dropna().astype(int).clip(1, 5)
    expo = sum(SALARY_COST_LAKH[b - 1] * .5 for b in band)
    return [("Est. productivity impact", f"~{df['risk'].mean() * .22:.1f}%", "of capacity, illustrative"),
            ("Turnover exposure", f"₹{round(expo)} L", "replacement cost, illustrative"),
            ("High-risk employees", str(len(hi)), f"{round(len(hi) / max(len(df), 1) * 100)}% of selection"),
            ("Highest-risk department", str(top[0]), f"avg score {round(top[1])}"),
            ("Absence impact", f"{int(df['absd'].sum()):,}", "person-days / year, est."),
            ("Critical roles at risk", str(int((df["crit"].astype(bool) & (df["risk"] >= th[1])).sum())), "Senior / Manager level")]


def trend_means(df, prefix, n=6):
    cols = [f"{prefix}_t{i}" for i in range(1, 7)][-n:]
    return [None if df[c].dropna().empty else float(df[c].mean()) for c in cols]


def apply_range(df, col, choice):
    ranges = {"Below 60": (0, 60), "60–75": (60, 75), "75–90": (75, 90), "90 and above": (90, 101)}
    if choice in ranges:
        a, b = ranges[choice]
        return df[(df[col] >= a) & (df[col] < b)]
    return df


def summary_table(df, dim_col):
    rows = []
    for k, s in groups(df, dim_col):
        rows.append({"Group": k, "Count": len(s), "Avg risk": round(s["risk"].mean()), "High / Critical": int((s["lv"] >= 2).sum()),
                     "Avg performance": s["perf"].mean(), "Avg attendance": s["att"].mean()})
    return pd.DataFrame(rows)


def correlation_matrix(df, mode):
    cols = ["att", "perf", "eng", "gpa", "back", "risk"] if mode == "edu" else ["att", "perf", "eng", "prod", "ot", "risk"]
    names = {"att": "Attendance", "perf": "Performance", "eng": "Engagement", "gpa": "GPA", "back": "Backlog", "prod": "Productivity", "ot": "Overtime", "risk": "Risk"}
    m = pd.DataFrame(np.nan, index=[names[c] for c in cols], columns=[names[c] for c in cols])
    for a in cols:
        for b in cols:
            p = df[[a, b]].dropna()
            if len(p) >= 3:
                m.loc[names[a], names[b]] = 1.0 if a == b else p[a].corr(p[b])
    return m


def snapshot_html(df, mode, th, source, nf="pct") -> str:
    dim = DIMS[mode][0]
    by = summary_table(df, dim[0])
    top = df.sort_values("risk", ascending=False).head(10)
    t = lambda frame: frame.to_html(index=False, border=1, escape=True)  # noqa: E731
    by = by.assign(**{"Avg performance": by["Avg performance"].map(lambda v: fmt_pct(v, nf)), "Avg attendance": by["Avg attendance"].map(lambda v: fmt_pct(v, nf))}) if len(by) else by
    topv = top[["name", "risk", "main", "action"]].assign(level=[LEVELS[i][0] for i in top["lv"]])
    return (f"<!DOCTYPE html><meta charset='utf-8'><title>At-Risk snapshot</title><body style='font:14px system-ui;max-width:900px;margin:24px auto'>"
            f"<h1>At-Risk Intelligence snapshot</h1><p>{dt.datetime.now():%Y-%m-%d %H:%M} • {len(df)} records • {html.escape(source)}</p>"
            f"<p><i>Demonstration heuristic, not a validated prediction. Use to support, not replace, human judgement.</i></p>"
            f"<h2>By {dim[1].lower()}</h2>{t(by)}<h2>Top priorities</h2>{t(topv)}</body>")


# =====================================================================================
# 5. PERSISTENCE (settings, statuses, saved views and the uploaded dataset stay on this machine)
# =====================================================================================
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    keep = {k: state.get(k) for k in ("weights", "th", "nf", "status", "views", "maps")}
    try:
        STATE_FILE.write_text(json.dumps(keep), encoding="utf-8")
    except Exception:
        pass


# =====================================================================================
# 6. STREAMLIT UI
# =====================================================================================
NAV = ["Overview", "Students", "Employees", "Risk Analysis", "Compare", "Performance", "Attendance", "Insights", "Reports", "Settings"]
RANGE_OPTS = ["All", "Below 60", "60–75", "75–90", "90 and above"]
FILTERS = {
    "edu": [("Course", "grp", "cat"), ("Year", "year", "cat"), ("Semester", "sem", "cat"), ("Risk level", "lv", "lv"), ("Attendance", "att", "range"), ("Performance", "perf", "range")],
    "corp": [("Department", "grp", "cat"), ("Role", "role", "cat"), ("Tenure", "band_t", "cat"), ("Risk level", "lv", "lv"), ("Performance", "perf", "range"), ("Engagement", "eng", "range")],
}


def theme_css(theme: str) -> str:
    dark = theme == "dark"
    v = dict(bg="#070b16", sd="#0a1020", cd="#0e1527", cd2="#111a31", bd="#1b2542", tx="#e7ebf8", mu="#8a94b4", hv="#ffffff0a", sh="#0006", glow="#1a2352") if dark else \
        dict(bg="#f3f5fb", sd="#ffffff", cd="#ffffff", cd2="#fbfcff", bd="#e2e7f2", tx="#141b34", mu="#5b6788", hv="#0f172a0a", sh="#1e2a5018", glow="#dde6ff")
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
:root{{--bg:{v['bg']};--cd:{v['cd']};--cd2:{v['cd2']};--bd:{v['bd']};--tx:{v['tx']};--mu:{v['mu']};--hv:{v['hv']}}}
html,body,[class*="css"],.stApp{{font-family:Manrope,system-ui,sans-serif!important}}
.stApp{{background:radial-gradient(900px 500px at 85% -10%,{v['glow']} 0,transparent 60%),{v['bg']};color:{v['tx']}}}
[data-testid="stSidebar"]{{background:{v['sd']};border-right:1px solid {v['bd']}}}
[data-testid="stHeader"]{{background:transparent}}
.stApp,.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricValue"],.stApp [data-testid="stMetricLabel"],.stApp summary{{color:{v['tx']}}}
.stApp small,.stApp [data-testid="stCaptionContainer"],.stApp .sub{{color:{v['mu']}!important}}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"],textarea,input{{background:{v['cd']}!important;color:{v['tx']}!important;border-color:{v['bd']}!important}}
div[data-baseweb="popover"] *{{background:{v['cd']};color:{v['tx']}}}
.stButton>button,.stDownloadButton>button{{background:{v['cd']};color:{v['tx']};border:1px solid {v['bd']};border-radius:10px;transition:.15s}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:#4f8cff;box-shadow:0 0 12px #4f8cff44;color:{v['tx']}}}
.stButton>button[kind="primary"]{{background:linear-gradient(135deg,#4f8cff,#9b6bff);border:0;color:#fff}}
.ari-brand{{display:flex;gap:10px;align-items:center;margin-bottom:10px}}.ari-brand b{{font-size:13px;letter-spacing:.06em;font-weight:800}}
.ari-logo{{width:34px;height:34px;border-radius:10px;background:conic-gradient(from 200deg,#4f8cff,#9b6bff,#f472b6,#22d3ee,#4f8cff);box-shadow:0 0 18px #4f8cff66}}
.ari-card{{background:linear-gradient(180deg,{v['cd2']},{v['cd']});border:1px solid {v['bd']};border-radius:16px;padding:14px 16px;box-shadow:0 8px 24px {v['sh']};margin-bottom:10px;transition:border-color .2s}}
.ari-card:hover{{border-color:#4f8cff66}}
.kpi{{position:relative;overflow:hidden;transition:transform .2s}}.kpi:hover{{transform:translateY(-3px)}}
.kpi:before{{content:"";position:absolute;inset:0 0 auto;height:3px;background:var(--k)}}
.kpi small{{font-size:12px}}.kpi b{{display:block;font-size:26px;font-weight:800;letter-spacing:-.02em}}.kpi span{{font-size:11px;color:{v['mu']}}}
.pill{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700;border:1px solid}}
.note{{border:1px dashed #4f8cff66;background:#4f8cff10;border-radius:12px;padding:8px 14px;font-size:12px;margin-bottom:12px}}
.row{{display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid {v['bd']}}}
.dot{{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:12px;color:#08101f;flex:none}}
.meter{{position:relative;height:10px;border-radius:9px;margin:12px 0}}.meter u{{position:absolute;top:-4px;width:4px;height:18px;border-radius:3px;background:{v['tx']}}}
.tier{{border-radius:12px;padding:12px;border:1px solid;font-size:13px}}
</style>"""


GLOW_JS = """<script>try{const d=window.parent.document;if(!d.getElementById('ari-glow')&&!matchMedia('(hover:none)').matches){
const g=d.createElement('div');g.id='ari-glow';g.style.cssText='position:fixed;left:-270px;top:-270px;width:540px;height:540px;border-radius:50%;pointer-events:none;z-index:1;opacity:0;transition:opacity .5s;filter:blur(24px);background:radial-gradient(circle,rgba(79,140,255,.16) 0,rgba(155,107,255,.07) 40%,transparent 70%)';d.body.appendChild(g);
let tx=0,ty=0,x=0,y=0,raf=0;const step=()=>{x+=(tx-x)*.12;y+=(ty-y)*.12;g.style.transform='translate3d('+x+'px,'+y+'px,0)';raf=(Math.abs(tx-x)>.5||Math.abs(ty-y)>.5)?requestAnimationFrame(step):0};
d.addEventListener('pointermove',e=>{tx=e.clientX;ty=e.clientY;if(g.style.opacity!=='1'){x=tx;y=ty;g.style.opacity=1}if(!raf)raf=requestAnimationFrame(step)},{passive:true});}}catch(e){}</script>"""


def pill(lv: int) -> str:
    c = LEVELS[int(lv)][1]
    return f"<span class='pill' style='color:{c};border-color:{c}66;background:{c}18'>{LEVELS[int(lv)][0]}</span>"


def esc(x) -> str:
    return html.escape(str(x))


def style_fig(fig, h=300, legend=True):
    dark = st.session_state.theme == "dark"
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=30, b=10), height=h, showlegend=legend, legend=dict(orientation="h", y=-0.2),
                      font=dict(family="Manrope, sans-serif", color="#e7ebf8" if dark else "#141b34"))
    fig.update_xaxes(gridcolor="#1a2340" if dark else "#e6eaf4")
    fig.update_yaxes(gridcolor="#1a2340" if dark else "#e6eaf4")
    return fig


def show(fig, key, h=300, legend=True):
    st.plotly_chart(style_fig(fig, h, legend), use_container_width=True, key=key)


def card(title, sub="", body=""):
    st.markdown(f"<div class='ari-card'><b>{esc(title)}</b><div class='sub' style='font-size:12px;margin-bottom:6px'>{esc(sub)}</div>{body}</div>", unsafe_allow_html=True)


def note():
    s = st.session_state
    txt = (f"{s.up['rs']}. This is a demonstration heuristic, not a scientifically validated prediction. Uploaded data is processed locally."
           if s.up else "Demonstration analytics model using synthetic data. Risk scores are a weighted heuristic for decision support, not a scientifically validated prediction.")
    st.markdown(f"<div class='note'>{esc(txt)}</div>", unsafe_allow_html=True)


# ---------- state ----------
def init_state():
    s = st.session_state
    if "init" in s:
        return
    saved = load_state()
    s.weights = saved.get("weights") or {m: list(w) for m, w in DEFAULT_WEIGHTS.items()}
    s.th = saved.get("th") or list(DEFAULT_TH)
    s.nf = saved.get("nf") or "pct"
    s.status = saved.get("status") or {}
    s.views = saved.get("views") or {}
    s.maps = saved.get("maps") or {}
    s.seed = 42
    s.demo = generate_demo(s.seed, s.weights, s.th)
    s.data, s.up, s.pending, s.up_counter, s.opened = dict(s.demo), None, None, 0, None
    s.theme = st.query_params.get("theme", "dark") if hasattr(st, "query_params") else "dark"
    try:                                     # restore a previously uploaded dataset
        blob = pd.read_pickle(DATA_FILE)
        s.up, s.data = blob["up"], {"edu": blob["edu"], "corp": blob["corp"]}
        s.mode_pref = blob["up"]["mode"]
    except Exception:
        pass
    s.mode = "Corporate / HR" if s.get("mode_pref") == "corp" else "Education"
    s.nav = "Overview"
    s.init = True


def persist():
    s = st.session_state
    save_state(dict(s))


def rescore_all():
    s = st.session_state
    for store in (s.data, s.demo):
        for m in ("edu", "corp"):
            if len(store[m]):
                store[m] = score_frame(store[m], m, s.weights[m], s.th)
    persist()


def status_of(pid):
    return st.session_state.status.get(f"{'u' if st.session_state.up else 'd'}:{pid}", {"s": "New", "n": ""})


def set_status(pid, **kw):
    s = st.session_state
    k = f"{'u' if s.up else 'd'}:{pid}"
    cur = dict(status_of(pid))
    cur.update(kw)
    s.status[k] = cur
    persist()


# ---------- sidebar ----------
def on_nav():
    s = st.session_state
    if s.nav == "Students":
        s.mode = "Education"
    elif s.nav == "Employees":
        s.mode = "Corporate / HR"


def on_theme():
    st.session_state.theme = "light" if st.session_state.theme_tg else "dark"
    try:
        st.query_params["theme"] = st.session_state.theme
    except Exception:
        pass


def sidebar():
    s = st.session_state
    with st.sidebar:
        st.markdown("<div class='ari-brand'><div class='ari-logo'></div><div><b>AT-RISK INTELLIGENCE</b><br><small>Predict • Detect • Act</small></div></div>", unsafe_allow_html=True)
        st.radio("Navigation", NAV, key="nav", on_change=on_nav, label_visibility="collapsed")
        st.radio("Mode", ["Education", "Corporate / HR"], key="mode", horizontal=True)
        st.toggle("Light theme", value=s.theme == "light", key="theme_tg", on_change=on_theme)
        st.divider()
        st.markdown("<small>DATA SOURCE</small>", unsafe_allow_html=True)
        if s.up:
            st.markdown(f"**● Uploaded Dataset**  \n{esc(s.up['fn'])}  \n{s.up['n']} records • {s.up['cols']} columns" + (f"  \nSnapshot: {esc(s.up['snap'])}" if s.up.get("snap") else ""))
            if s.up["issues"]:
                st.warning(f"{len(s.up['issues'])} values flagged")
                st.download_button("Error report (CSV)", pd.DataFrame(s.up["issues"], columns=["Row", "Column", "Problem"]).to_csv(index=False), "flagged-values.csv", use_container_width=True)
        else:
            st.markdown("**● Synthetic Demo Data**")
        f = st.file_uploader("Replace CSV / Excel" if s.up else "Upload CSV / Excel", type=["csv", "xlsx", "xls"], key=f"up{s.up_counter}")
        if f is not None and (s.pending is None or s.pending["fn"] != f.name):
            start_upload(f)
        if s.up:
            snap = st.file_uploader("Add earlier snapshot", type=["csv", "xlsx", "xls"], key=f"snap{s.up_counter}")
            if snap is not None:
                add_snapshot(snap)
            if st.button("Reset to Demo Data", use_container_width=True):
                reset_demo()
        st.caption("CSV data is processed locally on the machine running this app.")
    return s.mode


def start_upload(f):
    s = st.session_state
    try:
        raw = read_table(f.name, f.getvalue())
    except ValueError as e:
        st.sidebar.error(str(e))
        return
    except Exception:
        st.sidebar.error(GENERIC_ERR)
        return
    headers = list(raw.columns)
    mode, sure = detect_mode(headers)
    sig = "|".join(sorted(norm(h) for h in headers))
    saved = s.maps.get(sig)
    if saved and all(v in headers for v in saved["mp"].values()):
        mode, sure = saved["mode"], True
        mp, remembered = saved["mp"], True
    else:
        mp, remembered = auto_map(headers, mode), False
    s.pending = dict(fn=f.name, raw=raw, headers=headers, mode=mode if sure else None, mp=mp, sig=sig, remembered=remembered, sure=sure)
    st.rerun()


def reset_demo():
    s = st.session_state
    s.data, s.up, s.pending = dict(s.demo), None, None
    s.up_counter += 1
    try:
        DATA_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    reset_filters()
    st.rerun()


def add_snapshot(f):
    s = st.session_state
    m = s.up["mode"]
    key = f"snap_done_{f.name}_{f.size}"
    if s.get(key):
        return
    try:
        prev_raw = read_table(f.name, f.getvalue())
        prev, _ = build_records(prev_raw, auto_map(list(prev_raw.columns), m), m, s.weights[m], s.th)
        cur, hit, alerts = apply_snapshot(s.data[m], prev, bool(s.up.get("has_id")) and "id" in auto_map(list(prev_raw.columns), m), m, s.weights[m], s.th)
    except ValueError as e:
        st.sidebar.error(str(e))
        return
    s.data[m] = cur
    s.up.update(snap=f.name, alerts=alerts.to_dict("records"))
    s[key] = True
    save_dataset()
    st.sidebar.success(f"Matched {hit} of {len(cur)} people; real trends now shown.")


def save_dataset():
    s = st.session_state
    try:
        pd.to_pickle({"up": s.up, "edu": s.data["edu"], "corp": s.data["corp"]}, DATA_FILE)
    except Exception:
        pass


def review_panel():
    """Mapping screen: choose dataset type, map columns, preview, continue."""
    s = st.session_state
    p = s.pending
    st.subheader("Review your data")
    if p.get("error"):
        st.error(p["error"])
    st.caption(f"{p['fn']} • {len(p['raw'])} records • {len(p['headers'])} columns")
    if p["remembered"]:
        st.success("Using your saved column mapping for this file layout.")
    elif p["sure"]:
        st.success("Dataset type detected automatically. Check the mapping below.")
    else:
        st.error("We couldn't identify this dataset. Please choose Students or Employees manually.")
    labels = {"edu": "Students", "corp": "Employees"}
    idx = None if p["mode"] is None else ["edu", "corp"].index(p["mode"])
    choice = st.radio("What type of dataset is this?", ["edu", "corp"], index=idx, format_func=labels.get, horizontal=True, key=f"type_{p['fn']}")
    if choice is None:
        return st.button("Cancel", on_click=lambda: _cancel())
    if choice != p["mode"]:
        p["mode"], p["mp"] = choice, auto_map(p["headers"], choice)
    mp = {}
    st.markdown("**Map your CSV**")
    cols = st.columns(2)
    for i, (field, label, _a) in enumerate(FIELDS[choice]):
        opts = ["— none —"] + p["headers"]
        default = opts.index(p["mp"][field]) if field in p["mp"] else 0
        pick = cols[i % 2].selectbox(label, opts, index=default, key=f"map_{p['fn']}_{choice}_{field}")
        if pick != "— none —":
            mp[field] = pick
    has = any(f in mp for f in METRICS)
    try:
        prev, _iss = build_records(p["raw"].head(8), mp, choice, s.weights[choice], s.th)
    except Exception:
        prev = pd.DataFrame()
        has = False
    if not has:
        st.error("Some required fields could not be detected. Please map at least one of attendance, performance, engagement or GPA manually.")
    elif len(prev):
        st.markdown("**Data preview**")
        st.dataframe(pd.DataFrame({"Name": prev["name"], "Attendance": prev["att"].map(fmt_pct), "Performance": prev["perf"].map(fmt_pct),
                                   "Engagement": prev["eng"].map(fmt_pct), "Risk": prev["risk"].astype(int)}), hide_index=True, use_container_width=True)
        st.caption(("Risk score taken from the uploaded column where provided; otherwise calculated" if ("risk" in mp or "lvl" in mp) else "Risk score calculated")
                   + " from available uploaded fields. Missing values show as N/A and are excluded from averages.")
    c1, c2 = st.columns(2)
    c1.button("Continue", type="primary", disabled=not has, use_container_width=True, on_click=finish_upload, args=(choice, mp))
    c2.button("Cancel", use_container_width=True, on_click=_cancel)


def _cancel():
    s = st.session_state
    s.pending = None
    s.up_counter += 1


def finish_upload(mode, mp):
    """Runs as a button callback (before the page reruns), so it may safely change widget state."""
    s = st.session_state
    p = s.pending
    try:
        prev = s.data[mode] if s.up and s.up["mode"] == mode else None
        df, issues = build_records(p["raw"], mp, mode, s.weights[mode], s.th)
        if df.empty:
            p["error"] = "This file contains no usable records."
            return
    except Exception:
        p["error"] = "Unable to process this file. Please check the column mapping."
        return
    alerts = None
    if prev is not None and "id" in prev:
        old = prev.set_index("id")["risk"]
        mm = df[df["id"].isin(old.index)]
        alerts = [{"id": r.id, "name": r.name, "prev_risk": old[r.id], "risk": r.risk} for r in mm.itertuples() if old[r.id] < s.th[1] <= r.risk]
    empty = pd.DataFrame()
    s.data = {"edu": df, "corp": empty} if mode == "edu" else {"edu": empty, "corp": df}
    s.up = dict(fn=p["fn"], n=len(df), cols=len(p["headers"]), mode=mode, issues=issues, alerts=alerts, has_id="id" in mp,
                rs=("Risk score taken from the uploaded column where provided, otherwise calculated from available fields" if df["has_provided_risk"].any() else "Risk score calculated from available uploaded fields"))
    s.maps[p["sig"]] = {"mode": mode, "mp": mp}
    s.mode = "Education" if mode == "edu" else "Corporate / HR"
    s.pending = None
    s.up_counter += 1
    s.nav = "Overview"
    reset_filters()
    save_dataset()
    persist()


# ---------- filters ----------
def reset_filters():
    s = st.session_state
    for m in FILTERS:
        for i in range(6):
            s[f"flt_{m}_{i}"] = "All"
    s["search"] = ""


def load_view():
    s = st.session_state
    v = s.views.get(s.get("view_pick"))
    if not v:
        return
    for i, val in enumerate(v["f"]):
        s[f"flt_{v['mode']}_{i}"] = val
    s["search"] = v["q"]
    s.mode = "Education" if v["mode"] == "edu" else "Corporate / HR"


def delete_view():
    s = st.session_state
    name = s.get("view_pick")
    if name in s.views:
        del s.views[name]
        s["view_pick"] = "—"
        persist()


def filter_bar(m, df):
    s = st.session_state
    with st.expander("Filters", expanded=True):
        top = st.columns([2, 1, 1, 1])
        top[0].text_input("Search", key="search", placeholder="Name or ID")
        top[1].radio("Time range", ["Last 3 months", "Last 6 months"], index=1, key="rng", horizontal=True)
        top[2].button("Reset filters", on_click=reset_filters, use_container_width=True)
        saved = list(s.views)
        pick = top[3].selectbox("Saved views", ["—"] + saved, key="view_pick")
        cols = st.columns(3)
        for i, (label, col, kind) in enumerate(FILTERS[m]):
            if kind == "cat":
                opts = ["All"] + sorted(df[col].dropna().astype(str).unique(), key=nat_key)
            elif kind == "lv":
                opts = ["All"] + [l[0] for l in LEVELS]
            else:
                opts = RANGE_OPTS
            key = f"flt_{m}_{i}"
            if s.get(key) not in opts:
                s[key] = "All"
            cols[i % 3].selectbox(label, opts, key=key)
        b = st.columns([2, 1, 1, 1])
        name = b[0].text_input("View name", key="view_name", label_visibility="collapsed", placeholder="Name this view")
        if b[1].button("Save view") and name.strip():
            s.views[name.strip()] = {"mode": m, "f": [s[f"flt_{m}_{i}"] for i in range(6)], "q": s.get("search", "")}
            persist()
            st.toast(f"Saved view: {name.strip()}")
        b[2].button("Load view", on_click=load_view)
        b[3].button("Delete view", on_click=delete_view)
    out = df
    q = (s.get("search") or "").strip().lower()
    if q:
        out = out[(out["name"].astype(str) + " " + out["id"].astype(str)).str.lower().str.contains(q, regex=False)]
    for i, (label, col, kind) in enumerate(FILTERS[m]):
        v = s.get(f"flt_{m}_{i}", "All")
        if v == "All":
            continue
        if kind == "cat":
            out = out[out[col].astype(str) == v]
        elif kind == "lv":
            out = out[out["lv"] == [l[0] for l in LEVELS].index(v)]
        else:
            out = apply_range(out, col, v)
    return out


# ---------- shared blocks ----------
def kpi_row(df, m):
    items = kpi_items(df, m, st.session_state.th, st.session_state.nf)
    for col, (label, value, sub, color) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='ari-card kpi' style='--k:{color}'><small>{esc(label)}</small><b>{esc(value)}</b><span>{esc(sub)}</span></div>", unsafe_allow_html=True)


def donut(df, key):
    fig = go.Figure(go.Pie(labels=[l[0] for l in LEVELS], values=[int((df["lv"] == i).sum()) for i in range(4)], hole=.68, marker=dict(colors=[l[1] for l in LEVELS])))
    show(fig, key, 260)


def factor_chart(df, m, key):
    ft = factor_totals(df, m).iloc[::-1]
    fig = go.Figure(go.Bar(x=ft["share"], y=ft["factor"], orientation="h", marker=dict(color=ft["share"], colorscale=[[0, "#4f8cff"], [1, "#f472b6"]]),
                           hovertemplate="%{y}: %{x}% of total risk contribution<extra></extra>"))
    fig.update_xaxes(ticksuffix="%")
    show(fig, key, 260, False)


def dim_bar(df, col, field, title, key, h=260):
    g = groups(df, col)
    vals = [s[field].mean() for _, s in g]
    colors = [LEVELS[int(level_of(v, st.session_state.th))][1] if field == "risk" and pd.notna(v) else "#22d3ee" for v in vals]
    fig = go.Figure(go.Bar(x=[k for k, _ in g], y=[None if pd.isna(v) else round(v, 1) for v in vals], marker=dict(color=colors), name=title))
    show(fig, key, h, False)


def scatter(df, m, key):
    xk, xl = ("att", "Attendance %") if m == "edu" else ("eng", "Engagement %")
    fig = go.Figure()
    for i, (name, color) in enumerate(LEVELS):
        d = df[df["lv"] == i]
        fig.add_trace(go.Scatter(x=d[xk], y=d["perf"], mode="markers", name=name, marker=dict(color=color, size=9 if i > 1 else 6, opacity=.9 if i > 1 else .6),
                                 text=d["name"], hovertemplate="%{text}: %{x}%, %{y}%<extra></extra>"))
    fig.update_xaxes(title=xl)
    fig.update_yaxes(title="Academic performance %" if m == "edu" else "Performance %")
    show(fig, key, 300)


def trend_lines(df, prefix, key, series, title_y=""):
    s = st.session_state
    snap = bool(s.up and s.up.get("snap"))
    n = 6 if snap else (3 if s.get("rng") == "Last 3 months" else 6)
    labels = ["Earlier", "", "", "", "", "Current"] if snap else MONTHS[-n:]
    x = list(range(n))
    fig = go.Figure()
    any_data = False
    for name, sub, color, fill in series:
        y = trend_means(sub, prefix, n)
        any_data |= any(v is not None for v in y)
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=color, width=3, shape="spline"), connectgaps=True,
                                 fill="tozeroy" if fill else None, fillcolor=color + "22"))
    fig.update_xaxes(tickvals=x, ticktext=labels)
    if not any_data:
        st.info("This dataset has no monthly history. Add an earlier snapshot in the sidebar to see real trends.")
        return
    show(fig, key, 280)


def person_rows(df, n, key_prefix):
    for r in df.sort_values("risk", ascending=False).head(n).itertuples():
        st.markdown(f"<div class='row'><span class='dot' style='background:{LEVELS[int(r.lv)][1]}'>{int(r.risk)}</span><div style='flex:1'><b>{esc(r.name)}</b>"
                    f"<br><small>{esc(r.main)} → {esc(r.action)}</small></div>{pill(r.lv)}<small style='width:70px;text-align:right'>{esc(r.priority)}</small></div>", unsafe_allow_html=True)


# ---------- pages ----------
def page_overview(df, m):
    note()
    kpi_row(df, m)
    a, b = st.columns([1, 2])
    with a:
        card("Risk distribution", "Share by risk level")
        donut(df, "ov_donut")
    with b:
        card("Average performance trend", "All vs. high-risk")
        hi = df[df["risk"] >= st.session_state.th[1]]
        trend_lines(df, "perf", "ov_trend", [("All", df, "#4f8cff", True), (f"High risk ({st.session_state.th[1]}+)", hi, "#f87171", False)])
    a, b = st.columns(2)
    with a:
        card(f"Average risk by {DIMS[m][0][1].lower()}", "Bar colour shows the group's risk level")
        dim_bar(df, "grp", "risk", "Avg risk", "ov_dim")
    with b:
        card("Risk factors", "Share of total risk contribution")
        factor_chart(df, m, "ov_fac")
    a, b = st.columns([3, 2])
    with a:
        card("Academic performance vs. attendance" if m == "edu" else "Performance vs. engagement", "Each dot is one person; colour = risk level")
        scatter(df, m, "ov_sc")
    with b:
        card("Intervention priorities", "Highest scores first")
        person_rows(df, 6, "ov")


def page_people(df, m):
    s = st.session_state
    note()
    one = "Student" if m == "edu" else "Employee"
    if m == "edu":
        view = pd.DataFrame({one: df["name"], "ID": df["id"], "Course": df["grp"], "Year": df["year"], "GPA": df["gpa"], "Attendance": df["att"], "Performance": df["perf"]})
    else:
        view = pd.DataFrame({one: df["name"], "ID": df["id"], "Department": df["grp"], "Role": df["role"], "Performance": df["perf"], "Attendance": df["att"], "Engagement": df["eng"]})
    view["Risk Score"] = df["risk"].astype(int)
    view["Risk Level"] = [LEVELS[i][0] for i in df["lv"]]
    view["Action"] = df["action"]
    view = view.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    st.caption(f"{len(view)} {('students' if m == 'edu' else 'employees')}. Click a column header to sort; select a row to open the profile.")
    cfg = {c: st.column_config.NumberColumn(format="%.0f%%") for c in ("Attendance", "Performance", "Engagement") if c in view}
    cfg["Risk Score"] = st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d")
    ev = st.dataframe(view.drop(columns=[]), hide_index=True, use_container_width=True, column_config=cfg, on_select="rerun", selection_mode="single-row", key=f"tbl_{m}")
    rows = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    if rows:
        pid = view.loc[rows[0], "ID"]
        if s.opened != (m, pid):
            s.opened = (m, pid)
            open_profile(m, pid)
    else:
        s.opened = None


def profile_body(m, pid):
    s = st.session_state
    df = s.data[m]
    p = df[df["id"] == pid]
    if p.empty:
        st.warning("This record is no longer in the active dataset.")
        return
    r = p.iloc[0]
    ed = m == "edu"
    th = s.th
    sub = f"{r['grp']} • {r['year']} • {r['sem']}" if ed else f"{r['grp']} • {r['role']} • {r['loc']} • {fmt_num(r['ten'], 1)} yrs"
    st.markdown(f"### {esc(r['name'])}  \n<small>{esc(r['id'])} • {esc(sub)}</small>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;align-items:baseline;gap:12px'><b style='font-size:40px'>{int(r['risk'])}</b>{pill(r['lv'])}<span class='sub'>Risk score (0-100)</span></div>"
                f"<div class='meter' style='background:linear-gradient(90deg,#34d399 0 {th[0]}%,#fbbf24 {th[0]}% {th[1]}%,#fb923c {th[1]}% {th[2]}%,#f87171 {th[2]}%)'><u style='left:calc({r['risk']}% - 2px)'></u></div>", unsafe_allow_html=True)
    stats = ([("GPA", fmt_num(r["gpa"], 2)), ("Attendance", fmt_pct(r["att"], s.nf)), ("Performance", fmt_pct(r["perf"], s.nf)), ("Engagement", fmt_pct(r["eng"], s.nf)),
              ("Assignments done", fmt_pct(r["assign"], s.nf)), ("Late arrivals", fmt_num(r["late"])), ("Classes missed", fmt_num(r["missed"])), ("Failed subjects", fmt_num(r["fail"]))] if ed else
             [("Performance", fmt_pct(r["perf"], s.nf)), ("Attendance", fmt_pct(r["att"], s.nf)), ("Engagement", fmt_pct(r["eng"], s.nf)), ("Productivity", fmt_num(r["prod"])),
              ("Goal completion", fmt_pct(r["goal"], s.nf)), ("Quality score", fmt_pct(r["qual"], s.nf)), ("Hours / week", fmt_num(r["hrs"], 1)), ("Salary band", "N/A" if pd.isna(r["band"]) else f"Band {int(r['band'])}")])
    for chunk in (stats[:4], stats[4:]):
        for col, (k, v) in zip(st.columns(4), chunk):
            col.metric(k, v)
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Performance and attendance trend**")
        snap = bool(s.up and s.up.get("snap"))
        x = list(range(6))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[r[f"perf_t{i}"] for i in range(1, 7)], name="Performance", mode="lines+markers", connectgaps=True, line=dict(color="#4f8cff", width=3)))
        fig.add_trace(go.Scatter(x=x, y=[r[f"att_t{i}"] for i in range(1, 7)], name="Attendance", mode="lines+markers", connectgaps=True, line=dict(color="#22d3ee", width=3)))
        fig.update_xaxes(tickvals=x, ticktext=["Earlier", "", "", "", "", "Current"] if snap else MONTHS)
        show(fig, f"prof_tr_{pid}", 220)
    with b:
        st.markdown("**Why this score**")
        for i, f in enumerate(FACTORS[m]):
            v = float(r[f"c{i}"])
            st.progress(min(1.0, v / 25), text=f"{f.name}: {v:.1f}")
    st.markdown("**Recent activity**")
    act = ([f"{fmt_num(r['missed'])} classes missed this term", f"{fmt_num(r['back'], 1)} assignments pending", f"LMS activity at {fmt_pct(r['lms'])}", f"Study time about {fmt_num(r['study'], 1)} h/day, library visits {fmt_num(r['lib'])}/month"] if ed else
           [f"Averaging {fmt_num(r['hrs'], 1)} hours/week ({fmt_num(r['ot'], 1)} overtime)", f"{fmt_num(r['absd'])} absence days per year (estimated)", f"Training participation {fmt_pct(r['train'])}", f"Tenure {fmt_num(r['ten'], 1)} yrs, {fmt_num(r['promos'])} promotion(s), last {fmt_num(r['since'], 1)} yrs ago"])
    for line in act:
        st.markdown(f"- {line}")
    st.markdown(f"**Recommended {'intervention' if ed else 'HR action'}:** {r['action']} — priority: {r['priority']}.  \n<small>Main issue: {esc(r['main'])}. Suggestions support, and do not replace, human judgement. Demo heuristic model.</small>", unsafe_allow_html=True)
    st.markdown("**What-if** (estimate from the demo model, not a forecast)")
    base = score_frame(p.assign(risk_in=np.nan), m, s.weights[m], th).iloc[0]
    cols = st.columns(3)
    new = {}
    for col, (k, lab) in zip(cols, [("att", "Attendance"), ("perf", "Performance"), ("eng", "Engagement")]):
        cur = r[k]
        new[k] = col.slider(lab, 0, 100, int(cur) if pd.notna(cur) else 70, key=f"wi_{pid}_{k}")
    est = score_frame(p.assign(risk_in=np.nan, **new), m, s.weights[m], th).iloc[0]
    st.info(f"Model estimate: **{int(est['risk'])}** ({LEVELS[int(est['lv'])][0]}) vs {int(base['risk'])} at current values ({int(est['risk'] - base['risk']):+d})")
    st.markdown("**Intervention status**")
    cur = status_of(pid)
    c1, c2 = st.columns([1, 2])
    c1.selectbox("Status", STATUSES, index=STATUSES.index(cur["s"]) if cur["s"] in STATUSES else 0, key=f"st_{pid}", on_change=lambda: set_status(pid, s=st.session_state[f"st_{pid}"]))
    c2.text_area("Notes (stored only on this machine)", value=cur["n"], key=f"nt_{pid}", on_change=lambda: set_status(pid, n=st.session_state[f"nt_{pid}"]))


def open_profile(m, pid):
    if hasattr(st, "dialog"):
        st.dialog("Profile", width="large")(profile_body)(m, pid)
    else:
        with st.expander("Profile", expanded=True):
            profile_body(m, pid)


def page_risk(df, m):
    note()
    a, b = st.columns([1, 2])
    with a:
        card("Risk distribution", "All levels")
        donut(df, "rk_donut")
    with b:
        card("Risk factor analysis", "What drives elevated scores")
        factor_chart(df, m, "rk_fac")
    dims = DIMS[m]
    dim = st.selectbox("Group by", [d[1] for d in dims], key=f"rk_dim_{m}")
    col = dict((d[1], d[0]) for d in dims)[dim]
    g = groups(df, col)
    a, b = st.columns([3, 2])
    with a:
        card(f"Risk levels by {dim.lower()}", "Stacked counts")
        fig = go.Figure([go.Bar(name=n, x=[k for k, _ in g], y=[int((s["lv"] == i).sum()) for _, s in g], marker=dict(color=c)) for i, (n, c) in enumerate(LEVELS)])
        fig.update_layout(barmode="stack")
        show(fig, "rk_stack", 300)
    with b:
        card(f"Heatmap: {dim.lower()} × risk level", "Number of people; darker means more")
        z = [[int((s["lv"] == i).sum()) for i in range(4)] for _, s in g]
        fig = go.Figure(go.Heatmap(z=z, x=[l[0] for l in LEVELS], y=[k for k, _ in g], text=z, texttemplate="%{text}", colorscale=[[0, "#16213f"], [1, "#4f8cff"]], showscale=False))
        show(fig, "rk_heat", 300, False)
    card("Risk score distribution", "People per 10-point band")
    bins = [int((np.minimum(9, df["risk"] // 10) == i).sum()) for i in range(10)]
    fig = go.Figure(go.Bar(x=[f"{i * 10}–{i * 10 + 9}" for i in range(10)], y=bins, marker=dict(color=[LEVELS[int(level_of(i * 10 + 5, st.session_state.th))][1] for i in range(10)])))
    show(fig, "rk_hist", 240, False)


def page_compare(df, m):
    note()
    dims = DIMS[m]
    dim = st.selectbox("Compare by", [d[1] for d in dims], key=f"cmp_dim_{m}")
    col = dict((d[1], d[0]) for d in dims)[dim]
    g = dict(groups(df, col))
    if len(g) < 2:
        st.info("Only one group in this selection. Clear a filter or pick another dimension to compare cohorts.")
    else:
        keys = list(g)
        c1, c2 = st.columns(2)
        a = c1.selectbox("Group A", keys, index=0, key=f"cmp_a_{m}_{dim}")
        b = c2.selectbox("Group B", [k for k in keys if k != a], key=f"cmp_b_{m}_{dim}_{a}")
        A, B = g[a], g[b]
        rows = [("People", len(A), len(B)), ("Avg risk", A["risk"].mean(), B["risk"].mean()),
                ("High / Critical %", (A["risk"] >= st.session_state.th[1]).mean() * 100, (B["risk"] >= st.session_state.th[1]).mean() * 100),
                ("Avg performance", A["perf"].mean(), B["perf"].mean()), ("Avg attendance", A["att"].mean(), B["att"].mean()), ("Avg engagement", A["eng"].mean(), B["eng"].mean())]
        t = pd.DataFrame([{"Metric": n, a: x, b: y, "Difference": (x - y) if pd.notna(x) and pd.notna(y) else np.nan} for n, x, y in rows])
        left, right = st.columns([3, 2])
        left.dataframe(t.round(1), hide_index=True, use_container_width=True)
        with right:
            fig = go.Figure([go.Bar(name=k, x=["Risk", "Performance", "Attendance", "Engagement"], y=[S["risk"].mean(), S["perf"].mean(), S["att"].mean(), S["eng"].mean()], marker=dict(color=c)) for k, S, c in ((a, A, "#4f8cff"), (b, B, "#f472b6"))])
            fig.update_layout(barmode="group")
            show(fig, "cmp_bar", 260)
    st.markdown("**Correlation matrix** (Pearson r between metrics; needs 3+ complete records)")
    cm = correlation_matrix(df, m)
    fig = go.Figure(go.Heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), zmin=-1, zmax=1, zmid=0, colorscale="RdBu", text=cm.round(2).values, texttemplate="%{text}"))
    show(fig, "cmp_corr", 340, False)


def page_performance(df, m):
    note()
    hi = df[df["risk"] >= st.session_state.th[1]]
    a, b = st.columns([2, 1])
    with a:
        card("Performance trend", "Average score by month")
        trend_lines(df, "perf", "pf_tr", [("All", df, "#4f8cff", True), (f"Not at risk (<{st.session_state.th[1]})", df[df["risk"] < st.session_state.th[1]], "#34d399", False), (f"High risk ({st.session_state.th[1]}+)", hi, "#f87171", False)])
    with b:
        card("Largest declines", "Drop in score over the period")
        d = df.assign(drop=df["perf_t1"] - df["perf_t6"]).dropna(subset=["drop"]).sort_values("drop", ascending=False).head(6)
        if d.empty:
            st.caption("No trend data in this dataset.")
        for r in d.itertuples():
            st.markdown(f"<div class='row'><span class='dot' style='background:{LEVELS[int(r.lv)][1]}'>{int(r.perf_t6)}</span><div><b>{esc(r.name)}</b><br><small>{int(r.perf_t1)} → {int(r.perf_t6)} ({int(r.perf_t6 - r.perf_t1)})</small></div></div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        card(f"Average performance by {DIMS[m][0][1].lower()}")
        dim_bar(df, "grp", "perf", "Avg performance", "pf_dim")
    with b:
        if m == "edu":
            card("Score distribution")
            bins = [int((np.minimum(9, df["perf"].dropna() // 10) == i).sum()) for i in range(10)]
            show(go.Figure(go.Bar(x=[f"{i * 10}s" for i in range(10)], y=bins, marker=dict(color="#9b6bff"))), "pf_hist", 260, False)
        else:
            card(f"Productivity by {DIMS[m][0][1].lower()}")
            dim_bar(df, "grp", "prod", "Avg productivity", "pf_prod")


def page_attendance(df, m):
    note()
    hi = df[df["risk"] >= st.session_state.th[1]]
    a, b = st.columns([2, 1])
    with a:
        card("Attendance trend", "Average % by month")
        trend_lines(df, "att", "at_tr", [("All", df, "#22d3ee", True), (f"High risk ({st.session_state.th[1]}+)", hi, "#f87171", False)])
    with b:
        card("Lowest attendance", "Follow-up candidates")
        for r in df.dropna(subset=["att"]).sort_values("att").head(6).itertuples():
            extra = f"{fmt_num(r.missed)} classes missed, {fmt_num(r.late)} late" if m == "edu" else f"{fmt_num(r.absd)} absence days/yr (est.)"
            st.markdown(f"<div class='row'><span class='dot' style='background:{LEVELS[int(r.lv)][1]}'>{int(r.att)}</span><div><b>{esc(r.name)}</b><br><small>{extra}</small></div></div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        card(f"Average attendance by {DIMS[m][0][1].lower()}")
        dim_bar(df, "grp", "att", "Avg attendance", "at_d1")
    with b:
        card(f"Average attendance by {DIMS[m][1][1].lower()}")
        dim_bar(df, DIMS[m][1][0], "att", "Avg attendance", "at_d2")


def page_insights(df, m):
    s = st.session_state
    note()
    al = s.up.get("alerts") if s.up else None
    if not s.up:
        card("Threshold alerts", "", "<span class='sub'>Replace the file with a newer one, or add an earlier snapshot, to see who crossed the High threshold between datasets.</span>")
    elif al is None:
        card("Threshold alerts", "", "<span class='sub'>No earlier data to compare with yet. Replace the file with a newer one or add an earlier snapshot.</span>")
    elif not al:
        card("Threshold alerts", "", "<span style='color:#34d399'>No one crossed the High threshold since the earlier data.</span>")
    else:
        card(f"Threshold alerts ({len(al)})", "People whose risk crossed into High or above", "".join(f"<div class='row'><b>{esc(a['name'])}</b><span class='sub'>risk rose from {int(a['prev_risk'])} to {int(a['risk'])}</span></div>" for a in al[:8]))
    card("Insights", "Calculated from the current selection", "".join(f"<div class='row'>✧ {esc(t)}</div>" for t in build_insights(df, m, s.th)))
    st.markdown("#### " + ("Academic impact indicators" if m == "edu" else "Business impact"))
    st.caption("Illustrative estimates based on synthetic/demo data, not financial predictions.")
    items = impact_items(df, m, s.th)
    for col, (k, v, sub) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='ari-card'><small>{esc(k)}</small><b style='display:block;font-size:20px'>{esc(v)}</b><span class='sub' style='font-size:11px'>{esc(sub)}</span></div>", unsafe_allow_html=True)
    st.markdown("#### Intervention Center")
    many = "students" if m == "edu" else "employees"
    t = st.columns(3)
    tiers = [("🔴 Critical", 3, "#f87171", f"{{n}} {many} may need immediate attention."), ("🟠 High", 2, "#fb923c", f"{{n}} {many} show elevated risk."), ("🟡 Moderate", 1, "#fbbf24", "{n} people show early warning indicators.")]
    for col, (title, lv, color, txt) in zip(t, tiers):
        col.markdown(f"<div class='tier' style='border-color:{color}66;background:{color}12'><b>{title}</b><br>{txt.format(n=int((df['lv'] == lv).sum()))}</div>", unsafe_allow_html=True)
    top = df.sort_values("risk", ascending=False).head(10)
    st.caption("Suggested actions are starting points for a human review, not guaranteed solutions.")
    st.dataframe(pd.DataFrame({"Name": top["name"], "Risk": top["risk"].astype(int), "Level": [LEVELS[i][0] for i in top["lv"]], "Main issue": top["main"],
                               "Suggested action": top["action"], "Priority": top["priority"], "Status": [status_of(i)["s"] for i in top["id"]]}), hide_index=True, use_container_width=True)


def page_reports(df, m):
    s = st.session_state
    note()
    dim = DIMS[m][0]
    t = summary_table(df, dim[0])
    st.caption(f"Summary by {dim[1].lower()} — {len(df)} records in the current selection")
    if len(t):
        shown = t.assign(**{"Avg performance": t["Avg performance"].map(lambda v: fmt_pct(v, s.nf)), "Avg attendance": t["Avg attendance"].map(lambda v: fmt_pct(v, s.nf))}).rename(columns={"Group": dim[1]})
        st.dataframe(shown, hide_index=True, use_container_width=True)
    export = df[["id", "name", "grp", "perf", "att", "eng", "risk", "main", "action"]].assign(level=[LEVELS[i][0] for i in df["lv"]])
    c = st.columns(3)
    c[0].download_button("Export CSV", export.to_csv(index=False), f"at-risk-{m}.csv", "text/csv", use_container_width=True)
    buf = io.BytesIO()
    export.to_excel(buf, index=False, sheet_name="At-risk")
    c[1].download_button("Export Excel", buf.getvalue(), f"at-risk-{m}.xlsx", use_container_width=True)
    c[2].download_button("Download snapshot (HTML)", snapshot_html(df, m, s.th, "Uploaded dataset" if s.up else "Synthetic demo data", s.nf), "at-risk-snapshot.html", "text/html", use_container_width=True)
    st.caption("Tip: use your browser's Print → Save as PDF for a PDF copy of any page.")


def reset_model():
    s = st.session_state
    s.weights = {k: list(v) for k, v in DEFAULT_WEIGHTS.items()}
    s.th = list(DEFAULT_TH)
    for key in [k for k in list(s.keys()) if k.startswith(("w_", "th_"))]:
        del s[key]
    rescore_all()


def page_settings(m):
    s = st.session_state
    note()
    a, b = st.columns([2, 3])
    with a:
        st.markdown("**Preferences**")
        s.nf = st.selectbox("Number format", ["pct", "dec"], index=["pct", "dec"].index(s.nf), format_func={"pct": "Percent (85%)", "dec": "Decimal (0.85)"}.get)
        if st.button("Regenerate synthetic data", use_container_width=True):
            s.seed = int(np.random.randint(1, 10 ** 6))
            s.demo = generate_demo(s.seed, s.weights, s.th)
            s.data, s.up = dict(s.demo), None
            try:
                DATA_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()
    with b:
        st.markdown(f"**Risk model ({'students' if m == 'edu' else 'employees'})**")
        st.caption("Drag to change how much each factor counts. Weights are re-balanced to total 100% and scores update immediately.")
        raw = [st.slider(f.name, 0, 100, int(round(w * 100)), key=f"w_{m}_{i}") for i, (f, w) in enumerate(zip(FACTORS[m], s.weights[m]))]
        tot = sum(raw)
        new_w = [x / tot for x in raw] if tot else s.weights[m]
        st.caption(" • ".join(f"{f.name}: {round(w * 100)}%" for f, w in zip(FACTORS[m], new_w)))
        c = st.columns(3)
        th = [c[i].number_input(n, 1, 99, int(s.th[i]), key=f"th_{i}") for i, n in enumerate(["Moderate from", "High from", "Critical from"])]
        if not (th[0] < th[1] < th[2]):
            st.error("Bands must increase: Moderate < High < Critical.")
            th = s.th
        changed = (new_w != s.weights[m]) or (th != s.th)
        if changed:
            s.weights[m], s.th = new_w, th
            rescore_all()
            st.rerun()
        st.button("Reset model to defaults", on_click=reset_model)
    persist()


PAGES = {"Overview": page_overview, "Students": page_people, "Employees": page_people, "Risk Analysis": page_risk, "Compare": page_compare,
         "Performance": page_performance, "Attendance": page_attendance, "Insights": page_insights, "Reports": page_reports}


def run_app():
    global st, go
    import streamlit as _st
    import plotly.graph_objects as _go
    st, go = _st, _go
    st.set_page_config(page_title="AT-RISK INTELLIGENCE", page_icon="◧", layout="wide", initial_sidebar_state="expanded")
    init_state()
    s = st.session_state
    st.markdown(theme_css(s.theme), unsafe_allow_html=True)
    try:
        import streamlit.components.v1 as components
        components.html(GLOW_JS, height=0)
    except Exception:
        pass
    mode_label = sidebar()
    m = "edu" if mode_label == "Education" else "corp"
    st.markdown(f"## {s.nav if s.nav not in ('Students', 'Employees') else ('Students' if m == 'edu' else 'Employees')}")
    if s.pending is not None:
        review_panel()
        st.stop()
    if s.nav == "Settings":
        page_settings(m)
    else:
        full = s.data[m]
        if full is None or len(full) == 0:
            other = "students" if s.up and s.up["mode"] == "edu" else "employees"
            st.info(f"The uploaded dataset contains {other} only. Switch the mode in the sidebar or reset to the demo data.")
        else:
            df = filter_bar(m, full)
            if df.empty:
                st.markdown("<div class='ari-card' style='text-align:center;padding:50px'><b>No records match these filters</b><br><span class='sub'>Adjust or reset the filters to see data again.</span></div>", unsafe_allow_html=True)
                st.button("Reset filters", on_click=reset_filters, key="empty_reset")
            else:
                PAGES[s.nav](df, m)
    st.caption("Demo environment • Synthetic data • Analytics for decision support")


if __name__ == "__main__":
    try:
        run_app()
    except ModuleNotFoundError as e:                       # plain `python file.py` or missing packages
        print(f"Missing package: {e.name}\nInstall with:  pip install 'streamlit>=1.37' plotly pandas numpy openpyxl\nThen run:       streamlit run {Path(__file__).name}")
