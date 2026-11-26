"""Run:  python tests/test_at_risk.py
Part 1 tests the pure-Python logic (needs only pandas/numpy/openpyxl).
Part 2 runs every page of the Streamlit UI against a lightweight stand-in for streamlit/plotly, which catches
Python errors (typos, bad column names, NaN crashes). It does NOT replace opening the real app once."""
import importlib.util, io, sys, tempfile, types
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, HERE.parent, Path.cwd()) if (p / "at_risk_intelligence.py").exists()), None)
if ROOT is None:
    sys.exit("Could not find at_risk_intelligence.py. Put this test file in the same folder as it (or in a 'tests' sub-folder).")
print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
spec = importlib.util.spec_from_file_location("ari", ROOT / "at_risk_intelligence.py")
ari = importlib.util.module_from_spec(spec); sys.modules["ari"] = ari; spec.loader.exec_module(ari)
tmp = Path(tempfile.mkdtemp()); ari.STATE_FILE = tmp / "s.json"; ari.DATA_FILE = tmp / "d.pkl"
fails = 0
def ok(c, m):
    global fails
    fails += (not c); print(("ok   " if c else "FAIL ") + m)

# ---------------- Part 1: logic ----------------
d = ari.generate_demo(1)
ok(len(d["edu"]) == 120 and len(d["corp"]) == 120, "120 students and 120 employees generated")
for m in ("edu", "corp"):
    df = d[m]; ok(df["risk"].between(0, 100).all() and df["risk"].notna().all(), f"{m}: risk within 0-100")
    c = [(df.loc[df["lv"] == i, "risk"].size) for i in range(4)]; ok(all(x > 0 for x in c), f"{m}: all four levels present {c}")
ok(d["edu"]["att"].corr(d["edu"]["risk"]) < -.4, "students: lower attendance -> higher risk")
ok(d["corp"]["eng"].corr(d["corp"]["risk"]) < -.4 and d["corp"]["ot"].corr(d["corp"]["risk"]) > .3, "employees: low engagement / overtime -> higher risk")
w = [0] * 6; w[0] = 1
only_att = ari.score_frame(d["edu"], "edu", w)
ok(only_att["risk"].corr(d["edu"]["att"]) < -.95, "custom weights change scores")
hi = ari.score_frame(d["edu"], "edu", th=[10, 20, 30]); ok((hi["lv"] == 3).sum() > (d["edu"]["lv"] == 3).sum(), "lower thresholds raise levels")

h = ["Student_Name", "Attendance_Percentage", "CGPA", "Engagement_Score", "Department", "Academic_Year"]
ok(ari.detect_mode(h) == ("edu", True), "detects a student file"); ok(ari.auto_map(h, "edu")["gpa"] == "CGPA", "alias mapping (CGPA)")
ok(ari.auto_map(["employee id", "Employee Name", "Dept", "job-title"], "corp")["role"] == "job-title", "tolerates spaces and hyphens")
ok(ari.detect_mode(["a", "b"])[1] is False, "unknown file asks the user")
raw = pd.DataFrame({"Student_Name": ["John", "Alex", "Sam", "<b>x</b>"], "Attendance_Percentage": ["85", "", "91", "250"], "CGPA": ["8.4", "7.1", "", "3"],
                    "Engagement_Score": ["70", "", "88", "20"], "Department": ["CS", "CS", "Arts", "CS"], "Academic_Year": ["2", "3", "1", "4"]})
r, iss = ari.build_records(raw, ari.auto_map(h, "edu"), "edu")
ok(len(r) == 4 and r["risk"].notna().all(), "missing values never give NaN risk"); ok(pd.isna(r.loc[1, "att"]), "blank stays missing (not zero)")
ok(len(iss) == 1 and "250" in iss[0][2], "out-of-range value flagged"); ok(pd.isna(r.loc[3, "att"]), "invalid value excluded")
r2, _ = ari.build_records(pd.DataFrame({"Name": ["A"], "Attendance": ["90"], "Risk Level": ["High"]}), ari.auto_map(["Name", "Attendance", "Risk Level"], "edu"), "edu")
ok(r2.loc[0, "lv"] == 2, "uploaded risk level is respected")
fr, _ = ari.build_records(pd.DataFrame({"n": ["A", "B"], "Attendance": ["0.9", "0.5"], "Performance": ["0.8", "0.4"]}), {"name": "n", "att": "Attendance", "perf": "Performance"}, "edu")
ok(fr["att"].tolist() == [90.0, 50.0], "0-1 fractions are scaled to percent")
try:
    import openpyxl  # noqa: F401
    buf = io.BytesIO(); pd.DataFrame({"Student_Name": ["A", "B", "C"], "Attendance": [0.9, 0.5, None], "CGPA": [8, 5, 7]}).to_excel(buf, index=False)
    x = ari.read_table("t.xlsx", buf.getvalue()); ok(x.shape == (3, 3), "reads .xlsx")
except ImportError:
    print("SKIP .xlsx tests (install openpyxl:  pip install openpyxl)")
for bad, msg in [(("t.csv", b""), "no usable"), (("t.csv", b"onlyone\n1\n2"), "Unable"), (("t.pdf", b"abc"), "valid CSV")]:
    try: ari.read_table(*bad); ok(False, "bad file rejected: " + bad[0])
    except ValueError as e: ok(msg in str(e), f"friendly error for {bad[0]} ({str(e)[:35]}...)")
cls = ROOT / "classroom_50_students.xlsx"
if cls.exists() and "openpyxl" in sys.modules:
    cx = ari.read_table(cls.name, cls.read_bytes()); mp = ari.auto_map(list(cx.columns), "edu"); cd, ci = ari.build_records(cx, mp, "edu")
    ok(len(cd) == 50 and not ci and ari.detect_mode(list(cx.columns)) == ("edu", True), "classroom_50_students.xlsx loads cleanly")
cur = d["edu"].copy(); prev = cur.copy(); prev["risk"] = (prev["risk"] - 25).clip(0); prev["perf"] = prev["perf"] + 5
sn, hit, al = ari.apply_snapshot(cur, prev, True, "edu"); ok(hit == 120 and sn["perf_t1"].notna().all(), "snapshot gives real trend values"); ok(len(al) > 0, f"{len(al)} threshold alerts")
for m in ("edu", "corp"):
    ok(len(ari.build_insights(d[m], m, ari.DEFAULT_TH)) >= 5, f"{m}: insights computed"); ok(len(ari.impact_items(d[m], m, ari.DEFAULT_TH)) == 6, f"{m}: impact metrics")
    ok(len(ari.kpi_items(d[m], m, ari.DEFAULT_TH)) == 6, f"{m}: 6 KPIs"); ok(ari.correlation_matrix(d[m], m).shape == (6, 6), f"{m}: correlation matrix")
    ok("<table" in ari.snapshot_html(d[m], m, ari.DEFAULT_TH, "demo"), f"{m}: snapshot html")
ok(ari.fmt_pct(float("nan")) == "N/A" and ari.fmt_pct(85, "dec") == "0.85", "number formatting")

# ---------------- Part 2: UI smoke test with a stand-in for streamlit / plotly ----------------
class Stub:
    def __init__(s, *a, **k): pass
    def __call__(s, *a, **k): return Stub()
    def __getattr__(s, n): return Stub()
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def __iter__(s): return iter(())
    def __bool__(s): return False
class SS(dict):
    __getattr__ = lambda s, k: s[k] if k in s else (_ for _ in ()).throw(AttributeError(k)); __setattr__ = dict.__setitem__; __delattr__ = dict.__delitem__
class Stop(Exception): pass
class Sel:
    def __init__(s, rows=()): s.selection = types.SimpleNamespace(rows=list(rows))
class StMod(types.ModuleType):
    def __getattr__(s, n): return Stub()
def make_st(state, selected_rows=()):
    m = StMod("streamlit"); m.session_state = state; m.query_params = {}
    def widget(default_fn):
        def f(label, *a, key=None, **k):
            v = state.get(key) if key and key in state else default_fn(label, *a, **k)
            if key and key not in state: state[key] = v
            return v
        return f
    m.radio = widget(lambda l, o, index=0, **k: o[index or 0] if o else None); m.selectbox = widget(lambda l, o, index=0, **k: list(o)[index or 0] if len(list(o)) else None)
    m.toggle = widget(lambda l, value=False, **k: value); m.text_input = widget(lambda l, value="", **k: value); m.text_area = widget(lambda l, value="", **k: value)
    m.slider = widget(lambda l, mn, mx, value=0, **k: value); m.number_input = widget(lambda l, mn, mx, value=0, **k: value)
    W = {"radio", "selectbox", "toggle", "text_input", "text_area", "slider", "number_input", "button", "download_button", "file_uploader", "dataframe"}
    class Col(Stub):
        def __getattr__(s, n): return getattr(m, n) if n in W else Stub()
    m.columns = lambda n, **k: [Col() for _ in range(n if isinstance(n, int) else len(n))]; m.tabs = lambda t: [Stub() for _ in t]
    m.button = lambda *a, **k: False; m.download_button = lambda *a, **k: False; m.file_uploader = lambda *a, **k: None
    m.dataframe = lambda *a, **k: Sel(selected_rows); m.rerun = lambda: None
    def stop(): raise Stop()
    m.stop = stop; m.dialog = lambda *a, **k: (lambda f: f); m.sidebar = Col(); m.expander = lambda *a, **k: Col()
    m.components = types.ModuleType("c"); return m
class Mod(types.ModuleType):
    def __getattr__(s, n): return Stub()

state = SS()
st_mod = make_st(state); gomod = Mod("plotly.graph_objects")
ari.st, ari.go = st_mod, gomod
sys.modules.update({"streamlit": st_mod, "streamlit.components": st_mod.components, "streamlit.components.v1": Mod("c1"), "plotly": Mod("plotly"), "plotly.graph_objects": gomod})
ari.init_state()
def run_pages(label):
    for mode_label, m in (("Education", "edu"), ("Corporate / HR", "corp")):
        state["mode"] = mode_label
        full = state["data"][m]
        if len(full) == 0: continue
        for nav in ari.NAV:
            state["nav"] = nav
            try:
                if nav == "Settings": ari.page_settings(m)
                else: ari.PAGES[nav](ari.filter_bar(m, full), m)
                ok(True, f"{label} / {m} / {nav}")
            except Exception as e:
                import traceback; ok(False, f"{label} / {m} / {nav}: {type(e).__name__}: {e}"); traceback.print_exc()
        try: ari.profile_body(m, full["id"].iloc[0]); ari.profile_body(m, full["id"].iloc[-1]); ok(True, f"{label} / {m} / profile")
        except Exception as e:
            import traceback; ok(False, f"{label} / {m} / profile: {e}"); traceback.print_exc()
run_pages("demo")
class F:
    def __init__(s, n, b): s.name, s.b, s.size = n, b, len(b)
    def getvalue(s): return s.b
for fn in (ROOT / "classroom_50_students.xlsx",):
    if not fn.exists() or "openpyxl" not in sys.modules: continue
    ari.start_upload(F(fn.name, fn.read_bytes())); p = state["pending"]; ok(p is not None and p["mode"] == "edu", "upload: pending review created")
    try: ari.review_panel(); ok(True, "upload: review panel renders")
    except Exception as e: ok(False, f"review panel: {e}")
    ari.finish_upload("edu", p["mp"]); ok(state["up"] is not None and len(state["data"]["edu"]) == 50 and len(state["data"]["corp"]) == 0, "upload: dataset becomes active, other mode empty")
    run_pages("uploaded")
state.views["v1"] = {"mode": "edu", "f": ["All"] * 6, "q": "x"}; state["view_pick"] = "v1"; ari.load_view(); ok(state["search"] == "x", "load saved view")
ari.delete_view(); ok("v1" not in state.views, "delete saved view")
state.weights["edu"][0] = .9; ari.reset_model(); ok(abs(state.weights["edu"][0] - .22) < 1e-9, "reset model to defaults")
ari.start_upload(F("bad.csv", b"a\n1")); ok(state["pending"] is None, "bad upload is rejected without crashing")
print("\nALL PASSED" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
