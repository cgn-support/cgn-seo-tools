"""Streamlit UI rendering checks for pages/internal-pagerank.py.

Streamlit swallows exceptions raised while rendering and shows them in the
browser instead of the terminal, so a broken tab looks fine from a unit test.
`streamlit.testing.v1.AppTest` runs the script for real and exposes anything
that was raised, which is the only way to catch that class of bug here.

Run with `python tests/test_ui_render.py`. Requires streamlit (unlike
tests/test_internal_pagerank.py, which is dependency-free). Skips with exit
code 0 if streamlit is not installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "pages" / "internal-pagerank.py"

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("SKIP  streamlit not installed — UI rendering checks skipped")
    sys.exit(0)

FAIL: list[str] = []


def check(name: str, cond: object, extra: object = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        FAIL.append(name)


# A synthetic All Inlinks export covering every branch the render code has to
# handle: broken targets, redirects, canonicals, nofollow, external links,
# non-hyperlink rows, an orphan, and a page unreachable from the homepage.
HARNESS = f'''
import importlib.util, sys
import pandas as pd
import streamlit as st

spec = importlib.util.spec_from_file_location("ipr", r"{PAGE}")
ipr = importlib.util.module_from_spec(spec)
sys.modules["ipr"] = ipr
spec.loader.exec_module(ipr)

H = "https://ex.com/"
rows = [
    ("Hyperlink", H, "https://ex.com/kitchens", "200", "true", "Content", "Kitchens", ""),
    ("Hyperlink", H, "https://ex.com/baths", "200", "true", "Navigation", "Baths", ""),
    ("Hyperlink", H, "https://ex.com/baths", "200", "true", "Content", "Baths", ""),
    ("Hyperlink", H, "https://ex.com/privacy", "200", "true", "Footer", "Privacy", ""),
    ("Hyperlink", H, "https://ex.com/gone", "404", "true", "Content", "Gone", ""),
    ("Hyperlink", H, "https://ex.com/old", "301", "true", "Content", "Old", ""),
    ("HTTP Redirect", "https://ex.com/old", "https://ex.com/kitchens", "200",
     "true", "", "", ""),
    ("Hyperlink", H, "https://ex.com/spam", "200", "false", "Content", "Spam", ""),
    ("Hyperlink", H, "https://twitter.com/x", "200", "true", "Footer", "Tw", ""),
    ("Image", H, "https://ex.com/logo.png", "200", "true", "Header", "", "Logo"),
    ("CSS", H, "https://ex.com/app.css", "200", "true", "Head", "", ""),
    ("Hyperlink", "https://ex.com/kitchens", H, "200", "true", "Navigation", "Home", ""),
    ("Hyperlink", "https://ex.com/baths", "https://ex.com/kitchens", "200", "true",
     "Content", "K", ""),
    ("Hyperlink", "https://ex.com/privacy", H, "200", "true", "Footer", "Home", ""),
    ("Hyperlink", "https://ex.com/orphan", "https://ex.com/deep", "200", "true",
     "Content", "d", ""),
    ("Canonical", "https://ex.com/baths?page=2", "https://ex.com/baths", "200",
     "true", "", "", ""),
]
df = pd.DataFrame(rows, columns=[
    "Type", "Source", "Destination", "Status Code", "Follow", "Link Position",
    "Anchor", "Alt Text",
])
colmap = ipr._detect_columns(list(df.columns))
cfg = ipr.PRConfig()
data = ipr.run_analysis(df, colmap, cfg, "ex.com", "")
res = data["result"]

# /deep is reachable only from an orphan, so it is unreachable from the homepage.
matched, missing, ambiguous = ipr.match_priority_urls(
    "https://ex.com/kitchens\\n/privacy\\n/deep", list(res["url"]), cfg
)
priority = res[res["url"].isin(set(matched.values()))].copy()

st.write("MARK overview")
ipr._render_overview(data, priority)
st.write("MARK opportunities")
ipr._render_opportunities(data, priority)
st.write("MARK waste")
ipr._render_waste(data)
st.write("MARK results")
ipr._render_results(data, 500)
st.markdown(ipr.METHOD_NOTES)

# The no-priority-pages branches must render too.
st.write("MARK empty")
ipr._render_overview(data, priority.iloc[0:0])
ipr._render_opportunities(data, priority.iloc[0:0])
st.write("MARK done")
'''

# 1) The page itself must render up to the file_uploader gate.
page = AppTest.from_file(str(PAGE), default_timeout=120).run()
check("page renders without exception", not page.exception,
      [e.value for e in page.exception])
sidebar_widgets = (
    len(page.sidebar.checkbox) + len(page.sidebar.slider)
    + len(page.sidebar.text_input) + len(page.sidebar.number_input)
)
check("sidebar controls render", sidebar_widgets > 15, sidebar_widgets)

# 2) Every render function must survive real data.
app = AppTest.from_string(HARNESS, default_timeout=180).run()
check("render functions raise nothing", not app.exception,
      [e.value for e in app.exception])

body = [str(m.value) for m in app.markdown]
check("all render sections reached", any("MARK done" in b for b in body))

metrics = {m.label: m.value for m in app.metric}
check("overview metrics present",
      {"URLs in graph", "Links counted", "Orphaned URLs"} <= set(metrics), metrics)
check("orphan detected", metrics.get("Orphaned URLs") == "1", metrics)
# /deep is unreachable from the homepage (click_depth -1); it must be counted as
# buried, not silently pass a "> 3" test.
check("unreachable priority page counted as buried",
      metrics.get("Buried deeper than 3 clicks") == "1", metrics)
check("wasted-equity share reported",
      metrics.get("Links passing no equity", "").endswith("%"), metrics)

check("no deprecation warnings from our widgets",
      not any("use_container_width" in str(w.value) for w in app.warning),
      [w.value for w in app.warning])

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
