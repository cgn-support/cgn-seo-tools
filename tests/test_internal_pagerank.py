"""Verification suite for pages/internal-pagerank.py.

Dependency-free: run with `python tests/test_internal_pagerank.py`. Exits
non-zero if any check fails.

Covers URL normalisation, Screaming Frog column detection, link-type and
placement classification, redirect/canonical consolidation, PageRank against a
slow reference implementation, BFS click depth, degenerate inputs and a
performance smoke test.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAGE = Path(__file__).resolve().parent.parent / "pages" / "internal-pagerank.py"
spec = importlib.util.spec_from_file_location("ipr", PAGE)
ipr = importlib.util.module_from_spec(spec)
sys.modules["ipr"] = ipr
spec.loader.exec_module(ipr)

FAIL = []


def check(name: str, cond: object, extra: object = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        FAIL.append(name)


# ── URL normalisation ─────────────────────────────────────────────────────────
cfg = ipr.PRConfig()
def n(u: object, c: ipr.PRConfig = cfg) -> str:
    return ipr._normalize_url(u, c)
check("fragment stripped", n("https://ex.com/a#gallery") == "https://ex.com/a")
check("host lowercased", n("HTTPS://Ex.COM/a") == "https://ex.com/a")
check("default port dropped", n("https://ex.com:443/a") == "https://ex.com/a")
check("nonstd port kept", n("https://ex.com:8443/a") == "https://ex.com:8443/a")
check("utm stripped", n("https://ex.com/a?utm_source=fb&id=3") == "https://ex.com/a?id=3")
check("gclid stripped", n("https://ex.com/a?gclid=x") == "https://ex.com/a")
check("empty path -> /", n("https://ex.com") == "https://ex.com/")
check("trailing slash kept by default", n("https://ex.com/a/") == "https://ex.com/a/")
check("trailing slash unified when on",
      n("https://ex.com/a/", ipr.PRConfig(unify_trailing_slash=True)) == "https://ex.com/a")
check("non-string -> empty", n(float("nan")) == "" and n(None) == "")
check("relative left alone", n("/about") == "/about")
check("strip_all_query", n("https://ex.com/a?id=3", ipr.PRConfig(strip_all_query=True))
      == "https://ex.com/a")

# ── value parsers ─────────────────────────────────────────────────────────────
check("rel sponsored blocked", ipr._is_follow_value("noopener sponsored") is False)
check("rel ugc blocked", ipr._is_follow_value("ugc") is False)
check("rel nofollow blocked", ipr._is_follow_value("nofollow") is False)
check("rel noopener unknown", ipr._is_follow_value("noopener noreferrer") is None)
check("follow true", ipr._is_follow_value("true") is True)
check("follow false", ipr._is_follow_value("False") is False)
check("nan unknown", ipr._is_follow_value(float("nan")) is None)

check("type hyperlink", ipr._classify_link_type("Hyperlink") == "hyperlink")
check("type http redirect", ipr._classify_link_type("HTTP Redirect") == "redirect")
check("type meta refresh", ipr._classify_link_type("Meta Refresh") == "redirect")
check("type canonical", ipr._classify_link_type("Canonical") == "canonical")
check("type image", ipr._classify_link_type("Image") == "image")
check("type css", ipr._classify_link_type("CSS") == "css")
check("type js", ipr._classify_link_type("JavaScript") == "javascript")
check("type relnext", ipr._classify_link_type("Rel Next") == "pagination_rel")
check("type blank", ipr._classify_link_type("") == "unknown")

check("pos content", ipr._classify_position("Content") == "content")
check("pos header not head", ipr._classify_position("Header") == "header")
check("pos head", ipr._classify_position("Head") == "head")
check("pos footer", ipr._classify_position("Footer") == "footer")
check("pos nav", ipr._classify_position("Navigation") == "navigation")
check("pos aside", ipr._classify_position("Aside") == "sidebar")
check("pos unknown", ipr._classify_position(None) == "unknown")

# ── domain scoping (the old substring bug) ────────────────────────────────────
check("same host", ipr._same_site("https://ex.com/a", "ex.com"))
check("subdomain internal", ipr._same_site("https://blog.ex.com/a", "ex.com"))
check("lookalike rejected", not ipr._same_site("https://evil-ex.com/a", "ex.com"))
check("suffix spoof rejected", not ipr._same_site("https://ex.com.evil.ru/a", "ex.com"))
check("domain in query rejected",
      not ipr._same_site("https://other.com/a?ref=ex.com", "ex.com"))

# ── chain flattening / cycles ────────────────────────────────────────────────
flat, cyc = ipr._resolve_chains({"a": "b", "b": "c", "c": "d"})
check("chain flattened", flat == {"a": "d", "b": "d", "c": "d"}, flat)
flat, cyc = ipr._resolve_chains({"a": "b", "b": "a"})
check("cycle detected", set(cyc) == {"a", "b"}, cyc)
flat, cyc = ipr._resolve_chains({"a": "b", "b": "c", "c": "a"})
check("3-cycle detected", set(cyc) == {"a", "b", "c"} and flat == {}, (flat, cyc))

# The result must be idempotent: no resolved destination may itself be a key, or
# a single .map() pass would leave some links on an intermediate URL and others
# on the final one, silently splitting one page into two nodes.
def _idempotent(mapping: dict) -> bool:
    resolved, _ = ipr._resolve_chains(mapping)
    return not (set(resolved.values()) & set(resolved))


long_chain = {f"u{i}": f"u{i + 1}" for i in range(40)}
flat, cyc = ipr._resolve_chains(long_chain)
check("40-hop chain fully flattened", flat.get("u0") == "u40", flat.get("u0"))
check("long chain idempotent", _idempotent(long_chain))
# Chain running into a cycle: a→b→c→b. 'a' must not resolve onto a loop member.
lead_in = {"a": "b", "b": "c", "c": "b"}
flat, cyc = ipr._resolve_chains(lead_in)
check("chain into cycle left unresolved", "a" not in flat, flat)
check("chain into cycle idempotent", _idempotent(lead_in))
check("self-map reported as cycle", ipr._resolve_chains({"a": "a"}) == ({}, ["a"]))

# ── PageRank maths ────────────────────────────────────────────────────────────
# Known 4-node case, uniform weights: compare to a slow reference implementation.
src = np.array([0, 0, 1, 2, 3, 3])
dst = np.array([1, 2, 2, 0, 0, 1])
w = np.ones(6)
pr, iters, delta = ipr.pagerank(4, src, dst, w, damping=0.85, max_iters=500, tol=1e-14)
check("pagerank sums to 1", abs(pr.sum() - 1.0) < 1e-12, pr.sum())


def reference(n_nodes, edges, d=0.85, iters=2000):
    out = [[] for _ in range(n_nodes)]
    for s, t in edges:
        out[s].append(t)
    p = [1.0 / n_nodes] * n_nodes
    for _ in range(iters):
        new = [(1 - d) / n_nodes] * n_nodes
        dang = sum(p[i] for i in range(n_nodes) if not out[i])
        for j in range(n_nodes):
            new[j] += d * dang / n_nodes
        for i in range(n_nodes):
            if out[i]:
                share = d * p[i] / len(out[i])
                for j in out[i]:
                    new[j] += share
        p = new
    return p


ref = reference(4, list(zip(src.tolist(), dst.tolist(), strict=True)))
check("matches reference impl", np.allclose(pr, ref, atol=1e-9), f"{pr} vs {ref}")

# Dangling node handling
src2 = np.array([0, 1])
dst2 = np.array([1, 2])  # node 2 is dangling
pr2, _, _ = ipr.pagerank(3, src2, dst2, np.ones(2), max_iters=500, tol=1e-14)
ref2 = reference(3, [(0, 1), (1, 2)])
check("dangling matches reference", np.allclose(pr2, ref2, atol=1e-9), f"{pr2} vs {ref2}")

# Empty graph / single node
pr3, i3, d3 = ipr.pagerank(0, np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                           np.array([]))
check("empty graph safe", len(pr3) == 0)
pr4, _, _ = ipr.pagerank(2, np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                         np.array([]))
check("no edges -> uniform", np.allclose(pr4, [0.5, 0.5]), pr4)

# Weighting actually shifts equity toward the heavier link
srcw = np.array([0, 0])
dstw = np.array([1, 2])
heavy, _, _ = ipr.pagerank(3, srcw, dstw, np.array([1.0, 0.2]), max_iters=500, tol=1e-14)
check("weight shifts equity", heavy[1] > heavy[2], f"{heavy[1]:.5f} vs {heavy[2]:.5f}")
even, _, _ = ipr.pagerank(3, srcw, dstw, np.array([1.0, 1.0]), max_iters=500, tol=1e-14)
check("equal weights -> equal split", abs(even[1] - even[2]) < 1e-12)

# ── click depth ──────────────────────────────────────────────────────────────
depth = ipr.click_depth(5, np.array([0, 1, 2]), np.array([1, 2, 3]), 0)
check("bfs depths", depth.tolist() == [0, 1, 2, 3, -1], depth.tolist())
check("bad start", (ipr.click_depth(3, src2, dst2, -1) == -1).all())

# ── full pipeline on a synthetic SF export ───────────────────────────────────
H = "https://ex.com/"
rows = [
    # type, source, destination, status, follow, position, anchor, alt
    ("Hyperlink", H, "https://ex.com/kitchens", "200", "true", "Content", "Kitchens", ""),
    ("Hyperlink", H, "https://ex.com/baths", "200", "true", "Navigation", "Baths", ""),
    ("Hyperlink", H, "https://ex.com/baths", "200", "true", "Content", "Baths", ""),   # dup
    ("Hyperlink", H, "https://ex.com/privacy", "200", "true", "Footer", "Privacy", ""),
    ("Hyperlink", H, "https://ex.com/gone", "404", "true", "Content", "Gone", ""),
    ("Hyperlink", H, "https://ex.com/old-kitchens", "301", "true", "Content", "Old", ""),
    ("HTTP Redirect", "https://ex.com/old-kitchens", "https://ex.com/kitchens", "200",
     "true", "", "", ""),
    ("Hyperlink", H, "https://ex.com/spam", "200", "false", "Content", "Spam", ""),
    ("Hyperlink", H, "https://twitter.com/x", "200", "true", "Footer", "Tw", ""),
    ("Hyperlink", H, H, "200", "true", "Content", "Home", ""),                        # self
    ("Image", H, "https://ex.com/logo.png", "200", "true", "Header", "", "Logo"),
    ("CSS", H, "https://ex.com/app.css", "200", "true", "Head", "", ""),
    ("Hyperlink", "https://ex.com/kitchens", H, "200", "true", "Navigation", "Home", ""),
    ("Hyperlink", "https://ex.com/kitchens#gallery", "https://ex.com/baths", "200",
     "true", "Content", "Baths", ""),   # fragment must fold into /kitchens
    ("Hyperlink", "https://ex.com/baths", "https://ex.com/kitchens?utm_source=nl", "200",
     "true", "Content", "K", ""),       # utm must fold into /kitchens
    ("Canonical", "https://ex.com/baths?page=2", "https://ex.com/baths", "200", "true",
     "", "", ""),
    ("Hyperlink", "https://ex.com/baths?page=2", "https://ex.com/kitchens", "200",
     "true", "Content", "K", ""),
    ("Hyperlink", "https://ex.com/privacy", H, "200", "true", "Footer", "Home", ""),
]
df = pd.DataFrame(rows, columns=[
    "Type", "Source", "Destination", "Status Code", "Follow", "Link Position",
    "Anchor", "Alt Text",
])

detected = ipr._detect_columns(list(df.columns))
check("detect source", detected["source"] == "Source", detected["source"])
check("detect target", detected["target"] == "Destination", detected["target"])
check("detect follow", detected["follow"] == "Follow", detected["follow"])
check("detect status", detected["status"] == "Status Code", detected["status"])
check("detect type", detected["link_type"] == "Type", detected["link_type"])
check("detect position", detected["position"] == "Link Position", detected["position"])
check("detect anchor", detected["anchor"] == "Anchor", detected["anchor"])
check("detect alt", detected["alt"] == "Alt Text", detected["alt"])

# SF's bare "Target" column (anchor target attr) must not win over Destination.
d2 = ipr._detect_columns(["Source", "Destination", "Target", "Status Code"])
check("bare Target not chosen as URL", d2["target"] == "Destination", d2["target"])

check("auto domain", ipr._auto_detect_domain(df, "Source") == "ex.com")

edges, status_by_url, diag, rewrite = ipr.build_edges(df, detected, cfg, "ex.com")

pairs = set(zip(edges["source"], edges["target"], strict=True))
check("image row excluded", not any("logo.png" in t for _, t in pairs))
check("css row excluded", not any("app.css" in t for _, t in pairs))
check("nofollow excluded", not any(t.endswith("/spam") for _, t in pairs))
check("external excluded", not any("twitter" in t for _, t in pairs))
check("self loop excluded", (H, H) not in pairs)
check("redirect rewritten to final",
      (H, "https://ex.com/kitchens") in pairs
      and not any("old-kitchens" in u for pair in pairs for u in pair))
check("redirect map recorded",
      rewrite.get("https://ex.com/old-kitchens") == "https://ex.com/kitchens", rewrite)
check("fragment source folded",
      ("https://ex.com/kitchens", "https://ex.com/baths") in pairs)
check("utm target folded",
      ("https://ex.com/baths", "https://ex.com/kitchens") in pairs)
check("canonical consolidated",
      rewrite.get("https://ex.com/baths?page=2") == "https://ex.com/baths"
      and not any("page=2" in u for pair in pairs for u in pair))
check("404 kept as dead end by default",
      any(t.endswith("/gone") for _, t in pairs))
check("dup link merged once",
      sum(1 for s, t in pairs if s == H and t.endswith("/baths")) == 1)

home_baths = edges[(edges["source"] == H) & (edges["target"] == "https://ex.com/baths")]
check("dedupe keeps best placement",
      home_baths.iloc[0]["position"] == "content", home_baths.iloc[0]["position"])
home_priv = edges[(edges["source"] == H) & (edges["target"].str.endswith("/privacy"))]
check("footer weight applied", abs(home_priv.iloc[0]["weight"] - 0.20) < 1e-9,
      home_priv.iloc[0]["weight"])

check("diag nofollow counted", diag.nofollow_links == 1, diag.nofollow_links)
check("diag broken counted", diag.broken_target_links == 1, diag.broken_target_links)
check("diag self loop counted", diag.self_loop_links == 1, diag.self_loop_links)
check("diag external counted", diag.external_links == 1, diag.external_links)
# 3 merges: home->baths twice in the export, plus home->old-kitchens and
# baths?page=2->kitchens becoming duplicates after redirect/canonical consolidation.
check("diag dup counted", diag.duplicate_links == 3, diag.duplicate_links)
check("diag types dropped", diag.dropped_by_link_type.get("image") == 1
      and diag.dropped_by_link_type.get("css") == 1, diag.dropped_by_link_type)
check("head position zero-weighted out", diag.zero_weight_links == 0,
      diag.zero_weight_links)  # css row already removed by type filter

# drop_dead_end_edges variant
cfg_drop = ipr.PRConfig(drop_dead_end_edges=True)
edges_d, _, diag_d, _ = ipr.build_edges(df, detected, cfg_drop, "ex.com")
check("404 removed when toggled",
      not any(t.endswith("/gone") for t in edges_d["target"]))

# full run
data = ipr.run_analysis(df, detected, cfg, "ex.com", "")
res = data["result"]
check("run_analysis produced rows", data is not None and len(res) > 0)
check("pagerank normalised", abs(res["pagerank"].sum() - 1.0) < 1e-9)
check("homepage detected", data["homepage"] == H, data["homepage"])
check("converged", data["converged"], data["final_delta"])
check("kitchens above privacy",
      res.set_index("url").loc["https://ex.com/kitchens", "pagerank"]
      > res.set_index("url").loc["https://ex.com/privacy", "pagerank"])
depths = res.set_index("url")["click_depth"]
check("home depth 0", depths[H] == 0, depths[H])
check("kitchens depth 1", depths["https://ex.com/kitchens"] == 1)
check("is_dead_end flags 404", bool(res.set_index("url").loc["https://ex.com/gone",
                                                             "is_dead_end"]))
check("percentile present", res["pagerank_percentile"].between(0, 100).all())

# priority matching + donors
matched, missing, ambiguous = ipr.match_priority_urls(
    "https://ex.com/kitchens\n/baths\n/does-not-exist\n", list(res["url"]), cfg)
check("priority exact match", matched.get("https://ex.com/kitchens")
      == "https://ex.com/kitchens", matched)
check("priority path match", matched.get("/baths") == "https://ex.com/baths", matched)
check("priority missing reported", missing == ["/does-not-exist"], missing)
check("no false ambiguity", ambiguous == {}, ambiguous)

# Same path on two hosts must be reported as ambiguous, not as missing — the page
# is in the graph, it is just unclear which node was meant.
amb_nodes = ["https://ex.com/dup", "https://blog.ex.com/dup"]
m2, miss2, amb2 = ipr.match_priority_urls("/dup", amb_nodes, cfg)
check("ambiguous path reported separately",
      m2 == {} and miss2 == [] and amb2 == {"/dup": sorted(amb_nodes)}, (m2, miss2, amb2))

donors = ipr.donor_suggestions("https://ex.com/kitchens", res, edges)
check("donor excludes existing linkers", H not in set(donors.get("url", [])))
check("donor excludes dead ends",
      not any("gone" in u for u in donors.get("url", [])))

# ── no-optional-columns export (From/To only) ────────────────────────────────
minimal = pd.DataFrame({
    "From": [H, H, "https://ex.com/a"],
    "To": ["https://ex.com/a", "https://ex.com/b", H],
})
dmin = ipr._detect_columns(list(minimal.columns))
check("minimal detect", dmin["source"] == "From" and dmin["target"] == "To")
d_min = ipr.run_analysis(minimal, dmin, cfg, "ex.com", "")
check("minimal export works", d_min is not None and len(d_min["result"]) == 3,
      None if d_min is None else len(d_min["result"]))
check("minimal all weight 1", (d_min["edges"]["weight"] == 1.0).all())

# ── empty / degenerate input ─────────────────────────────────────────────────
empty = pd.DataFrame({"From": [], "To": []})
check("empty export returns None",
      ipr.run_analysis(empty, ipr._detect_columns(["From", "To"]), cfg, None, "") is None)
one = pd.DataFrame({"From": [H], "To": [H]})
check("single self-loop returns None",
      ipr.run_analysis(one, ipr._detect_columns(["From", "To"]), cfg, None, "") is None)

# ── redirect ambiguity must not be guessed ───────────────────────────────────
amb = pd.DataFrame({
    "Source": [H, "https://ex.com/r", "https://ex.com/r"],
    "Destination": ["https://ex.com/r", "https://ex.com/x", "https://ex.com/y"],
    "Status Code": ["301", "200", "200"],
    "Type": ["Hyperlink", "Hyperlink", "Hyperlink"],
})
_, _, diag_a, rw_a = ipr.build_edges(amb, ipr._detect_columns(list(amb.columns)), cfg,
                                     "ex.com")
check("ambiguous redirect left unresolved", rw_a == {}, rw_a)
check("unresolved redirect reported", diag_a.unresolved_redirect_links == 1,
      diag_a.unresolved_redirect_links)

# unambiguous status-only redirect (no typed row)
una = pd.DataFrame({
    "Source": [H, "https://ex.com/r"],
    "Destination": ["https://ex.com/r", "https://ex.com/x"],
    "Status Code": ["301", "200"],
    "Type": ["Hyperlink", "Hyperlink"],
})
e_u, _, _, rw_u = ipr.build_edges(una, ipr._detect_columns(list(una.columns)), cfg,
                                  "ex.com")
check("single-outlink redirect resolved",
      rw_u == {"https://ex.com/r": "https://ex.com/x"}, rw_u)

# ── performance smoke test ───────────────────────────────────────────────────
rng = np.random.default_rng(0)
N, E = 50_000, 500_000
big = pd.DataFrame({
    "From": [f"https://ex.com/p{i}" for i in rng.integers(0, N, E)],
    "To": [f"https://ex.com/p{i}" for i in rng.integers(0, N, E)],
})
t0 = time.time()
big_data = ipr.run_analysis(big, ipr._detect_columns(["From", "To"]), cfg, "ex.com", "")
elapsed = time.time() - t0
# Build the detail string defensively: it is evaluated before check() is called,
# so indexing a None result here would abort the suite before the exit summary.
check("large export produces a result", big_data is not None)
detail = (
    f"{elapsed:.1f}s, {big_data['n_nodes']:,} nodes, {big_data['iters_used']} iters"
    if big_data is not None else f"{elapsed:.1f}s, no result"
)
# Generous bound: this guards against an O(rows) Python-loop regression (a prior
# revision used iterrows() and took minutes), not against normal runner jitter.
check("50k nodes / 500k edges under 120s", elapsed < 120, detail)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
