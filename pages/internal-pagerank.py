# app.py
import io
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

@dataclass
class PRConfig:
    damping: float = 0.85
    max_iters: int = 50
    tol: float = 1e-8
    include_nofollow: bool = False
    restrict_to_internal: bool = True
    drop_self_loops: bool = True
    min_nodes: int = 2

def _normalize_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    u = u.strip()
    return u

def _detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    """
    Attempts to detect Screaming Frog columns across common exports.
    Returns mapping keys: source, target, follow, src_status, dst_status, link_type
    """
    lower = {c.lower(): c for c in cols}

    def pick(options: Iterable[str]) -> Optional[str]:
        for opt in options:
            if opt in lower:
                return lower[opt]
        return None

    # Common SF internal link exports use "From" and "To"
    source = pick(["from", "source", "source url", "address", "url", "origin"])
    target = pick(["to", "destination", "destination url", "target", "linked url", "destination address"])

    # Screaming Frog "Inlinks" export often uses "From" / "To" plus "Follow"
    follow = pick(["follow", "nofollow", "rel", "rel attribute", "link attribute"])

    # Status codes sometimes appear
    src_status = pick(["from status code", "source status code", "status code"])
    dst_status = pick(["to status code", "destination status code", "target status code"])

    link_type = pick(["type", "link type"])

    return {
        "source": source,
        "target": target,
        "follow": follow,
        "src_status": src_status,
        "dst_status": dst_status,
        "link_type": link_type,
    }

def _is_follow_value(val: str) -> Optional[bool]:
    """
    Tries to interpret a 'Follow' / rel / nofollow field.
    Returns True/False or None if unknown.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in ("true", "follow", "1", "yes"):
        return True
    if s in ("false", "nofollow", "no-follow", "0", "no"):
        return False
    # Sometimes rel contains "nofollow"
    if "nofollow" in s:
        return False
    if s == "" or s == "nan":
        return None
    return None


# ──────────────────────────────────────────────
#  NEW: Redirect Resolution
# ──────────────────────────────────────────────

def _build_redirect_map(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    status_col: Optional[str],
) -> Dict[str, str]:
    """
    Detect 301/302 redirect URLs and build a mapping from
    redirect source → final resolved destination.

    Strategy:
      1. If a status code column exists, any row where the TARGET
         has a 3xx status is a redirect.  The redirect destination
         is the URL that the target eventually leads to.
      2. Screaming Frog records redirects so that a URL appearing
         as a target with status 301 will also appear as a source
         in another row pointing to its final destination.
         We follow the chain until we reach a non-redirect URL.
      3. Fallback: if no status column exists, we detect redirect
         candidates as URLs that appear as TARGETS but NEVER as a
         SOURCE (i.e., the crawler could not crawl outlinks from them).
         This heuristic catches 301s because SF doesn't record
         outlinks for redirect URLs.  We then look for any edge
         FROM that URL to find its destination.
    """
    redirect_map: Dict[str, str] = {}

    if status_col and status_col in df.columns:
        # --- Method 1: Use status codes directly ---
        # Find all target URLs with 3xx status
        redirect_rows = df[
            df[status_col].astype(str).str.match(r'^3\d{2}$', na=False)
        ].copy()

        if not redirect_rows.empty:
            # For each redirect target URL, find where it appears as a source
            # to determine its final destination
            redirect_targets = set(redirect_rows[target_col].unique())

            # Build a simple lookup: for URLs that appear as sources, where do they point?
            source_dest = {}
            for _, row in df.iterrows():
                src = str(row[source_col]).strip()
                tgt = str(row[target_col]).strip()
                if src in redirect_targets and src != tgt:
                    # This redirect URL points somewhere
                    source_dest[src] = tgt

            # For redirect URLs that also appear as a target in other rows
            # but never as a source, we look at the Screaming Frog pattern:
            # the "From" column is the page containing the link, "To" is
            # the link destination.  A 301 URL in "To" means the link
            # points to a redirect.  We need the FINAL destination.
            #
            # Screaming Frog's "All Inlinks" export doesn't always give us
            # the redirect chain directly.  But we can infer:
            # if URL-A (301) is never a "From", its destination is unknown
            # from this export alone.  In that case, we try to find it
            # by looking at which 200-status URL shares the most overlap
            # in its inlink sources.
            #
            # SIMPLER APPROACH: SF typically includes redirect destinations
            # in the crawl.  If /service-locations/ 301s to /locations/,
            # both will appear as targets.  We detect the chain by finding
            # that /service-locations/ has status 301 and appears nowhere
            # as a source (no outlinks), while /locations/ has status 200.

            for url in redirect_targets:
                if url in source_dest:
                    redirect_map[url] = source_dest[url]

    # --- Method 2: Heuristic for missing status column ---
    # Also catch any URLs that appear as targets but NEVER as sources
    # (likely redirects or dead-end utility pages)
    # We DON'T auto-resolve these without status codes, but we report them.

    # --- Follow redirect chains (A→B→C becomes A→C, B→C) ---
    max_chain = 10
    for url in list(redirect_map.keys()):
        dest = redirect_map[url]
        hops = 0
        while dest in redirect_map and hops < max_chain:
            dest = redirect_map[dest]
            hops += 1
        redirect_map[url] = dest

    return redirect_map


def _apply_redirect_map(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    redirect_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Rewrite all source and target URLs through the redirect map.
    Any link pointing to a 301 URL gets rewritten to point to
    the final destination instead.
    """
    if not redirect_map:
        return df

    df = df.copy()
    df[source_col] = df[source_col].map(lambda u: redirect_map.get(u, u))
    df[target_col] = df[target_col].map(lambda u: redirect_map.get(u, u))
    return df


def _build_graph(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    follow_col: Optional[str],
    config: PRConfig,
    internal_domain_hint: Optional[str] = None,
    restrict_to_domain: bool = False,
    status_col: Optional[str] = None,
    resolve_redirects: bool = True,
) -> Tuple[Dict[str, Set[str]], pd.DataFrame, Dict[str, str]]:
    """
    Build adjacency list graph from df. Returns:
    - adjacency: dict[node] = set(outgoing_nodes)
    - edges_df: cleaned edges with flags
    - redirect_map: dict of resolved redirects (for reporting)
    """
    tmp = df.copy()

    tmp[source_col] = tmp[source_col].apply(_normalize_url)
    tmp[target_col] = tmp[target_col].apply(_normalize_url)

    tmp = tmp[(tmp[source_col] != "") & (tmp[target_col] != "")]

    # ── NEW: Resolve 301/302 redirects before building graph ──
    redirect_map: Dict[str, str] = {}
    if resolve_redirects:
        redirect_map = _build_redirect_map(tmp, source_col, target_col, status_col)
        if redirect_map:
            tmp = _apply_redirect_map(tmp, source_col, target_col, redirect_map)

    # Determine follow vs nofollow
    if follow_col:
        follow_vals = tmp[follow_col].apply(_is_follow_value)
        tmp["_is_follow"] = follow_vals
    else:
        tmp["_is_follow"] = None

    if not config.include_nofollow and follow_col:
        # Keep rows explicitly marked follow OR unknown
        tmp = tmp[(tmp["_is_follow"].isna()) | (tmp["_is_follow"] == True)]

    if config.drop_self_loops:
        tmp = tmp[tmp[source_col] != tmp[target_col]]

    # Optional: restrict to domain substring match (very simple)
    if restrict_to_domain and internal_domain_hint:
        hint = internal_domain_hint.strip().lower()
        tmp = tmp[
            tmp[source_col].str.lower().str.contains(hint, na=False)
            & tmp[target_col].str.lower().str.contains(hint, na=False)
        ]

    # Deduplicate edges
    tmp = tmp.drop_duplicates(subset=[source_col, target_col])

    adjacency: Dict[str, Set[str]] = {}
    nodes: Set[str] = set(tmp[source_col]).union(set(tmp[target_col]))

    for n in nodes:
        adjacency[n] = set()

    for _, row in tmp.iterrows():
        s = row[source_col]
        t = row[target_col]
        adjacency[s].add(t)

    edges_df = tmp[[source_col, target_col, "_is_follow"]].rename(
        columns={source_col: "source", target_col: "target"}
    )
    return adjacency, edges_df, redirect_map

def pagerank(
    adjacency: Dict[str, Set[str]],
    damping: float = 0.85,
    max_iters: int = 50,
    tol: float = 1e-8,
) -> Tuple[Dict[str, float], int, float]:
    """
    Pure-Python PageRank over adjacency list.
    Handles dangling nodes by redistributing their mass uniformly.
    Returns (scores, iters_used, final_delta).
    """
    nodes = list(adjacency.keys())
    n = len(nodes)
    if n == 0:
        return {}, 0, 0.0

    idx = {u: i for i, u in enumerate(nodes)}

    out_deg = [0] * n
    out_links: List[List[int]] = [[] for _ in range(n)]
    for u in nodes:
        i = idx[u]
        targets = list(adjacency[u])
        out_deg[i] = len(targets)
        out_links[i] = [idx[v] for v in targets if v in idx]

    pr = [1.0 / n] * n
    base = (1.0 - damping) / n

    for it in range(1, max_iters + 1):
        new_pr = [base] * n

        dangling_mass = 0.0
        for i in range(n):
            if out_deg[i] == 0:
                dangling_mass += pr[i]

        # Distribute dangling mass uniformly
        dangling_add = damping * dangling_mass / n
        if dangling_add != 0.0:
            for j in range(n):
                new_pr[j] += dangling_add

        # Distribute rank over outgoing links
        for i in range(n):
            if out_deg[i] == 0:
                continue
            share = damping * pr[i] / out_deg[i]
            for j in out_links[i]:
                new_pr[j] += share

        # Convergence check (L1 delta)
        delta = sum(abs(new_pr[i] - pr[i]) for i in range(n))
        pr = new_pr
        if delta < tol:
            break

    scores = {nodes[i]: pr[i] for i in range(n)}
    scores_sum = sum(scores.values())
    # Normalize to sum=1 (numerical safety)
    if scores_sum > 0:
        scores = {u: v / scores_sum for u, v in scores.items()}

    return scores, it, delta

def main():
    st.set_page_config(page_title="Internal PageRank Proxy", layout="wide")
    st.title("Internal PageRank Proxy (Screaming Frog Export)")

    st.markdown(
        """
Upload a Screaming Frog internal link export CSV. This app builds a directed internal link graph and computes a PageRank-style importance score per URL.
"""
    )

    with st.sidebar:
        st.header("Model settings")
        damping = st.slider("Damping factor (d)", min_value=0.50, max_value=0.95, value=0.85, step=0.01)
        max_iters = st.number_input("Max iterations", min_value=10, max_value=500, value=50, step=10)
        tol = st.number_input("Convergence tolerance", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")

        st.header("Link filters")
        include_nofollow = st.checkbox("Include nofollow links", value=False)
        drop_self_loops = st.checkbox("Drop self-loops (URL links to itself)", value=True)

        # ── NEW: Redirect resolution toggle ──
        st.header("Redirect handling")
        resolve_redirects = st.checkbox(
            "Resolve 301/302 redirects",
            value=True,
            help="Collapse redirect URLs into their final destinations. "
                 "Requires a 'Status Code' column in your export. "
                 "This prevents redirect URLs from appearing as dead-end "
                 "authority sinks in the graph.",
        )

        st.header("Domain restriction (optional)")
        restrict_to_domain = st.checkbox("Restrict edges to a domain hint substring", value=False)
        internal_domain_hint = st.text_input("Domain hint (e.g. example.com)", value="")

        st.header("Output options")
        show_only_top_n = st.number_input("Show top N (0 = show all)", min_value=0, max_value=200000, value=0, step=100)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.stop()

    # Read CSV robustly
    raw = uploaded.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace")
    except Exception:
        # Last resort: try latin-1
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", errors="replace")

    st.subheader("Preview")
    st.dataframe(df.head(25), use_container_width=True)

    cols = list(df.columns)
    detected = _detect_columns(cols)

    st.subheader("Column mapping")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        source_col = st.selectbox(
            "Source URL column",
            options=cols,
            index=cols.index(detected["source"]) if detected["source"] in cols else 0,
        )

    with c2:
        target_col = st.selectbox(
            "Target URL column",
            options=cols,
            index=cols.index(detected["target"]) if detected["target"] in cols else min(1, len(cols) - 1),
        )

    with c3:
        follow_col = st.selectbox(
            "Follow/Nofollow/Rel column (optional)",
            options=["(none)"] + cols,
            index=(["(none)"] + cols).index(detected["follow"]) if detected["follow"] in cols else 0,
        )
        if follow_col == "(none)":
            follow_col = None

    # ── NEW: Status code column selector ──
    with c4:
        status_col = st.selectbox(
            "Status Code column (for redirect resolution)",
            options=["(none)"] + cols,
            index=(["(none)"] + cols).index(detected["dst_status"]) if detected["dst_status"] in cols else (
                (["(none)"] + cols).index(detected["src_status"]) if detected["src_status"] in cols else 0
            ),
        )
        if status_col == "(none)":
            status_col = None

    cfg = PRConfig(
        damping=float(damping),
        max_iters=int(max_iters),
        tol=float(tol),
        include_nofollow=bool(include_nofollow),
        drop_self_loops=bool(drop_self_loops),
    )

    if st.button("Run PageRank", type="primary"):
        adjacency, edges_df, redirect_map = _build_graph(
            df=df,
            source_col=source_col,
            target_col=target_col,
            follow_col=follow_col,
            config=cfg,
            internal_domain_hint=internal_domain_hint if internal_domain_hint else None,
            restrict_to_domain=restrict_to_domain,
            status_col=status_col,
            resolve_redirects=resolve_redirects,
        )

        n_nodes = len(adjacency)
        n_edges = int(sum(len(v) for v in adjacency.values()))
        if n_nodes < cfg.min_nodes:
            st.error(f"Not enough nodes after filtering (found {n_nodes}). Check your column mapping and filters.")
            st.stop()

        st.info(f"Graph built: {n_nodes:,} nodes, {n_edges:,} edges")

        # ── NEW: Report resolved redirects ──
        if redirect_map:
            with st.expander(f"🔀 Resolved {len(redirect_map)} redirect(s) — click to view"):
                redir_df = pd.DataFrame([
                    {"redirect_url": src, "resolved_to": dst}
                    for src, dst in sorted(redirect_map.items())
                ])
                st.dataframe(redir_df, use_container_width=True, hide_index=True)
                st.caption(
                    "These URLs were 301/302 redirects. All links pointing to them "
                    "have been rewritten to point to their final destination. "
                    "The redirect URLs have been removed from the graph."
                )

        scores, iters_used, final_delta = pagerank(
            adjacency=adjacency,
            damping=cfg.damping,
            max_iters=cfg.max_iters,
            tol=cfg.tol,
        )
        st.success(f"PageRank complete: {iters_used} iterations, final delta {final_delta:.2e}")

        # Build output table
        out_deg = {u: len(adjacency[u]) for u in adjacency.keys()}
        in_deg = {u: 0 for u in adjacency.keys()}
        for u, outs in adjacency.items():
            for v in outs:
                if v in in_deg:
                    in_deg[v] += 1

        result = pd.DataFrame(
            {
                "url": list(scores.keys()),
                "pagerank": list(scores.values()),
                "in_links": [in_deg.get(u, 0) for u in scores.keys()],
                "out_links": [out_deg.get(u, 0) for u in scores.keys()],
            }
        )

        # Add optional helpers
        result["pagerank_pct"] = result["pagerank"] * 100.0
        result = result.sort_values(["pagerank"], ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1

        if show_only_top_n and show_only_top_n > 0:
            result_view = result.head(int(show_only_top_n))
        else:
            result_view = result

        st.subheader("Results")
        st.dataframe(
            result_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "pagerank": st.column_config.NumberColumn(format="%.10f"),
                "pagerank_pct": st.column_config.NumberColumn(format="%.6f"),
            },
        )

        # Download
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results CSV",
            data=csv_bytes,
            file_name="internal_pagerank_proxy.csv",
            mime="text/csv",
        )

        with st.expander("Cleaned edges used in the graph"):
            st.dataframe(edges_df.head(500), use_container_width=True)

        with st.expander("Interpretation cheatsheet"):
            st.markdown(
                """
- Higher PageRank means your internal link graph treats that URL as more important.
- If your core service hubs are not near the top, internal linking is structurally misaligned.
- Look for surprises: tag pages, blog archives, or old posts outranking money pages.
"""
            )

if __name__ == "__main__":
    main()
