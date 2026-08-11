"""Internal PageRank / link-equity analyser for Screaming Frog exports.

Models internal link flow the way Google plausibly does:

* only real hyperlinks pass PageRank (not images, CSS, JS, hreflang or rel=next/prev)
* redirects are collapsed into their final destination
* canonicalised URLs are consolidated into the canonical target
* links to 4xx/5xx are dead ends, not normal edges
* nofollow / sponsored / ugc links are dropped by default
* links are weighted by placement (reasonable-surfer), so a footer link in a
  200-link boilerplate block is not worth the same as an in-content link

On top of the scores it answers the operational question: is authority actually
reaching the pages that matter, and if not, which pages should link to them.
"""

from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Reasonable-surfer placement weights. A surfer is far more likely to click an
# in-content link than one of 200 identical footer links, so boilerplate
# placements pass proportionally less equity.
DEFAULT_POSITION_WEIGHTS: Dict[str, float] = {
    "content": 1.00,
    "navigation": 0.50,
    "header": 0.50,
    "sidebar": 0.35,
    "footer": 0.20,
    "head": 0.00,  # <link>/<meta> in <head> is not a clickable link at all
    "unknown": 1.00,  # no data => don't penalise
}

# Link types that pass PageRank between HTML pages.
HYPERLINK_TYPES = {"hyperlink", "unknown"}

# Query parameters that never identify a distinct page.
TRACKING_PARAMS = {
    "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "yclid", "igshid",
    "mc_cid", "mc_eid", "_hsenc", "_hsmi", "vero_id", "twclid", "ttclid",
}


@dataclass
class PRConfig:
    # Algorithm
    damping: float = 0.85
    max_iters: int = 100
    tol: float = 1e-10
    min_nodes: int = 2

    # URL normalisation
    strip_fragment: bool = True
    strip_tracking_params: bool = True
    strip_all_query: bool = False
    unify_trailing_slash: bool = False

    # Graph hygiene
    hyperlinks_only: bool = True
    consolidate_canonicals: bool = True
    resolve_redirects: bool = True
    include_nofollow: bool = False
    drop_self_loops: bool = True
    drop_dead_end_edges: bool = False  # False => keep as sink (standard PageRank)

    # Reasonable surfer
    use_position_weights: bool = True
    position_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_POSITION_WEIGHTS)
    )
    empty_anchor_weight: float = 1.0


@dataclass
class Diagnostics:
    """Per-stage edge accounting so nothing disappears silently."""

    rows_uploaded: int = 0
    rows_usable: int = 0
    dropped_by_link_type: Dict[str, int] = field(default_factory=dict)
    canonical_pairs: int = 0
    canonical_rewrites: int = 0
    redirects_resolved: Dict[str, str] = field(default_factory=dict)
    redirect_cycles: List[str] = field(default_factory=list)
    unresolved_redirect_links: int = 0
    broken_target_links: int = 0
    nofollow_links: int = 0
    self_loop_links: int = 0
    external_links: int = 0
    duplicate_links: int = 0
    zero_weight_links: int = 0
    final_edges: int = 0
    dead_end_targets: Dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
#  URL normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_url(url: object, config: PRConfig) -> str:
    """Canonicalise a URL so that one page is one node.

    Google discards fragments outright and treats scheme/host case-
    insensitively, so those are always normalised. Trailing slash and query
    handling are configurable because they are genuinely site-dependent.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return url

    if not host:
        # Relative or malformed URL: leave alone rather than mangle it.
        return url

    host = host.lower()
    scheme = parts.scheme.lower() or "https"

    try:
        port = parts.port
    except ValueError:
        port = None
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"

    path = parts.path or "/"
    if config.unify_trailing_slash and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query = parts.query
    if config.strip_all_query:
        query = ""
    elif query and config.strip_tracking_params:
        kept = [
            (k, v)
            for k, v in parse_qsl(query, keep_blank_values=True)
            if not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
        ]
        query = urlencode(kept)

    fragment = "" if config.strip_fragment else parts.fragment
    return urlunsplit((scheme, netloc, path, query, fragment))


def _normalize_series(series: pd.Series, config: PRConfig) -> pd.Series:
    """Normalise a URL column, parsing each distinct URL only once.

    Exports repeat the same URLs across thousands of rows, so normalising the
    unique values and mapping back is far cheaper than a per-row apply.
    """
    lookup = {
        value: _normalize_url(value, config)
        for value in pd.unique(series)
        if isinstance(value, str)
    }
    return series.map(lookup, na_action="ignore").fillna("")


def _map_unique(series: pd.Series, fn) -> pd.Series:
    """Apply `fn` once per distinct value rather than once per row."""
    lookup = {value: fn(value) for value in pd.unique(series)}
    return series.map(lookup)


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _path_of(url: str) -> str:
    try:
        return urlsplit(url).path or "/"
    except ValueError:
        return url


def _same_site(url: str, domain: str) -> bool:
    """True when url's host is domain or a subdomain of it."""
    host = _host_of(url)
    if not host or not domain:
        return False
    domain = domain.lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


# ──────────────────────────────────────────────────────────────────────────────
#  Column detection & value parsing
# ──────────────────────────────────────────────────────────────────────────────

def _detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    """Best-effort mapping of Screaming Frog / generic crawler column names."""
    lower = {str(c).strip().lower(): c for c in cols}

    def pick(options: Iterable[str]) -> Optional[str]:
        for opt in options:
            if opt in lower:
                return lower[opt]
        return None

    return {
        # "Source"/"Destination" = SF bulk export; "From"/"To" = older exports.
        "source": pick(["source", "from", "source url", "from url", "origin", "address", "url"]),
        # NB: SF's bare "Target" column is the anchor target attribute
        # (_blank), not a URL, so it is deliberately the last candidate.
        "target": pick([
            "destination", "to", "destination url", "to url", "destination address",
            "target url", "linked url", "link", "target",
        ]),
        "follow": pick(["follow", "nofollow", "rel", "rel attribute", "link attribute"]),
        # In SF's All Inlinks export "Status Code" is the DESTINATION's status.
        "status": pick([
            "status code", "destination status code", "to status code",
            "target status code", "dest status code",
        ]),
        "link_type": pick(["type", "link type"]),
        "position": pick(["link position", "position"]),
        "anchor": pick(["anchor", "anchor text", "link anchor", "text"]),
        "alt": pick(["alt text", "alt", "image alt text"]),
    }


def _is_follow_value(val: object) -> Optional[bool]:
    """Interpret a Follow / rel column. None means "unknown, assume followed"."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in ("true", "follow", "followed", "1", "yes", "y"):
        return True
    if s in ("false", "nofollow", "no-follow", "not followed", "0", "no", "n"):
        return False
    # rel can hold several tokens, e.g. "noopener nofollow sponsored".
    # Google treats sponsored and ugc as link hints too, so none of them pass.
    tokens = set(s.replace(",", " ").split())
    if tokens & {"nofollow", "sponsored", "ugc"}:
        return False
    return None


def _classify_link_type(val: object) -> str:
    """Bucket a crawler link Type value.

    Only 'hyperlink' (and 'unknown', when the export has no Type column) moves
    PageRank between pages. Canonical and redirect rows are captured separately
    because they tell us how to consolidate nodes.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "unknown"
    s = str(val).strip().lower()
    if s in ("", "nan"):
        return "unknown"
    if "redirect" in s or "refresh" in s:
        return "redirect"
    if "canonical" in s:
        return "canonical"
    if "hreflang" in s:
        return "hreflang"
    if "next" in s or "prev" in s:
        return "pagination_rel"
    if "amphtml" in s or s == "amp":
        return "amphtml"
    if "hyperlink" in s or s in ("html", "a", "ahref", "anchor"):
        return "hyperlink"
    if "image" in s or s == "img":
        return "image"
    if "javascript" in s or s == "js":
        return "javascript"
    if "css" in s or "stylesheet" in s:
        return "css"
    return "other"


def _classify_position(val: object) -> str:
    """Bucket a Screaming Frog Link Position value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "unknown"
    s = str(val).strip().lower()
    if s in ("", "nan"):
        return "unknown"
    if "foot" in s:
        return "footer"
    if "nav" in s or "menu" in s or "breadcrumb" in s:
        return "navigation"
    if "aside" in s or "side" in s:
        return "sidebar"
    if s == "head":
        return "head"
    if "header" in s or "masthead" in s:
        return "header"
    if "content" in s or "main" in s or "body" in s or "article" in s:
        return "content"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
#  Node consolidation: canonicals and redirects
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_chains(
    mapping: Dict[str, str], max_hops: int = 10
) -> Tuple[Dict[str, str], List[str]]:
    """Flatten A→B→C into A→C. Returns (flattened, urls_in_cycles)."""
    resolved: Dict[str, str] = {}
    cycles: List[str] = []
    for start in mapping:
        seen = {start}
        dest = mapping[start]
        hops = 0
        while dest in mapping and hops < max_hops:
            if dest in seen:
                cycles.append(start)
                break
            seen.add(dest)
            dest = mapping[dest]
            hops += 1
        if dest != start:
            resolved[start] = dest
        else:
            cycles.append(start)
    return resolved, sorted(set(cycles))


def _build_redirect_map(
    edges: pd.DataFrame, status_by_url: Dict[str, float]
) -> Dict[str, str]:
    """Map each redirecting URL to its final destination.

    Two evidence sources, strongest first:

    1. Rows the crawler typed as a redirect / meta refresh. Source is the
       redirecting URL, destination is where it points. This is authoritative.
    2. A URL that carries a 3xx status and has exactly one recorded outlink —
       that outlink must be the redirect target.

    Ambiguous cases (3xx URL with several recorded outlinks and no typed
    redirect row) are deliberately left unresolved rather than guessed at.
    """
    mapping: Dict[str, str] = {}

    typed = edges[edges["_link_type"] == "redirect"]
    for src, tgt in zip(typed["_source"], typed["_target"]):
        if src and tgt and src != tgt:
            mapping[src] = tgt

    redirecting = {
        url for url, code in status_by_url.items() if 300 <= code < 400
    } - set(mapping)
    if redirecting:
        candidates = edges[edges["_source"].isin(redirecting)]
        grouped = candidates.groupby("_source")["_target"].unique()
        for src, targets in grouped.items():
            targets = [t for t in targets if t != src]
            if len(targets) == 1:
                mapping[src] = targets[0]

    return mapping


def _build_canonical_map(edges: pd.DataFrame) -> Dict[str, str]:
    """Map non-canonical URLs to their canonical from typed canonical rows."""
    canonical_rows = edges[edges["_link_type"] == "canonical"]
    mapping: Dict[str, str] = {}
    for src, tgt in zip(canonical_rows["_source"], canonical_rows["_target"]):
        if src and tgt and src != tgt:
            mapping[src] = tgt
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
#  Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_edges(
    df: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
    config: PRConfig,
    internal_domain: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Diagnostics, Dict[str, str]]:
    """Turn a raw crawler export into a weighted, deduplicated edge list.

    Returns (edges, status_by_url, diagnostics, rewrite_map).
    """
    diag = Diagnostics(rows_uploaded=len(df))
    source_col = colmap["source"]
    target_col = colmap["target"]

    work = pd.DataFrame({
        "_source": _normalize_series(df[source_col], config),
        "_target": _normalize_series(df[target_col], config),
    })
    work["_link_type"] = (
        df[colmap["link_type"]].map(_classify_link_type)
        if colmap.get("link_type") else "unknown"
    )
    work["_follow"] = (
        df[colmap["follow"]].map(_is_follow_value)
        if colmap.get("follow") else None
    )
    work["_position"] = (
        df[colmap["position"]].map(_classify_position)
        if colmap.get("position") else "unknown"
    )

    anchor = df[colmap["anchor"]] if colmap.get("anchor") else None
    alt = df[colmap["alt"]] if colmap.get("alt") else None
    if anchor is not None or alt is not None:
        def _blank(series: Optional[pd.Series]) -> pd.Series:
            if series is None:
                return pd.Series(True, index=work.index)
            return series.fillna("").astype(str).str.strip().eq("")
        work["_no_anchor"] = _blank(anchor) & _blank(alt)
    else:
        work["_no_anchor"] = False

    work = work[(work["_source"] != "") & (work["_target"] != "")]
    diag.rows_usable = len(work)

    # Destination status codes, keyed by URL, before any rewriting.
    status_by_url: Dict[str, float] = {}
    if colmap.get("status"):
        codes = pd.to_numeric(df[colmap["status"]], errors="coerce")
        status_frame = pd.DataFrame(
            {"url": work["_target"], "code": codes.reindex(work.index)}
        ).dropna()
        if not status_frame.empty:
            status_by_url = (
                status_frame.groupby("url")["code"].first().to_dict()
            )

    # ── Consolidate nodes: canonicals then redirects ──
    rewrite: Dict[str, str] = {}
    if config.consolidate_canonicals:
        canonical_map = _build_canonical_map(work)
        diag.canonical_pairs = len(canonical_map)
        rewrite.update(canonical_map)

    if config.resolve_redirects:
        redirect_map = _build_redirect_map(work, status_by_url)
        diag.redirects_resolved = redirect_map
        rewrite.update(redirect_map)  # a redirect beats a canonical claim

    if rewrite:
        rewrite, diag.redirect_cycles = _resolve_chains(rewrite)
        before = work["_target"]
        work["_source"] = work["_source"].map(rewrite).fillna(work["_source"])
        work["_target"] = before.map(rewrite).fillna(before)
        diag.canonical_rewrites = int((before != work["_target"]).sum())

    # ── Keep only link types that move PageRank ──
    if config.hyperlinks_only and colmap.get("link_type"):
        keep = work["_link_type"].isin(HYPERLINK_TYPES)
        diag.dropped_by_link_type = (
            work.loc[~keep, "_link_type"].value_counts().to_dict()
        )
        work = work[keep]

    # ── Dead ends: 4xx/5xx and redirects we could not resolve ──
    if status_by_url:
        codes = work["_target"].map(status_by_url)
        broken = codes.between(400, 599, inclusive="both").fillna(False)
        unresolved = codes.between(300, 399, inclusive="both").fillna(False)
        diag.broken_target_links = int(broken.sum())
        diag.unresolved_redirect_links = int(unresolved.sum())
        dead = broken | unresolved
        if dead.any():
            diag.dead_end_targets = (
                work.loc[dead, "_target"].value_counts().to_dict()
            )
            if config.drop_dead_end_edges:
                work = work[~dead]

    # ── rel=nofollow / sponsored / ugc ──
    if not config.include_nofollow and colmap.get("follow"):
        blocked = work["_follow"].eq(False)
        diag.nofollow_links = int(blocked.sum())
        work = work[~blocked]

    if config.drop_self_loops:
        loops = work["_source"] == work["_target"]
        diag.self_loop_links = int(loops.sum())
        work = work[~loops]

    if internal_domain:
        is_internal = lambda u: _same_site(u, internal_domain)  # noqa: E731
        internal = _map_unique(work["_source"], is_internal) & _map_unique(
            work["_target"], is_internal
        )
        diag.external_links = int((~internal).sum())
        work = work[internal]

    # ── Reasonable-surfer edge weights ──
    if config.use_position_weights:
        weights = work["_position"].map(
            lambda p: config.position_weights.get(p, 1.0)
        ).astype(float)
    else:
        weights = pd.Series(1.0, index=work.index)
    if config.empty_anchor_weight != 1.0:
        weights = weights.where(~work["_no_anchor"], weights * config.empty_anchor_weight)
    work["_weight"] = weights

    zero = work["_weight"] <= 0
    diag.zero_weight_links = int(zero.sum())
    work = work[~zero]

    # Google consolidates repeated links from one page to the same target into
    # a single vote, so keep one edge per pair — the best-placed one.
    before_dedupe = len(work)
    work = (
        work.sort_values("_weight", ascending=False)
        .drop_duplicates(subset=["_source", "_target"], keep="first")
    )
    diag.duplicate_links = before_dedupe - len(work)
    diag.final_edges = len(work)

    edges = work[["_source", "_target", "_weight", "_position", "_link_type"]].rename(
        columns={
            "_source": "source",
            "_target": "target",
            "_weight": "weight",
            "_position": "position",
            "_link_type": "link_type",
        }
    ).reset_index(drop=True)

    return edges, status_by_url, diag, rewrite


def index_graph(edges: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Index nodes and return (nodes, src_idx, dst_idx, weights)."""
    nodes = pd.unique(pd.concat([edges["source"], edges["target"]], ignore_index=True))
    node_list = list(nodes)
    idx = {u: i for i, u in enumerate(node_list)}
    src = edges["source"].map(idx).to_numpy(dtype=np.int64)
    dst = edges["target"].map(idx).to_numpy(dtype=np.int64)
    weights = edges["weight"].to_numpy(dtype=np.float64)
    return node_list, src, dst, weights


# ──────────────────────────────────────────────────────────────────────────────
#  PageRank
# ──────────────────────────────────────────────────────────────────────────────

def pagerank(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    weights: np.ndarray,
    damping: float = 0.85,
    max_iters: int = 100,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, int, float]:
    """Weighted PageRank by power iteration.

    Each page splits its damped rank across its outbound links in proportion to
    edge weight rather than uniformly, which is what makes placement matter.
    Dangling nodes (no outbound links) have their mass redistributed uniformly,
    the standard treatment. Returns (scores, iterations_used, final_delta).
    """
    if n_nodes == 0:
        return np.zeros(0), 0, 0.0

    out_weight = np.bincount(src_idx, weights=weights, minlength=n_nodes)
    dangling = out_weight <= 0
    teleport = 1.0 / n_nodes

    # Fraction of the source's rank travelling down each edge.
    edge_share = weights / out_weight[src_idx] if len(src_idx) else weights

    pr = np.full(n_nodes, teleport)
    iters_used = 0
    delta = 0.0

    for iters_used in range(1, max_iters + 1):
        inflow = np.bincount(
            dst_idx, weights=pr[src_idx] * edge_share, minlength=n_nodes
        )
        dangling_mass = float(pr[dangling].sum())
        new_pr = (1.0 - damping) * teleport + damping * (
            inflow + dangling_mass * teleport
        )
        delta = float(np.abs(new_pr - pr).sum())
        pr = new_pr
        if delta < tol:
            break

    total = pr.sum()
    if total > 0:
        pr = pr / total
    return pr, iters_used, delta


def click_depth(
    n_nodes: int, src_idx: np.ndarray, dst_idx: np.ndarray, start: int
) -> np.ndarray:
    """BFS hop count from `start`. Unreachable nodes get -1."""
    depth = np.full(n_nodes, -1, dtype=np.int64)
    if not (0 <= start < n_nodes):
        return depth

    order = np.argsort(src_idx, kind="stable")
    sorted_src = src_idx[order]
    sorted_dst = dst_idx[order]
    row_start = np.searchsorted(sorted_src, np.arange(n_nodes), side="left")
    row_end = np.searchsorted(sorted_src, np.arange(n_nodes), side="right")

    depth[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in sorted_dst[row_start[u]:row_end[u]]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(int(v))
    return depth


# ──────────────────────────────────────────────────────────────────────────────
#  Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def guess_homepage(nodes: List[str], scores: np.ndarray, domain: Optional[str]) -> str:
    """Pick the most likely homepage: root path on the primary host, else top score."""
    roots = [
        u for u in nodes
        if _path_of(u) in ("", "/") and (not domain or _same_site(u, domain))
    ]
    if roots:
        by_score = {nodes[i]: scores[i] for i in range(len(nodes))}
        return max(roots, key=lambda u: by_score.get(u, 0.0))
    return nodes[int(np.argmax(scores))] if len(nodes) else ""


def match_priority_urls(
    raw_lines: str, nodes: List[str], config: PRConfig
) -> Tuple[Dict[str, str], List[str]]:
    """Match pasted priority URLs to graph nodes.

    Exact normalised match first, then a path-only match so a bare "/kitchens"
    still resolves. Returns ({input: node}, unmatched_inputs).
    """
    node_set = set(nodes)
    by_path: Dict[str, List[str]] = {}
    for u in nodes:
        by_path.setdefault(_path_of(u).rstrip("/") or "/", []).append(u)

    matched: Dict[str, str] = {}
    unmatched: List[str] = []
    for line in raw_lines.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        normalised = _normalize_url(candidate, config)
        if normalised in node_set:
            matched[candidate] = normalised
            continue
        path_key = _path_of(candidate if "://" in candidate else "/" + candidate.lstrip("/"))
        path_key = path_key.rstrip("/") or "/"
        hits = by_path.get(path_key, [])
        if len(hits) == 1:
            matched[candidate] = hits[0]
        else:
            unmatched.append(candidate)
    return matched, unmatched


def donor_suggestions(
    target_node: str,
    result: pd.DataFrame,
    edges: pd.DataFrame,
    limit: int = 10,
    donor_pool: int = 300,
) -> pd.DataFrame:
    """Highest-authority pages that do not yet link to `target_node`.

    Ranked by the rank a link would actually carry: the donor's PageRank
    divided across its existing outbound links, so a strong page that already
    links to 300 places is ranked below a strong page that links to 20.
    """
    existing = set(edges.loc[edges["target"] == target_node, "source"])
    pool = result.head(donor_pool)
    pool = pool[
        (pool["url"] != target_node)
        & (~pool["url"].isin(existing))
        & (pool["out_links"] > 0)
        & (pool["is_dead_end"] == False)  # noqa: E712 - pandas mask
    ].copy()
    if pool.empty:
        return pool
    pool["equity_per_link"] = pool["pagerank"] / (pool["out_links"] + 1)
    return (
        pool.sort_values("equity_per_link", ascending=False)
        .head(limit)[["url", "pagerank", "rank", "out_links", "equity_per_link"]]
        .reset_index(drop=True)
    )


def run_analysis(
    df: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
    config: PRConfig,
    internal_domain: Optional[str],
    homepage_override: str,
) -> Optional[dict]:
    edges, status_by_url, diag, rewrite = build_edges(
        df, colmap, config, internal_domain
    )
    if edges.empty:
        return None

    nodes, src_idx, dst_idx, weights = index_graph(edges)
    n_nodes = len(nodes)
    if n_nodes < config.min_nodes:
        return None

    scores, iters_used, final_delta = pagerank(
        n_nodes, src_idx, dst_idx, weights,
        damping=config.damping, max_iters=config.max_iters, tol=config.tol,
    )

    homepage = homepage_override.strip()
    homepage = (
        _normalize_url(homepage, config) if homepage
        else guess_homepage(nodes, scores, internal_domain)
    )
    idx = {u: i for i, u in enumerate(nodes)}
    depths = click_depth(n_nodes, src_idx, dst_idx, idx.get(homepage, -1))

    out_links = np.bincount(src_idx, minlength=n_nodes)
    in_links = np.bincount(dst_idx, minlength=n_nodes)
    out_weight = np.bincount(src_idx, weights=weights, minlength=n_nodes)

    result = pd.DataFrame({
        "url": nodes,
        "pagerank": scores,
        "pagerank_pct": scores * 100.0,
        "in_links": in_links,
        "out_links": out_links,
        "out_link_weight": out_weight,
        "click_depth": depths,
        "status": [status_by_url.get(u, np.nan) for u in nodes],
    })
    result["is_dead_end"] = result["status"].between(300, 599, inclusive="both").fillna(False)
    result = result.sort_values("pagerank", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    result["pagerank_percentile"] = result["pagerank"].rank(pct=True) * 100.0

    return {
        "edges": edges,
        "result": result,
        "diagnostics": diag,
        "rewrite": rewrite,
        "homepage": homepage,
        "iters_used": iters_used,
        "final_delta": final_delta,
        "converged": final_delta < config.tol,
        "n_nodes": n_nodes,
        "config": config,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar_controls() -> Tuple[PRConfig, dict]:
    with st.sidebar:
        st.header("Algorithm")
        damping = st.slider("Damping factor (d)", 0.50, 0.95, 0.85, 0.01)
        max_iters = st.number_input("Max iterations", 10, 1000, 100, 10)
        tol = st.number_input("Convergence tolerance", 1e-14, 1e-2, 1e-10, format="%.1e")

        st.header("Link graph")
        hyperlinks_only = st.checkbox(
            "Count hyperlinks only", value=True,
            help="Excludes image, CSS, JavaScript, hreflang and rel=next/prev rows. "
                 "Screaming Frog's All Inlinks export contains all of them and none "
                 "pass PageRank between pages.",
        )
        include_nofollow = st.checkbox(
            "Include nofollow / sponsored / ugc links", value=False,
            help="Google treats all three as hints that no ranking credit passes.",
        )
        drop_self_loops = st.checkbox("Drop self-loops", value=True)
        consolidate_canonicals = st.checkbox(
            "Consolidate canonicalised URLs", value=True,
            help="Uses the export's canonical rows to merge a URL into its canonical, "
                 "mirroring how Google assigns equity to the canonical version.",
        )
        resolve_redirects = st.checkbox(
            "Resolve 301/302 redirects", value=True,
            help="Rewrites links pointing at a redirect so they point at the final "
                 "destination. Needs a Status Code column or typed redirect rows.",
        )
        drop_dead_end_edges = st.checkbox(
            "Remove links to 4xx/5xx and unresolved redirects", value=False,
            help="Off (default): broken targets stay in the graph as dead ends, so you "
                 "can see how much authority is aimed at them. On: those links are "
                 "deleted, showing the graph you would have after fixing them.",
        )

        st.header("Reasonable surfer")
        use_position_weights = st.checkbox(
            "Weight links by placement", value=True,
            help="Google's reasonable-surfer model weights a link by how likely it is "
                 "to be clicked. Needs a Link Position column; without one every link "
                 "is weighted equally.",
        )
        weights = dict(DEFAULT_POSITION_WEIGHTS)
        if use_position_weights:
            with st.expander("Placement weights"):
                for key in ("content", "navigation", "header", "sidebar", "footer", "head"):
                    weights[key] = st.slider(
                        key.title(), 0.0, 1.0, DEFAULT_POSITION_WEIGHTS[key], 0.05
                    )
            empty_anchor_weight = st.slider(
                "Weight for links with no anchor or alt text", 0.0, 1.0, 1.0, 0.05,
                help="Anchorless wrapper links are less clickable and carry no "
                     "relevance signal. Leave at 1.0 to ignore this factor.",
            )
        else:
            empty_anchor_weight = 1.0

        st.header("URL normalisation")
        strip_fragment = st.checkbox(
            "Strip #fragments", value=True,
            help="Google always does. Leaving these in splits one page into many nodes.",
        )
        strip_tracking_params = st.checkbox("Strip utm_* and ad click IDs", value=True)
        unify_trailing_slash = st.checkbox(
            "Treat /page and /page/ as the same URL", value=False,
            help="Enable only if your site truly serves both as one page.",
        )
        strip_all_query = st.checkbox("Strip all query strings", value=False)

        st.header("Scope")
        internal_domain = st.text_input(
            "Internal domain (blank = auto-detect)", value="",
            help="Host-aware: matches the domain and its subdomains, not a substring.",
        )
        homepage_override = st.text_input("Homepage URL (blank = auto-detect)", value="")

        st.header("Your important pages")
        priority_raw = st.text_area(
            "One URL or path per line", height=140,
            placeholder="https://example.com/kitchen-remodeling\n/bathroom-remodeling",
            help="The money pages that should be receiving authority.",
        )

    config = PRConfig(
        damping=float(damping),
        max_iters=int(max_iters),
        tol=float(tol),
        hyperlinks_only=bool(hyperlinks_only),
        include_nofollow=bool(include_nofollow),
        drop_self_loops=bool(drop_self_loops),
        consolidate_canonicals=bool(consolidate_canonicals),
        resolve_redirects=bool(resolve_redirects),
        drop_dead_end_edges=bool(drop_dead_end_edges),
        use_position_weights=bool(use_position_weights),
        position_weights=weights,
        empty_anchor_weight=float(empty_anchor_weight),
        strip_fragment=bool(strip_fragment),
        strip_tracking_params=bool(strip_tracking_params),
        unify_trailing_slash=bool(unify_trailing_slash),
        strip_all_query=bool(strip_all_query),
    )
    ui = {
        "internal_domain": internal_domain.strip() or None,
        "homepage_override": homepage_override,
        "priority_raw": priority_raw,
    }
    return config, ui


def _read_csv(raw: bytes) -> pd.DataFrame:
    for kwargs in (
        {},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
        {"encoding": "latin-1", "engine": "python", "on_bad_lines": "skip"},
    ):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("Could not parse the uploaded file as CSV.")


def _auto_detect_domain(df: pd.DataFrame, source_col: str) -> Optional[str]:
    """Most common host among source URLs — the site being crawled."""
    hosts = df[source_col].dropna().head(20000).map(_host_of)
    hosts = hosts[hosts != ""]
    if hosts.empty:
        return None
    return hosts.value_counts().idxmax()


def _render_overview(data: dict, priority: pd.DataFrame) -> None:
    result = data["result"]
    diag = data["diagnostics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs in graph", f"{data['n_nodes']:,}")
    c2.metric("Links counted", f"{diag.final_edges:,}")

    orphans = result[(result["in_links"] == 0) & (result["url"] != data["homepage"])]
    c3.metric("Orphaned URLs", f"{len(orphans):,}")

    wasted = diag.broken_target_links + diag.unresolved_redirect_links + diag.nofollow_links
    total_links = max(diag.final_edges + wasted, 1)
    c4.metric("Links passing no equity", f"{wasted / total_links * 100:.1f}%")

    if not data["converged"]:
        st.warning(
            f"PageRank did not converge in {data['iters_used']} iterations "
            f"(final delta {data['final_delta']:.2e}). Raise max iterations."
        )

    if priority.empty:
        st.info(
            "Add your money pages in the sidebar under **Your important pages** to see "
            "whether internal linking is actually reaching them."
        )
        return

    below = priority[priority["pagerank_percentile"] < 90]
    # click_depth of -1 means the homepage cannot reach the page at all, which is
    # worse than being deep — so it has to count here, not slip past a "> 3" test.
    deep = priority[(priority["click_depth"] > 3) | (priority["click_depth"] < 0)]
    st.subheader("Are your important pages getting authority?")
    a, b, c = st.columns(3)
    a.metric("Priority pages tracked", len(priority))
    b.metric("Outside the top 10% of PageRank", len(below))
    c.metric("Buried deeper than 3 clicks", len(deep))

    display = priority.copy()
    display["click_depth"] = display["click_depth"].where(
        display["click_depth"] >= 0, pd.NA
    )
    st.dataframe(
        display[[
            "url", "rank", "pagerank_percentile", "pagerank_pct",
            "in_links", "out_links", "click_depth", "status",
        ]],
        width="stretch", hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("PR rank"),
            "pagerank_percentile": st.column_config.NumberColumn("Percentile", format="%.1f"),
            "pagerank_pct": st.column_config.NumberColumn("PageRank %", format="%.4f"),
            "click_depth": st.column_config.NumberColumn("Clicks from home"),
        },
    )
    st.caption(
        "An empty *Clicks from home* cell means the page cannot be reached from the "
        "homepage by following internal links at all."
    )
    if len(deep):
        st.warning(
            f"{len(deep)} priority page(s) are more than 3 clicks from the homepage or "
            "unreachable from it. Click depth is a ranking signal in its own right, "
            "independent of PageRank."
        )
    if len(below):
        st.warning(
            f"{len(below)} priority page(s) sit outside the top 10% of internal "
            "PageRank. See the **Link opportunities** tab for the pages best placed "
            "to fix that."
        )


def _render_opportunities(data: dict, priority: pd.DataFrame) -> None:
    if priority.empty:
        st.info("Add priority pages in the sidebar to get link recommendations.")
        return

    st.markdown(
        "For each priority page, the highest-authority pages that **do not currently "
        "link to it**. Ranked by how much equity a new link would actually carry — a "
        "strong page with 20 outbound links passes far more per link than an equally "
        "strong page with 300."
    )
    for _, row in priority.iterrows():
        depth = int(row["click_depth"])
        depth_label = f"{depth} clicks from home" if depth >= 0 else "unreachable from home"
        with st.expander(
            f"{row['url']}  —  rank #{int(row['rank'])}, "
            f"{int(row['in_links'])} internal inlinks, {depth_label}"
        ):
            donors = donor_suggestions(row["url"], data["result"], data["edges"])
            if donors.empty:
                st.write("Every high-authority page already links here.")
            else:
                st.dataframe(
                    donors, width="stretch", hide_index=True,
                    column_config={
                        "pagerank": st.column_config.NumberColumn(format="%.8f"),
                        "equity_per_link": st.column_config.NumberColumn(
                            "Equity per new link", format="%.8f"
                        ),
                    },
                )


def _render_waste(data: dict) -> None:
    diag = data["diagnostics"]
    result = data["result"]

    st.markdown("### Where internal link equity is leaking")
    rows = [
        ("Links to 4xx / 5xx URLs", diag.broken_target_links,
         "Equity aimed at these pages goes nowhere. Repoint or remove the links."),
        ("Links to unresolved redirects", diag.unresolved_redirect_links,
         "Update the links to point straight at the final destination."),
        ("nofollow / sponsored / ugc links", diag.nofollow_links,
         "Internal nofollow is almost never useful — it discards the equity."),
        ("Redirects collapsed", len(diag.redirects_resolved),
         "Handled automatically here, but each one is a link worth updating in the CMS."),
        ("Canonicalised URLs merged", diag.canonical_pairs,
         "Links pointing at non-canonical URLs. Point them at the canonical instead."),
        ("Self-referencing links dropped", diag.self_loop_links, "Harmless, no action needed."),
        ("Repeat links merged", diag.duplicate_links,
         "One page linking to the same target more than once is a single vote. Includes "
         "links that became duplicates once redirects and canonicals were consolidated."),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Issue", "Links", "What to do"]),
        width="stretch", hide_index=True,
    )

    if diag.dropped_by_link_type:
        st.markdown("#### Non-hyperlink rows excluded from the graph")
        st.dataframe(
            pd.DataFrame(
                sorted(diag.dropped_by_link_type.items(), key=lambda kv: -kv[1]),
                columns=["Link type", "Rows"],
            ),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Images, CSS, JavaScript, hreflang and rel=next/prev references do not "
            "pass PageRank. Counting them inflates out-degree and dilutes real links."
        )

    if diag.dead_end_targets:
        st.markdown("#### Broken and redirecting URLs receiving the most links")
        st.dataframe(
            pd.DataFrame(
                sorted(diag.dead_end_targets.items(), key=lambda kv: -kv[1])[:200],
                columns=["URL", "Internal links pointing here"],
            ),
            width="stretch", hide_index=True,
        )

    orphans = result[(result["in_links"] == 0) & (result["url"] != data["homepage"])]
    if not orphans.empty:
        st.markdown("#### Orphaned URLs (no internal links in)")
        st.dataframe(
            orphans[["url", "pagerank_pct", "out_links", "status"]].head(500),
            width="stretch", hide_index=True,
        )

    st.markdown("#### Biggest dilution points")
    st.caption(
        "High-authority pages spreading their equity across the most links. Trimming "
        "boilerplate here concentrates equity on the pages you care about."
    )
    dilution = data["result"].head(500).copy()
    dilution["equity_per_link"] = dilution["pagerank"] / dilution["out_links"].clip(lower=1)
    st.dataframe(
        dilution.sort_values("out_links", ascending=False)
        .head(25)[["url", "rank", "pagerank_pct", "out_links", "equity_per_link"]],
        width="stretch", hide_index=True,
        column_config={
            "equity_per_link": st.column_config.NumberColumn(format="%.8f"),
            "pagerank_pct": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    if diag.redirect_cycles:
        st.markdown("#### Redirect / canonical loops")
        st.dataframe(
            pd.DataFrame({"URL": diag.redirect_cycles}),
            width="stretch", hide_index=True,
        )


def _render_results(data: dict, top_n: int) -> None:
    result = data["result"]
    view = result.head(top_n) if top_n else result
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "pagerank": st.column_config.NumberColumn(format="%.10f"),
            "pagerank_pct": st.column_config.NumberColumn(format="%.6f"),
            "pagerank_percentile": st.column_config.NumberColumn(format="%.1f"),
            "out_link_weight": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download full results CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="internal_pagerank.csv",
        mime="text/csv",
    )
    with st.expander("Weighted edges used in the graph"):
        st.dataframe(data["edges"].head(1000), width="stretch", hide_index=True)
        st.download_button(
            "Download full edge list CSV",
            data=data["edges"].to_csv(index=False).encode("utf-8"),
            file_name="internal_link_edges.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="Internal PageRank", layout="wide")
    st.title("Internal PageRank & Link Equity Audit")
    st.markdown(
        "Upload a Screaming Frog **All Inlinks** export (`Bulk Export → Links → "
        "All Inlinks`). Include the **Type**, **Status Code**, **Follow** and "
        "**Link Position** columns — each one materially improves accuracy."
    )

    config, ui = _sidebar_controls()
    top_n = st.sidebar.number_input(
        "Show top N in results table (0 = all)", 0, 200000, 500, 100
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.stop()

    try:
        df = _read_csv(uploaded.read())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(25), width="stretch")

    cols = list(df.columns)
    detected = _detect_columns(cols)

    st.subheader("Column mapping")

    def selector(label: str, key: str, optional: bool, default_index: int = 0):
        options = (["(none)"] + cols) if optional else cols
        guess = detected.get(key)
        index = options.index(guess) if guess in options else default_index
        chosen = st.selectbox(label, options=options, index=index)
        return None if chosen == "(none)" else chosen

    c1, c2, c3 = st.columns(3)
    with c1:
        source_col = selector("Source URL *", "source", optional=False)
        follow_col = selector("Follow / rel", "follow", optional=True)
        anchor_col = selector("Anchor text", "anchor", optional=True)
    with c2:
        target_col = selector(
            "Target URL *", "target", optional=False,
            default_index=min(1, len(cols) - 1),
        )
        status_col = selector("Destination status code", "status", optional=True)
        alt_col = selector("Image alt text", "alt", optional=True)
    with c3:
        link_type_col = selector("Link type", "link_type", optional=True)
        position_col = selector("Link position", "position", optional=True)

    colmap = {
        "source": source_col, "target": target_col, "follow": follow_col,
        "status": status_col, "link_type": link_type_col, "position": position_col,
        "anchor": anchor_col, "alt": alt_col,
    }

    missing = [
        name for name, col in (
            ("Type", link_type_col), ("Status Code", status_col),
            ("Follow", follow_col), ("Link Position", position_col),
        ) if col is None
    ]
    if missing:
        st.warning(
            "Not mapped: " + ", ".join(missing) + ". Without **Type** every image, "
            "CSS and script reference is counted as a link; without **Status Code** "
            "redirects and 404s cannot be handled; without **Link Position** every "
            "link is weighted equally. Re-export with these columns for accurate "
            "results."
        )

    domain = ui["internal_domain"] or _auto_detect_domain(df, source_col)
    if domain:
        st.caption(f"Treating **{domain}** (and its subdomains) as internal.")

    if st.button("Run analysis", type="primary"):
        with st.spinner("Building link graph and computing PageRank…"):
            st.session_state["pr_data"] = run_analysis(
                df, colmap, config, domain, ui["homepage_override"]
            )
        st.session_state["pr_priority_raw"] = ui["priority_raw"]

    data = st.session_state.get("pr_data")
    if data is None:
        if "pr_data" in st.session_state:
            st.error(
                "No usable links after filtering. Check the source/target column "
                "mapping and loosen the filters."
            )
        st.stop()

    st.success(
        f"{data['n_nodes']:,} URLs · {data['diagnostics'].final_edges:,} links · "
        f"converged in {data['iters_used']} iterations "
        f"(delta {data['final_delta']:.2e}) · homepage: {data['homepage'] or 'not found'}"
    )

    matched, unmatched = match_priority_urls(
        st.session_state.get("pr_priority_raw", ""), list(data["result"]["url"]), config
    )
    priority = data["result"][data["result"]["url"].isin(set(matched.values()))].copy()
    if unmatched:
        st.warning(
            "These priority URLs are not in the link graph at all — they may be "
            "orphaned, blocked from crawling, or typed differently: "
            + ", ".join(unmatched[:20])
        )

    tabs = st.tabs(["Overview", "Link opportunities", "Wasted equity", "All URLs", "Method"])
    with tabs[0]:
        _render_overview(data, priority)
    with tabs[1]:
        _render_opportunities(data, priority)
    with tabs[2]:
        _render_waste(data)
    with tabs[3]:
        _render_results(data, int(top_n))
    with tabs[4]:
        st.markdown(METHOD_NOTES)


METHOD_NOTES = """
### What this models

PageRank over your internal link graph, with the adjustments that make it behave
like Google rather than like a textbook exercise.

**Only real links count.** A Screaming Frog All Inlinks export mixes hyperlinks
with image, CSS, JavaScript, hreflang, canonical and rel=next/prev rows. Only
hyperlinks move PageRank; counting the rest inflates every page's out-degree and
dilutes the links that matter.

**One page, one node.** Fragments are stripped (Google discards them), scheme and
host are lowercased, default ports and tracking parameters are removed, and
canonicalised URLs are merged into their canonical. Otherwise `/kitchens`,
`/kitchens#gallery` and `/kitchens?utm_source=fb` split one page's authority
three ways.

**Redirects are collapsed.** A link to a 301 is credited to the final
destination, using typed redirect rows where the export has them and 3xx status
codes plus a single recorded outlink otherwise. Genuinely ambiguous chains are
left unresolved and reported rather than guessed at.

**Dead ends stay visible.** Links to 4xx/5xx and unresolved redirects are kept in
the graph by default so you can see how much authority is aimed at nothing.
Enable *Remove links to 4xx/5xx* to model the graph you would have after fixing
them, then compare.

**Links are weighted by placement (reasonable surfer).** Google's reasonable-
surfer patent weights a link by how likely it is to be clicked. A footer link
repeated across 5,000 pages is not worth an in-content editorial link, so
placement weights scale each edge. Repeat links from one page to the same target
count once, as Google consolidates them.

**Repeat links count once.** Multiple links from the same page to the same URL are
a single vote.

### Known limits

- PageRank is one ranking input. High internal PageRank does not guarantee
  rankings, and a low score does not mean a page cannot rank.
- Placement weights are a reasonable approximation, not Google's real numbers.
  They are exposed as sliders so you can sanity-check how sensitive your
  conclusions are.
- Dangling pages (no outbound links) have their equity redistributed uniformly.
  That is the standard treatment, but it means "wasted" equity is recycled rather
  than destroyed — read the **Wasted equity** tab for what you would reclaim.
- External links are excluded. Real sites leak equity off-site, which lowers
  every internal score slightly but changes the ordering very little.
- `noindex` is not visible in an inlinks export. A noindexed page still passes
  PageRank through its links, so its presence here is correct, but it cannot
  rank itself.

### Reading the output

- **Priority pages outside the top 10%** is the headline. Your money pages should
  be near the top of internal PageRank; if tag archives and pagination outrank
  them, the link structure is working against you.
- **Clicks from home above 3** is a problem independent of PageRank. Depth is a
  strong signal on its own.
- **Link opportunities** ranks donors by equity per new link, so it favours
  strong pages that are not already spraying links everywhere.
- **Biggest dilution points** shows where trimming boilerplate concentrates
  equity on the pages you care about.
"""


if __name__ == "__main__":
    main()
