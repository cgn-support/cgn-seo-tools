# CLAUDE.md - AI Assistant Guidelines for CGN SEO Tools

## Project Overview

CGN SEO Tools is a Streamlit-based web application for SEO analysis. The primary tool implements a PageRank-style algorithm to analyze internal link structure and calculate importance scores for website URLs based on Screaming Frog CSV exports.

## Codebase Structure

```
cgn-seo-tools/
├── app.py                      # Main Streamlit entry point (home page)
├── pages/
│   └── internal-pagerank.py    # PageRank analysis tool (core logic)
├── requirements.txt            # Python dependencies
└── CLAUDE.md                   # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app entry point - renders home page with sidebar navigation |
| `pages/internal-pagerank.py` | Core PageRank implementation with file upload, graph construction, algorithm execution, and results visualization |
| `requirements.txt` | Dependencies: `streamlit`, `pandas` |

## Development Workflow

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

### Adding New Tools

1. Create a new Python file in the `pages/` directory
2. File name becomes the page title (use hyphens for spaces)
3. Streamlit auto-discovers pages in this directory

## Code Conventions

### Python Style

- **Type hints**: Use extensively for function parameters and return types
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes
- **Imports**: Standard library first, then third-party packages

### Data Processing Patterns

- Handle multiple file encodings (UTF-8, Latin-1 fallback)
- Use pandas vectorized operations for performance
- Implement defensive programming for null/NaN values

### Streamlit Patterns

- Configuration controls in `st.sidebar`
- Use `st.expander()` for collapsible sections
- Provide CSV download for results
- Display data with `st.dataframe()` using column configuration

## Architecture Details

### PageRank Implementation (`pages/internal-pagerank.py`)

The goal is to model internal link flow the way Google plausibly does, not
textbook PageRank. The **Method** tab in the app is the user-facing statement of
the model and its limits — keep it in sync with any behaviour change.

**Modelling decisions (do not silently regress these):**

- Only hyperlinks pass PageRank. Image, CSS, JavaScript, hreflang, canonical and
  rel=next/prev rows are excluded via the `Type` column. This matters more than
  anything else on real exports.
- One page is one node. Fragments, scheme/host case, default ports and tracking
  parameters are normalised away; canonicalised URLs merge into their canonical.
- Redirects collapse into their final destination. Ambiguous chains are left
  unresolved and reported, never guessed at.
- `nofollow`, `sponsored` and `ugc` all block equity.
- Edges are weighted by link placement (reasonable surfer), so boilerplate
  footer links pass less than in-content links.
- Repeat links from one page to the same target count once.

**Key components:**

1. **`PRConfig`** (dataclass): all algorithm, normalisation, hygiene and
   placement-weight settings. `DEFAULT_POSITION_WEIGHTS` holds the
   reasonable-surfer weights by link position.
2. **`Diagnostics`** (dataclass): per-stage edge accounting. Every filter records
   what it removed so nothing disappears silently from the graph.
3. **`_detect_columns()`**: maps Screaming Frog / generic crawler column names.
   Note that SF's bare `Target` column is the anchor target attribute, not a URL,
   so it is deliberately the last candidate for the target URL.
4. **`_normalize_url()` / `_normalize_series()`**: URL canonicalisation.
   `_normalize_series` normalises distinct values only — the per-row path is hot.
5. **`_build_redirect_map()` / `_build_canonical_map()` / `_resolve_chains()`**:
   node consolidation, including redirect-loop detection.
6. **`build_edges()`**: the full pipeline from raw export to weighted,
   deduplicated edge list. Order matters — canonical/redirect maps are built
   *before* the link-type filter because they need the typed rows.
7. **`pagerank()`**: numpy weighted power iteration. `np.bincount` for the
   scatter-add; O(E) per iteration. Dangling nodes redistribute uniformly.
8. **`click_depth()`**: BFS from the homepage over a CSR-style index.
   Returns `-1` for unreachable — treat that as worse than deep, not as `0`.
9. **`donor_suggestions()`**: highest-authority pages not yet linking to a
   priority page, ranked by equity per new link.

### Input Format

Screaming Frog **All Inlinks** export (`Bulk Export → Links → All Inlinks`).
Required: source and destination URL columns (`Source`/`Destination` or
`From`/`To`). Strongly recommended, each materially improving accuracy: `Type`,
`Status Code`, `Follow`, `Link Position`, `Anchor`, `Alt Text`. The app warns
when any of these are unmapped and explains the consequence.

### Output Format

DataFrame with `url`, `pagerank`, `pagerank_pct`, `pagerank_percentile`, `rank`,
`in_links`, `out_links`, `out_link_weight`, `click_depth`, `status`,
`is_dead_end` — plus the weighted edge list, a wasted-equity report, and
per-priority-page link recommendations.

### Performance

Graph build and PageRank are vectorised; ~50k nodes / 500k edges runs in a few
seconds. Avoid `df.iterrows()` and per-row `apply` on the edge frame — use
`_map_unique()` or a dict `.map()` instead. A previous revision used
`iterrows()` here and took minutes on large exports.

## Guidelines for AI Assistants

### When Adding Features

- Keep dependencies minimal
- Follow existing type hint patterns
- Add new pages to `pages/` directory for new tools
- Use `st.sidebar` for configuration options
- Provide helpful UI text explaining feature usage

### When Modifying PageRank Logic

- Preserve numerical stability (normalization to sum=1)
- Handle edge cases: dangling nodes, self-loops, nofollow links, empty graphs
- Maintain convergence detection, and surface non-convergence to the user
- Keep the **Method** tab honest — it documents both the model and its limits.
  If you change what the model does, change that text in the same commit.
- Do not add SEO heuristics that are not defensible as Google behaviour. Where a
  weighting is an approximation (placement weights), expose it as a control so
  users can test how sensitive their conclusions are to it.

### When Debugging

- Check file encoding issues first (common CSV problem)
- Verify column detection for different Screaming Frog versions
- Confirm the `Type` column is mapped — without it, image/CSS/JS rows are
  counted as links and every score is wrong
- Test with small datasets before large ones

### Testing

Two suites, both plain scripts that exit non-zero on failure:

- `python tests/test_internal_pagerank.py` — dependency-free (115+ checks):
  URL normalisation, column detection, link-type and placement classification,
  redirect/canonical consolidation, PageRank against a slow reference
  implementation, BFS depth, degenerate inputs, performance smoke test.
- `python tests/test_ui_render.py` — drives the page through
  `streamlit.testing.v1.AppTest`. Streamlit swallows render exceptions and shows
  them in the browser only, so this is the only way to catch a broken tab.
  Requires streamlit; skips cleanly if it is absent.

Run both after any change to the page — the pure-logic suite will not notice a
render regression, and vice versa.

When making changes, also check edge cases by hand: empty files, single node, no
edges, exports with only `From`/`To` columns.

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation and analysis |
| numpy | Vectorised PageRank power iteration |

All use latest versions (not pinned).
