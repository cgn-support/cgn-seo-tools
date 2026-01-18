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

**Key Components:**

1. **`PRConfig`** (dataclass): Algorithm configuration
   - `damping`: Damping factor (default 0.85)
   - `max_iters`: Maximum iterations (default 50)
   - `tol`: Convergence tolerance (default 1e-8)
   - `include_nofollow`: Include nofollow links
   - `restrict_to_internal`: Restrict to same domain
   - `drop_self_loops`: Remove self-referencing links

2. **`_detect_columns()`**: Auto-detects Screaming Frog CSV columns
   - Maps common column name variations
   - Returns dict with source, target, follow, status columns

3. **`_build_graph()`**: Constructs directed graph from CSV
   - Filters edges by configuration
   - Returns adjacency list and cleaned edges DataFrame

4. **`pagerank()`**: Core algorithm
   - Handles dangling nodes (redistributes mass uniformly)
   - Uses convergence detection (L1 delta)
   - Returns scores dict, iterations, final delta

### Input Format

Expects Screaming Frog internal link export CSV with columns like:
- Source/From URL
- Destination/To URL
- Follow/Nofollow status (optional)
- Status codes (optional)

### Output Format

DataFrame with:
- `url`: Page URL
- `pagerank`: Raw importance score
- `pagerank_pct`: Score as percentage
- `in_links`: Incoming link count
- `out_links`: Outgoing link count
- `rank`: Ordinal ranking

## Guidelines for AI Assistants

### When Adding Features

- Keep dependencies minimal
- Follow existing type hint patterns
- Add new pages to `pages/` directory for new tools
- Use `st.sidebar` for configuration options
- Provide helpful UI text explaining feature usage

### When Modifying PageRank Logic

- Preserve numerical stability (normalization to sum=1)
- Handle edge cases: dangling nodes, self-loops, nofollow links
- Maintain convergence detection
- Update interpretation guide if behavior changes

### When Debugging

- Check file encoding issues first (common CSV problem)
- Verify column detection for different Screaming Frog versions
- Test with small datasets before large ones

### Testing

No formal test suite exists. When making changes:
- Test with sample Screaming Frog CSV exports
- Verify UI responsiveness
- Check edge cases (empty files, single node, no edges)

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation and analysis |

Both use latest versions (not pinned).
