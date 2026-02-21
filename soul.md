# Soul (Phase 9 - Polish + Soul)

This document defines how we name things and how we speak in product-facing copy, especially terminal and API messages.

## Naming style

- Use `snake_case` for Python variables, functions, and module filenames.
- Use `UPPER_SNAKE_CASE` for module-level constants.
- Use `kebab-case` for CLI flags (for example, `--auto-tune-thresholds`).
- Keep data keys stable and explicit (`record_id`, `cluster_id`, `true_cluster_id`, `similarity_threshold`).
- Use the domain vocabulary consistently: `cluster`, `canonical label`, `threshold`, `embedding`, `evaluation report`.
- Prefer explicit names over abbreviations unless they are standard (`api`, `json`, `xlsx`).

### Naming Do/Don't

| Avoid | Prefer |
|---|---|
| `thr`, `simThr` | `similarity_threshold` |
| `cid` | `cluster_id` |
| `lbls` | `labels` or `canonical_labels` |
| Mixed terms (`group`, `bucket`, `cluster`) in one path | One term per concept (`cluster`) |

## Product tone

### Principles

- Neutral: describe system state, not emotion.
- Concise: short sentences with concrete wording.
- Factual: include only what we can verify from execution state.
- Actionable: when possible, include what to do next.
- Consistent: same concept should use the same term everywhere.

### Tone Do/Don't

| Avoid | Prefer |
|---|---|
| "Oops, something went wrong." | "Unexpected server error while processing upload." |
| "Sorry, your file is bad." | "Expected a .xlsx file upload." |
| "Great news! All done!" | "Wrote markdown report to {output_path}" |
| "You forgot to set the key." | "Server is missing OPENAI_API_KEY for embeddings." |

## Terminal messages

Use predictable message shapes so users can scan output quickly.

### Message templates

- Usage: `Usage: <command> <required_arg> [optional_arg]`
- Info: `<Process>: <current step>`
- Success: `<Action completed> to <path/output>`
- Warning: `<Issue detected>. <Impact or fallback>.`
- Error: `Error <context>: <reason>. <next action if available>.`

### Formatting rules

- Use sentence case.
- End complete sentences with a period.
- Keep one message focused on one issue.
- Include a next action when the user can fix it.
- Avoid hype, emojis, and exclamation marks.
- Avoid blame language ("you did X wrong"); describe constraints instead.

### Terminal phrasing Do/Don't

| Avoid | Prefer |
|---|---|
| "Uh oh, file read failed!" | `Error reading input file: {exc}` |
| "Need thresholds, please provide." | `Provide at least one threshold value.` |
| "File type is wrong." | `Expected a .xlsx file upload.` |
| "Quota issue." | `Embedding provider quota/rate limit reached. Check OPENAI_API_KEY billing/quota and retry.` |

### Approved examples

- `Usage: python generate_report.py <evaluation_json_path> [output_path]`
- `Wrote markdown report to {output_path}`
- `Error reading input file: {exc}`
- `Provide at least one threshold value.`
- `Expected a .xlsx file upload.`
- `Embedding provider quota/rate limit reached. Check OPENAI_API_KEY billing/quota and retry.`
