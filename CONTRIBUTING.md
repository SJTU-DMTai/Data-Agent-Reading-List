# Contributing

Thanks for helping grow this reading list! **Do not edit `README.md` directly** — it is
auto-generated from the YAML files in [`data/`](data/).

## Add a paper

1. Add an entry to the right file:
   - [`data/papers.yaml`](data/papers.yaml) — research papers (pick a `category`, see below)
   - [`data/benchmarks.yaml`](data/benchmarks.yaml) — benchmarks (please include a TLDR)
   - [`data/surveys.yaml`](data/surveys.yaml) — surveys and vision papers

   For arXiv papers there is a helper that fetches the metadata for you:

   ```bash
   python scripts/add_paper.py https://arxiv.org/abs/2510.16872 --category data-science
   ```

   **Only know the title?** This resolves the paper (Semantic Scholar / arXiv), auto-picks a
   category via the LLM, adds it, and regenerates the README — all in one step:

   ```bash
   export DEEPSEEK_API_KEY=sk-...   # for auto-classify; or pass -c to set the category
   python scripts/add_by_title.py "APEX-SQL: Talking to the data via Agentic Exploration"
   python scripts/add_by_title.py "Title A" "Title B" -c nl2sql   # several, fixed category
   ```

   Please verify the **corresponding author** (arXiv does not expose it) and prefer the
   author list from the paper's PDF.

2. Regenerate the README:

   ```bash
   pip install pyyaml
   python scripts/generate_readme.py
   ```

3. Open a pull request. CI checks that the README matches the data and that links resolve.

Not comfortable with YAML? Just [open an issue](../../issues/new/choose) with the paper
link — we will take it from there.

## Categories

| Key | Section |
|:---|:---|
| `data-preparation` | Data Preparation & Integration |
| `nl2sql` | NL2SQL (Text-to-SQL) |
| `table-reasoning` | Table Understanding & Reasoning |
| `data-analysis` | Data Analysis & Insight Discovery |
| `data-science` | Data Science & Machine Learning Agents |
| `db-operations` | Database Operations & Diagnosis |
| `memory` | Agent Memory & Context Engineering |
| `foundations` | General Agent Techniques |

Use `tags` for cross-cutting topics (e.g. `memory`, `nl2sql`, `multi-agent`, `rag`,
`multimodal`, `context-engineering`).

## Selection criteria

A candidate is included only if it clears one of three gates — **substance**,
**top institution**, or **top-venue acceptance**. The full rubric lives in
[`scripts/curation_rubric.md`](scripts/curation_rubric.md) and is what both human
maintainers and the automated triage follow. In short:

- Directly about **LLM-based agents for data tasks**, the **memory/context machinery**
  that powers them, or benchmarks evaluating them.
- Real artifact (named system/benchmark/method) — not a keyword match or a thin position
  piece. `memory` and `foundations` are held to a stricter bar than the core directions.
- One paper, one entry — if a benchmark paper is also a system paper, put it where
  readers are most likely to look for it.

## Automated curation pipeline

The whole "find new papers → judge them → hand a shortlist to a human" loop is scripted,
so it can run without any manual review or an interactive LLM session:

```bash
# 1. one command: scan arXiv, enrich, triage against the rubric, emit a digest
export DEEPSEEK_API_KEY=sk-...        # any OpenAI-compatible key; see below
python scripts/curate.py --from 2026-04-10 --to 2026-07-12 --out digest.md

# 2. review digest.md, then add the ones you want (links + code auto-fetched)
python scripts/add_paper.py 2504.01234 -c nl2sql
python scripts/generate_readme.py     # regenerate README from data/
```

`curate.py` chains three reusable steps you can also run on their own:

| Script | Does |
|---|---|
| `scripts/enrich.py` | arXiv ids → title/abstract/**acceptance comment** + Semantic Scholar citations/affiliations |
| `scripts/llm_triage.py` | enriched records → `keep`/`maybe`/`drop` + category + which gate, per the rubric |
| `scripts/curate.py` | scan → enrich → triage → Markdown digest grouped by category |

**LLM provider.** Triage uses any OpenAI-compatible chat API, configured by env vars —
defaults to DeepSeek (cheap, fine for this classification task):

| Var | Default | Notes |
|---|---|---|
| `LLM_API_KEY` / `DEEPSEEK_API_KEY` | — | required |
| `LLM_BASE_URL` | `https://api.deepseek.com` | e.g. `https://api.openai.com/v1` |
| `LLM_MODEL` | `deepseek-chat` | |

Run `python scripts/curate.py --no-triage` (or omit the key) to get a relevance-only
list without gate judgment.

## Weekly arXiv digest

The same pipeline runs as a [GitHub Action](.github/workflows/weekly-arxiv.yml) every
Monday and opens a `paper-candidate` issue with the triaged shortlist. Add a
`DEEPSEEK_API_KEY` repo secret to enable gate-based triage (without it the digest still
lists relevant candidates, untriaged). You can also trigger it manually with **from**/**to**
dates to backfill a gap. Maintainers tick the papers to add, then add them via the steps
above.
