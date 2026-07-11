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

- Directly about **LLM-based agents for data tasks**, the **memory/context machinery**
  that powers them, or benchmarks evaluating them.
- Peer-reviewed venues preferred; arXiv preprints are welcome if they are influential or
  from an active research line.
- One paper, one entry — if a benchmark paper is also a system paper, put it where
  readers are most likely to look for it.

## Weekly arXiv digest

A [GitHub Action](.github/workflows/weekly-arxiv.yml) scans arXiv every Monday and opens
a `paper-candidate` issue with new matches. Maintainers triage it: tick the papers to
add, then add them via the steps above.
