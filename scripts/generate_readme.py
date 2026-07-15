#!/usr/bin/env python3
"""Generate README.md from data/*.yaml.

Usage:
    python scripts/generate_readme.py            # rewrite README.md
    python scripts/generate_readme.py --check    # exit 1 if README.md is out of date (for CI)
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
README = ROOT / "README.md"

# (key, section title, blurb)
CATEGORIES = [
    ("data-preparation", "🧹 Data Preparation & Integration",
     "Agents that clean, transform, impute, and integrate data — structured or unstructured — before analysis."),
    ("nl2sql", "💬 NL2SQL (Text-to-SQL)",
     "Translating natural language questions into SQL, increasingly via agentic exploration, decomposition, and self-correction. See also papers tagged `nl2sql` in other sections."),
    ("table-reasoning", "📋 Table Understanding & Reasoning",
     "Question answering and reasoning over tables and semi-structured data, including SQL-hybrid approaches."),
    ("table-curation", "🧱 Table Generation, Curation & Synthesis",
     "Constructing and improving the tables themselves: synthetic table / tabular-data generation, table synthesis and augmentation, schema and column expansion, and dataset curation for training and evaluation."),
    ("data-analysis", "📊 Data Analysis & Insight Discovery",
     "End-to-end analytics systems: EDA, BI platforms, semantic operators over data lakes, report generation, and insight discovery."),
    ("data-science", "🔬 Data Science & Machine Learning Agents",
     "Autonomous agents for the full data science lifecycle: modeling, AutoML, and Kaggle-style competitions."),
    ("db-operations", "🛠️ Database Operations & Diagnosis",
     "LLM agents for database administration: configuration debugging, performance diagnosis, and tuning."),
    ("memory", "🧠 Agent Memory & Context Engineering",
     "Memory is a core capability for long-horizon data agents: episodic/procedural memory, memory operating systems, context folding, and experience-driven self-evolution."),
    ("foundations", "🧩 General Agent Techniques",
     "General-purpose agent building blocks frequently used by data agents: planning, workflow generation, multi-agent frameworks, and RAG."),
]

HEADER = """\
# Awesome Data Agents

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Commit](https://img.shields.io/github/last-commit/SJTU-DMTai/Awesome-Data-Agent-Papers)](https://github.com/SJTU-DMTai/Awesome-Data-Agent-Papers/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/SJTU-DMTai/Awesome-Data-Agent-Papers?style=social)](https://github.com/SJTU-DMTai/Awesome-Data-Agent-Papers/stargazers)

> A curated reading list on **LLM-based data agents** — autonomous agents that prepare,
> query, analyze, and learn from data. Covers data preparation, NL2SQL & table reasoning,
> data analysis & business intelligence, data science & AutoML agents, database operations,
> and **agent memory** (a key capability for long-horizon data agents), plus the benchmarks
> to evaluate them. Maintained by [SJTU-DMTai](https://github.com/SJTU-DMTai).

**How this list is maintained:** every paper lives as structured metadata in
[`data/`](data/), the README is auto-generated, links are checked in CI, and a weekly
GitHub Action scans arXiv for new candidate papers. Found a missing paper?
[Open an issue](https://github.com/SJTU-DMTai/Awesome-Data-Agent-Papers/issues/new/choose)
or see [CONTRIBUTING.md](CONTRIBUTING.md) — PRs are welcome!
"""

INTRO = """\
## Introduction

A **data agent** is an LLM-powered autonomous system that operates over data: it interprets
a user's intent expressed in natural language and then plans and executes the whole
workflow of **preparing, querying, analyzing, and learning from data** — calling tools
(SQL engines, code, databases, retrieval), inspecting intermediate results, and
self-correcting along the way. As data work shifts from hand-written pipelines to
natural-language-driven automation, data agents are becoming the interface between people
and their data: they lower the barrier to analytics and increasingly take on end-to-end
tasks that used to require a dedicated data engineer, analyst, or scientist.

Building a capable data agent spans many sub-problems. We **structure this list around the
directions we see as core to that goal** — from getting data ready, to querying and
reasoning over it, to full-lifecycle analysis and the agent capabilities underneath:

- 🧹 **Data Preparation & Integration** — clean, transform, impute, and integrate raw data
- 💬 **NL2SQL** — translate natural-language questions into executable SQL
- 📋 **Table Understanding & Reasoning** — QA and reasoning over tables and semi-structured data
- 🧱 **Table Generation, Curation & Synthesis** — construct and improve the tables themselves
- 📊 **Data Analysis & Insight Discovery** — EDA, BI, semantic operators, report generation
- 🔬 **Data Science & ML Agents** — the full modeling / AutoML lifecycle
- 🛠️ **Database Operations & Diagnosis** — configuration, tuning, and diagnosis for DBAs
- 🧠 **Agent Memory & Context Engineering** — a key capability for long-horizon data agents
- 🧩 **General Agent Techniques** — the planning, workflow, multi-agent, and RAG building blocks underneath

— plus the **surveys** that frame the field and the **benchmarks** that measure progress.
"""

FOOTER = ""


def slugify(title: str) -> str:
    """GitHub-style anchor slug."""
    s = title.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.ASCII)
    return s.replace(" ", "-")


def md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").strip()


def render_tags(tags) -> str:
    if not tags:
        return ""
    return " " + " ".join(f"`{t}`" for t in tags)


def render_links(item: dict) -> str:
    links = [f"[paper]({item['paper']})"]
    if item.get("code"):
        links.append(f"[code]({item['code']})")
    return " ".join(links)


def render_surveys(surveys: list) -> list:
    lines = []
    for s in sorted(surveys, key=lambda x: -x.get("year", 0)):
        extra = f" ({s['note']})" if s.get("note") else ""
        author = f" — {s['corresponding']}" if s.get("corresponding") else ""
        lines.append(f"- **{s['venue']}** — [{md_escape(s['title'])}]({s['paper']}){author}{extra}")
    lines.append("")
    return lines


def render_benchmark_group(group: dict) -> list:
    lines = [f"### {group['name']}", ""]
    label = group["label_column"]
    if group.get("show_tldr"):
        lines.append(f"| Benchmark | {label} | Year | Links | TLDR |")
        lines.append("|:---:|:---:|:---:|:---:|:---|")
    else:
        lines.append(f"| Benchmark | {label} | Year | Links |")
        lines.append("|:---:|:---:|:---:|:---:|")
    for it in group["items"]:
        row = [it["name"], it["label"], str(it["year"]), render_links(it)]
        if group.get("show_tldr"):
            row.append(md_escape(it.get("tldr", "")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _paper_sort_key(p):
    # Sort by the year SHOWN in the venue label, so the order matches the visible
    # venue column (the `year` field holds the preprint year, which often differs
    # from the acceptance year — e.g. a 2025 preprint accepted at SIGMOD'26).
    # Venue formats: "SIGMOD'26", "arXiv'26.07", "IEEE VIS'23", or no year at all.
    # Within a venue-year, conference papers (no month) sort above arXiv preprints,
    # then preprints newest-month-first; title breaks ties for a stable order.
    v = p.get("venue", "")
    m = re.search(r"'(\d{2})(?:\.(\d{2}))?", v)
    if m:
        year = 2000 + int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 99  # conf w/o month tops its year
    else:
        year, month = p.get("year", 0), 99
    return (-year, -month, p.get("title", "").lower())


def render_paper_rows(rows: list) -> list:
    """A paper table (Venue | Paper | Corresp. Author | Links) for a pre-filtered list."""
    lines = ["| Venue | Paper | Corresp. Author | Links |",
             "|:---|:---|:---:|:---:|"]
    for p in sorted(rows, key=_paper_sort_key):
        title = md_escape(p["title"]) + render_tags(p.get("tags"))
        lines.append(f"| {p['venue']} | {title} | {md_escape(p.get('authors', ''))} | {render_links(p)} |")
    lines.append("")
    return lines


def render_papers(papers: list, key: str) -> list:
    return render_paper_rows([p for p in papers if p["category"] == key])


def render_resources(res: dict) -> list:
    """Render the community-resources block from data/resources.yaml (all parts optional)."""
    lines = []
    talks = res.get("workshops_tutorials") or []
    if talks:
        lines += ["### 📅 Workshops & Tutorials", ""]
        for t in talks:
            links = " ".join(f"[{k}]({v})" for k, v in (t.get("links") or {}).items())
            lines.append(f"- **[{t['venue']}]** {md_escape(t['title'])} {links}".rstrip())
        lines.append("")
    repos = res.get("related_repos") or []
    if repos:
        lines += ["### 🔗 Related Repositories", ""]
        for r in repos:
            note = f" — {r['note']}" if r.get("note") else ""
            lines.append(f"- [{r['name']}]({r['url']}){note}")
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify README.md is up to date instead of writing it")
    args = parser.parse_args()

    surveys = yaml.safe_load((DATA / "surveys.yaml").read_text())
    benchmarks = yaml.safe_load((DATA / "benchmarks.yaml").read_text())
    papers = yaml.safe_load((DATA / "papers.yaml").read_text())

    known = {c[0] for c in CATEGORIES}
    for p in papers:
        if p["category"] not in known:
            sys.exit(f"error: unknown category {p['category']!r} in paper {p['title']!r}")

    sections = []
    sections.append(("📚 Surveys & Vision", render_surveys(surveys)))

    bench_lines = []
    for group in benchmarks:
        bench_lines += render_benchmark_group(group)
    sections.append(("🏆 Benchmarks", bench_lines))

    for key, title, blurb in CATEGORIES:
        body = [f"> {blurb}", ""] + render_papers(papers, key)
        sections.append((title, body))

    res_path = DATA / "resources.yaml"
    resources = yaml.safe_load(res_path.read_text()) if res_path.exists() else {}
    resource_body = render_resources(resources or {})
    if resource_body:
        sections.append(("🌐 Community & Resources", resource_body))

    n_papers = len(papers) + len(surveys) + sum(len(g["items"]) for g in benchmarks)

    out = [HEADER]
    out.append(f"**{n_papers} papers** and counting. Last generated from "
               f"[`data/`](data/) — do not edit this file by hand.\n")
    out.append(INTRO)
    out.append("## Contents\n")
    out.append("- [Introduction](#introduction)")
    for title, _ in sections:
        out.append(f"- [{title}](#{slugify(title)})")
    out.append("")
    for title, body in sections:
        out.append(f"## {title}\n")
        out.extend(body)
    out.append(FOOTER)
    content = "\n".join(out)

    if args.check:
        if README.read_text() != content:
            print("README.md is out of date. Run: python scripts/generate_readme.py")
            return 1
        print("README.md is up to date.")
        return 0

    README.write_text(content)
    print(f"Wrote {README} ({n_papers} papers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
