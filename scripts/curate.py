#!/usr/bin/env python3
"""One-command curation pipeline: scan arXiv -> enrich -> LLM-triage -> digest.

This automates end-to-end what used to be a manual review: it finds candidate papers,
enriches them, judges each against scripts/curation_rubric.md via an LLM (DeepSeek by
default — see scripts/llm_triage.py for the LLM_* env vars), and writes a Markdown digest
grouped by category listing only KEEP / MAYBE papers, each with a ready-to-run
`add_paper.py` command. Nothing is added automatically — the digest is for human sign-off.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    # weekly window (default 8 days):
    python scripts/curate.py
    # backfill a gap:
    python scripts/curate.py --from 2026-04-10 --to 2026-07-12
    # skip the LLM (relevance list only, no gate judgment):
    python scripts/curate.py --from 2026-04-10 --no-triage

Exit code 1 when nothing new is found (so a workflow can skip opening an issue).
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arxiv_scan  # noqa: E402  (reuse QUERIES, search, known_ids)
from enrich import enrich  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CATEGORY_TITLES = {
    "data-preparation": "🧹 Data Preparation & Integration",
    "nl2sql": "💬 NL2SQL (Text-to-SQL)",
    "table-reasoning": "📋 Table Understanding & Reasoning",
    "data-analysis": "📊 Data Analysis & Insight Discovery",
    "data-science": "🔬 Data Science & ML Agents",
    "db-operations": "🛠️ Database Operations & Diagnosis",
    "memory": "🧠 Agent Memory & Context Engineering",
    "foundations": "🧩 General Agent Techniques",
    "benchmark": "🏆 Benchmark",
    "survey": "📚 Survey / Vision",
    "": "❓ Uncategorized",
}


def scan(date_from, date_to, days, max_per_query):
    """Return fresh candidate records (not already in data/*.yaml)."""
    if date_from:
        lo = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        hi = (datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
              if date_to else datetime.now(timezone.utc))
        date_range, cutoff = (lo, hi), lo
    else:
        date_range, cutoff = None, datetime.now(timezone.utc) - timedelta(days=days)

    seen, fresh = arxiv_scan.known_ids(), {}
    for q in arxiv_scan.QUERIES:
        try:
            results = arxiv_scan.search(q, max_per_query, date_range)
        except Exception as exc:
            print(f"warning: query {q!r} failed: {exc}", file=sys.stderr)
            continue
        for r in results:
            if r["published"] >= cutoff and r["id"] not in seen and r["id"] not in fresh:
                fresh[r["id"]] = r
    return sorted(fresh.values(), key=lambda r: r["published"], reverse=True), date_range, days


def digest(verdicts, span):
    keep = [v for v in verdicts if v["decision"] == "keep"]
    maybe = [v for v in verdicts if v["decision"] == "maybe"]
    lines = [f"Curation digest for **{span}** — "
             f"**{len(keep)} KEEP**, {len(maybe)} MAYBE "
             f"(from {len(verdicts)} candidates that passed relevance).",
             "",
             "Each paper cleared at least one gate (A=substance, B=institution, "
             "C=top venue). Review, then run the `add_paper.py` command to include it.",
             ""]
    for bucket, name in ((keep, "✅ KEEP"), (maybe, "🤔 MAYBE (borderline)")):
        if not bucket:
            continue
        lines.append(f"# {name}\n")
        by_cat = {}
        for v in bucket:
            by_cat.setdefault(v.get("category") or "", []).append(v)
        for cat in CATEGORY_TITLES:
            group = by_cat.get(cat)
            if not group:
                continue
            lines.append(f"## {CATEGORY_TITLES[cat]}\n")
            for v in group:
                gates = f" `gates:{v['gates']}`" if v.get("gates") else ""
                cites = v.get("citations")
                cmeta = f", {cites} cites" if cites else ""
                lines.append(f"- [ ] **[{v['title']}]"
                             f"(https://arxiv.org/abs/{v['id']})** "
                             f"({v.get('published', '?')}{cmeta}){gates}")
                if v.get("reason"):
                    lines.append(f"      <sub>{v['reason']}</sub>")
                add_cat = cat if cat in (
                    "data-preparation", "nl2sql", "table-reasoning", "data-analysis",
                    "data-science", "db-operations", "memory", "foundations") else "TODO"
                lines.append(f"      <sub>`python scripts/add_paper.py {v['id']} "
                             f"-c {add_cat}`</sub>")
            lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD",
                   help="backfill start date (else scan the last --days days)")
    p.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD",
                   help="backfill end date (default: today)")
    p.add_argument("--days", type=int, default=8, help="window size when no --from")
    p.add_argument("--max-per-query", type=int, default=60)
    p.add_argument("--no-triage", action="store_true",
                   help="skip the LLM; list all relevant candidates uncategorized")
    p.add_argument("--out", help="write the digest here instead of stdout")
    args = p.parse_args()

    records, date_range, days = scan(args.date_from, args.date_to, args.days,
                                     args.max_per_query)
    if not records:
        print("nothing new found", file=sys.stderr)
        return 1
    span = (f"{date_range[0]:%Y-%m-%d} → {date_range[1]:%Y-%m-%d}" if date_range
            else f"the last {days} days")
    print(f"scanned: {len(records)} fresh candidates", file=sys.stderr)

    enriched = enrich([r["id"] for r in records])
    print(f"enriched: {len(enriched)} records", file=sys.stderr)

    if args.no_triage:
        verdicts = [{**r, "decision": "maybe", "category": "", "gates": "",
                     "reason": ""} for r in enriched]
    else:
        from llm_triage import triage
        verdicts = triage(enriched)

    text = digest(verdicts, span)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote digest to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
