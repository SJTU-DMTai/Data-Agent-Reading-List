#!/usr/bin/env python3
"""Scan arXiv for recent papers matching this list's topics and print a Markdown digest.

Used by .github/workflows/weekly-arxiv.yml to open a weekly digest issue.
Papers whose arXiv id already appears anywhere in data/*.yaml are skipped.

Usage:
    python scripts/arxiv_scan.py [--days 8] [--max-per-query 60]
"""
import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATOM = "{http://www.w3.org/2005/Atom}"

QUERIES = [
    '"data agent"',
    '"data agents"',
    '"data science agent"',
    '"data analysis agent"',
    '"data analytics" AND "LLM agent"',
    '"table reasoning"',
    '"tabular reasoning"',
    '"text-to-SQL" AND agent',
    '"agent memory"',
    '"memory" AND "LLM agents"',
    '"context engineering"',
]
CATS = "cat:cs.DB OR cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.IR"


def known_ids() -> set:
    ids = set()
    for f in (ROOT / "data").glob("*.yaml"):
        ids |= set(re.findall(r"(\d{4}\.\d{4,5})", f.read_text()))
    return ids


def search(query: str, max_results: int, retries: int = 3) -> list:
    q = f"({CATS}) AND ({query})"
    url = ("https://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(q)
           + f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))  # arXiv rate-limits bursts
    out = []
    for e in root.findall(f"{ATOM}entry"):
        title = re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip()
        aid = re.search(r"(\d{4}\.\d{4,5})", e.find(f"{ATOM}id").text).group(1)
        published = datetime.fromisoformat(e.find(f"{ATOM}published").text.replace("Z", "+00:00"))
        authors = [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")]
        summary = re.sub(r"\s+", " ", e.find(f"{ATOM}summary").text).strip()
        out.append({"id": aid, "title": title, "published": published,
                    "authors": authors, "summary": summary, "query": query})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--max-per-query", type=int, default=60)
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    seen, fresh = known_ids(), {}
    for q in QUERIES:
        try:
            results = search(q, args.max_per_query)
        except Exception as exc:  # keep the digest going if one query fails
            print(f"<!-- query {q!r} failed: {exc} -->")
            continue
        for r in results:
            if r["published"] >= cutoff and r["id"] not in seen and r["id"] not in fresh:
                fresh[r["id"]] = r
        time.sleep(3)  # arXiv API rate-limit etiquette

    if not fresh:
        return 1  # signal "nothing to report" so the workflow skips the issue

    papers = sorted(fresh.values(), key=lambda r: r["published"], reverse=True)
    print(f"Found **{len(papers)}** candidate papers from the last {args.days} days. "
          "Check the ones to add, then add them via `scripts/add_paper.py` "
          "(see CONTRIBUTING.md), or close if none apply.\n")
    for p in papers:
        authors = ", ".join(p["authors"][:8]) + (" et al." if len(p["authors"]) > 8 else "")
        date = p["published"].strftime("%Y-%m-%d")
        print(f"- [ ] **[{p['title']}](https://arxiv.org/abs/{p['id']})** ({date})")
        print(f"      {authors}")
        print(f"      <sub>matched `{p['query']}` — {p['summary'][:300]}…</sub>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
