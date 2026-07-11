#!/usr/bin/env python3
"""Add a paper to data/papers.yaml from an arXiv link.

Usage:
    python scripts/add_paper.py https://arxiv.org/abs/2510.16872 --category data-science
    python scripts/add_paper.py 2510.16872 -c memory -t procedural-memory -t rl

Fetches title/authors from the arXiv API and appends a YAML entry. The corresponding
author is not available from arXiv, so the last author is used with a TODO marker —
please verify it before committing.
"""
import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "papers.yaml"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
CODE_URL = re.compile(r"https?://(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-]+/[\w.\-]+", re.I)

CATEGORIES = ["data-preparation", "nl2sql", "table-reasoning", "data-analysis",
              "data-science", "db-operations", "memory", "foundations"]


def arxiv_id(text: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", text)
    if not m:
        sys.exit(f"error: could not find an arXiv id in {text!r}")
    return m.group(1)


def fetch(aid: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={aid}"
    with urllib.request.urlopen(url, timeout=30) as r:
        root = ET.fromstring(r.read())
    entry = root.find(f"{ATOM}entry")
    if entry is None or entry.find(f"{ATOM}title") is None:
        sys.exit(f"error: arXiv id {aid} not found")
    title = re.sub(r"\s+", " ", entry.find(f"{ATOM}title").text).strip()
    authors = [a.find(f"{ATOM}name").text for a in entry.findall(f"{ATOM}author")]
    published = entry.find(f"{ATOM}published").text  # e.g. 2025-10-19T...
    # Code repos are not a structured arXiv field; sniff the abstract and comment.
    haystack = entry.find(f"{ATOM}summary").text or ""
    comment = entry.find(f"{ARXIV_NS}comment")
    if comment is not None and comment.text:
        haystack += " " + comment.text
    code = CODE_URL.search(haystack)
    return {"title": title, "authors": authors, "published": published,
            "code": code.group(0).rstrip(".") if code else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv", help="arXiv URL or id, e.g. 2510.16872")
    parser.add_argument("-c", "--category", required=True, choices=CATEGORIES)
    parser.add_argument("-t", "--tag", action="append", default=[], dest="tags")
    parser.add_argument("--code", help="code repository URL")
    args = parser.parse_args()

    aid = arxiv_id(args.arxiv)
    if aid in PAPERS.read_text():
        sys.exit(f"error: arXiv id {aid} already exists in {PAPERS}")

    meta = fetch(aid)
    year, month = meta["published"][:4], meta["published"][5:7]
    lines = [
        "",
        f'- title: "{meta["title"]}"',
        f"  authors: {meta['authors'][-1]}  # TODO verify corresponding author",
        f"  venue: arXiv'{year[2:]}.{month}",
        f"  year: {year}",
        f"  paper: https://arxiv.org/pdf/{aid}",
    ]
    code = args.code or meta["code"]
    if code:
        suffix = "" if args.code else "  # auto-detected from abstract, please verify"
        lines.append(f"  code: {code}{suffix}")
    lines.append(f"  category: {args.category}")
    if args.tags:
        lines.append(f"  tags: [{', '.join(args.tags)}]")
    with PAPERS.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nAppended to {PAPERS}. Now run: python scripts/generate_readme.py")


if __name__ == "__main__":
    main()
