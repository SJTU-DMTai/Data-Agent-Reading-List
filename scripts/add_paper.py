#!/usr/bin/env python3
"""Add a paper to data/papers.yaml from an arXiv link.

Usage:
    python scripts/add_paper.py https://arxiv.org/abs/2510.16872 --category data-science
    python scripts/add_paper.py 2510.16872 -c memory -t procedural-memory -t rl
    python scripts/add_paper.py 2602.16720 -c nl2sql --venue "KDD'26"   # override venue

The venue is grounded from multiple sources (arXiv comment, Semantic Scholar, DBLP,
and optionally web+LLM) via scripts/venue.py. If none can be found, the paper is
written as `arXiv'YY.MM  # TODO verify venue` so it is easy to spot and fix.

The corresponding author is not available from arXiv, so the last author is used with a
TODO marker — please verify it before committing.
"""
import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from venue import resolve_venue, normalize_venue  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "papers.yaml"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
CODE_URL = re.compile(r"https?://(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-]+/[\w.\-]+", re.I)

CATEGORIES = ["data-preparation", "nl2sql", "table-reasoning", "table-curation",
              "data-analysis", "data-science", "db-operations", "memory", "foundations"]


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
    comment_el = entry.find(f"{ARXIV_NS}comment")
    comment = re.sub(r"\s+", " ", comment_el.text).strip() if comment_el is not None else ""
    # Code repos are not a structured arXiv field; sniff the abstract and comment.
    haystack = (entry.find(f"{ATOM}summary").text or "") + " " + comment
    code = CODE_URL.search(haystack)
    return {"id": aid, "title": title, "authors": authors, "published": published,
            "comment": comment, "code": code.group(0).rstrip(".") if code else None}


def insert_paper(aid, category, tags=(), code=None, venue=None, use_web=True):
    """Append one paper to data/papers.yaml. Returns a summary dict.

    venue: pass a string to force it (normalized to repo style when recognizable);
    leave None to ground it from arXiv comment / S2 / DBLP / web. When nothing grounds
    a real venue, falls back to arXiv'YY.MM and flags venue_todo=True.
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; choose from {CATEGORIES}")
    if aid in PAPERS.read_text():
        raise ValueError(f"arXiv id {aid} already exists in {PAPERS}")

    meta = fetch(aid)
    year, month = meta["published"][:4], meta["published"][5:7]
    fallback = f"arXiv'{year[2:]}.{month}"

    venue_todo = False
    venue_source = "manual"
    if venue:
        venue = normalize_venue(venue, year[2:]) or venue  # trust human input as-is if unusual
    else:
        venue, venue_source = resolve_venue(meta, use_web=use_web)
        if not venue:
            venue, venue_source, venue_todo = fallback, "arxiv-fallback", True

    lines = [
        "",
        f'- title: "{meta["title"]}"',
        f"  authors: {meta['authors'][-1]}  # TODO verify corresponding author",
        f"  venue: {venue}" + ("  # TODO verify venue" if venue_todo else ""),
        f"  year: {year}",
        f"  paper: https://arxiv.org/pdf/{aid}",
    ]
    code_auto = code is None            # was it sniffed from the abstract vs. passed in?
    code = code or meta["code"]
    if code:
        note = "  # auto-detected from abstract, please verify" if code_auto else ""
        lines.append(f"  code: {code}{note}")
    lines.append(f"  category: {category}")
    if tags:
        lines.append(f"  tags: [{', '.join(tags)}]")

    with PAPERS.open("a") as f:
        f.write("\n".join(lines) + "\n")

    return {"id": aid, "title": meta["title"], "venue": venue,
            "venue_source": venue_source, "venue_todo": venue_todo,
            "category": category, "code": code, "yaml": "\n".join(lines)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv", help="arXiv URL or id, e.g. 2510.16872")
    parser.add_argument("-c", "--category", required=True, choices=CATEGORIES)
    parser.add_argument("-t", "--tag", action="append", default=[], dest="tags")
    parser.add_argument("--code", help="code repository URL")
    parser.add_argument("--venue", help="force the venue (else grounded automatically)")
    parser.add_argument("--no-web", action="store_true", help="skip the web+LLM venue channel")
    args = parser.parse_args()

    try:
        res = insert_paper(arxiv_id(args.arxiv), args.category, tags=args.tags,
                           code=args.code, venue=args.venue, use_web=not args.no_web)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    print(res["yaml"])
    tag = " [TODO verify venue]" if res["venue_todo"] else f" [via {res['venue_source']}]"
    print(f"\nvenue: {res['venue']}{tag}")
    print(f"Appended to {PAPERS}. Now run: python scripts/generate_readme.py")


if __name__ == "__main__":
    main()
