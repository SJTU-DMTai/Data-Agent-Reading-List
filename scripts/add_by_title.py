#!/usr/bin/env python3
"""Add a paper to the list from just its TITLE — resolve, classify, and insert.

Given a paper title, this:
  1. resolves it to an arXiv paper (Semantic Scholar title match, arXiv search fallback),
  2. auto-classifies it into a category via the LLM (scripts/llm_triage.py; DeepSeek by
     default — or pass -c to set the category yourself),
  3. inserts it via scripts/add_paper.py (fetches authors/code, appends YAML),
  4. regenerates README.md.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/add_by_title.py "APEX-SQL: Talking to the data via Agentic Exploration"
    python scripts/add_by_title.py "Some Paper Title" -c nl2sql        # skip auto-classify
    python scripts/add_by_title.py "Some Paper Title" --yes            # accept fuzzy match
    python scripts/add_by_title.py "T1" "T2" "T3"                      # several at once
"""
import argparse
import difflib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich import enrich, fetch  # noqa: E402

HERE = Path(__file__).resolve().parent
ATOM = "{http://www.w3.org/2005/Atom}"
ACCEPT_THRESHOLD = 0.82  # title-similarity below this needs --yes / confirmation


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _s2_match(title):
    """Semantic Scholar's title-match endpoint returns the single closest paper."""
    url = ("https://api.semanticscholar.org/graph/v1/paper/search/match?query="
           + urllib.parse.quote(title) + "&fields=title,year,externalIds")
    try:
        data = json.loads(fetch(url))
    except Exception:
        return None
    for cand in data.get("data", []) or []:
        aid = (cand.get("externalIds") or {}).get("ArXiv")
        if aid:
            return {"id": aid, "title": cand["title"], "year": cand.get("year"),
                    "score": _ratio(title, cand["title"]), "source": "semantic-scholar"}
    return None


def _arxiv_query(q, title):
    url = ("https://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(q) + "&max_results=10")
    import re
    try:
        root = ET.fromstring(fetch(url))
    except Exception:
        return None
    best = None
    for e in root.findall(f"{ATOM}entry"):
        cand_title = " ".join(e.find(f"{ATOM}title").text.split())
        m = re.search(r"(\d{4}\.\d{4,5})", e.find(f"{ATOM}id").text)
        if not m:
            continue
        cand = {"id": m.group(1), "title": cand_title,
                "year": e.find(f"{ATOM}published").text[:4],
                "score": _ratio(title, cand_title), "source": "arxiv"}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _arxiv_search(title):
    """arXiv fallback: exact-phrase first, then a forgiving relevance search."""
    clean = title.replace('"', "")
    exact = _arxiv_query(f'ti:"{clean}"', title)
    if exact and exact["score"] >= 0.9:
        return exact
    # relevance search over all fields tolerates missing/re-ordered words
    relevance = _arxiv_query("all:" + clean, title)
    return max([c for c in (exact, relevance) if c],
               key=lambda c: c["score"], default=None)


def resolve(title):
    """Best arXiv match for a title, or None."""
    cands = [c for c in (_s2_match(title), _arxiv_search(title)) if c]
    return max(cands, key=lambda c: c["score"]) if cands else None


def add_one(title, category, tags, assume_yes):
    match = resolve(title)
    if not match:
        print(f"✗ no arXiv match found for: {title!r}", file=sys.stderr)
        return False
    print(f"→ matched: {match['title']} ({match['id']}, {match['year']}) "
          f"[{match['source']}, similarity {match['score']:.2f}]", file=sys.stderr)
    if match["score"] < ACCEPT_THRESHOLD and not assume_yes:
        print(f"  low similarity (<{ACCEPT_THRESHOLD}); re-run with --yes to accept, "
              f"or use `add_paper.py {match['id']} -c <cat>` directly.", file=sys.stderr)
        return False

    if not category:
        try:
            from llm_triage import classify
            record = enrich([match["id"]], use_s2=False)
            if not record:
                print("✗ could not fetch metadata for classification", file=sys.stderr)
                return False
            category = classify(record[0])
            print(f"→ auto-classified as: {category}", file=sys.stderr)
        except SystemExit:  # no LLM key configured
            print("✗ no LLM key for auto-classify; pass -c <category> "
                  "(set DEEPSEEK_API_KEY to enable auto-classification).", file=sys.stderr)
            return False

    cmd = ["python3", str(HERE / "add_paper.py"), match["id"], "-c", category]
    for t in tags:
        cmd += ["-t", t]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("titles", nargs="+", help="one or more paper titles")
    p.add_argument("-c", "--category", help="set the category (else auto-classify)")
    p.add_argument("-t", "--tag", action="append", default=[], dest="tags")
    p.add_argument("--yes", action="store_true", help="accept a low-similarity match")
    p.add_argument("--no-readme", action="store_true", help="don't regenerate README")
    args = p.parse_args()

    added = sum(add_one(t, args.category, args.tags, args.yes) for t in args.titles)
    print(f"\nadded {added}/{len(args.titles)} paper(s).", file=sys.stderr)
    if added and not args.no_readme:
        subprocess.run(["python3", str(HERE / "generate_readme.py")])


if __name__ == "__main__":
    main()
