#!/usr/bin/env python3
"""Scan a whole conference/venue for papers relevant to this reading list.

arXiv only knows topic + date, not *acceptances*, so the weekly/backfill scanner
(curate.py) can't answer "show me every ICDE'26 paper". This pulls a venue's full
paper list from a venue-aware source, keeps only the data-agent-relevant ones (a cheap
title keyword prefilter, then the same LLM rubric as curate.py), resolves each keeper's
title to an arXiv id, and prints a digest with ready-to-run `add_paper.py` commands.

Sources:
  --source dblp        DBLP venue listing — best for DB venues (SIGMOD/VLDB/ICDE/KDD/...)
                       once the proceedings are indexed.
  --source openreview  OpenReview — best for ML/NLP venues (NeurIPS/ICLR/ACL/EMNLP/...),
                       accepted papers only.
  --source auto        (default) pick by venue: DB venues -> dblp, ML/NLP -> openreview.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/scan_venue.py --venue ICDE --year 2026
    python scripts/scan_venue.py --venue ICLR --year 2026 --source openreview
    python scripts/scan_venue.py --venue VLDB --year 2025 --no-triage   # relevance list only

Exit code 1 when nothing relevant is found (so a workflow can skip opening an issue).
Runs best in CI (this repo's box hits DBLP SSL / OpenReview 403 from CN).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arxiv_scan  # noqa: E402  (known_ids + topic vocabulary)
from add_by_title import resolve as resolve_title  # noqa: E402

ROOT = HERE.parent

# Venues we know how to route in --source auto (uppercase acronym -> best backend).
DB_VENUES = {"SIGMOD", "VLDB", "PVLDB", "ICDE", "PODS", "CIDR", "EDBT", "KDD",
             "CIKM", "SIGIR", "WSDM", "ICDM", "WWW"}
ML_VENUES = {"NEURIPS", "NIPS", "ICLR", "ICML", "AAAI", "IJCAI", "ACL", "EMNLP",
             "NAACL", "COLING", "COLM", "TACL"}

# Broad title-level relevance vocabulary (cheap prefilter before the LLM). Generous on
# purpose — the LLM makes the real call; this only trims obviously-unrelated papers.
TOPIC_TERMS = [
    "data agent", "data science agent", "data analysis agent", "analytics agent",
    "text-to-sql", "text to sql", "nl2sql", "natural language to sql", "sql generation",
    "table", "tabular", "spreadsheet", "semi-structured",
    "data preparation", "data cleaning", "data wrangling", "entity matching",
    "data imputation", "data integration", "data lake",
    "insight discovery", "business intelligence", "exploratory data analysis",
    "automl", "kaggle", "machine learning agent", "data science",
    "database", "query optimization", "database tuning", "database diagnosis",
    "agent memory", "context engineering", "retrieval-augmented", "multi-agent",
    "llm agent", "language model agent", "autonomous agent", "workflow",
]


def _get(url, data=None, headers=None, retries=4, timeout=40):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers=headers or {"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))


def is_relevant_title(title):
    t = title.lower()
    return any(term in t for term in TOPIC_TERMS)


# --------------------------------------------------------------------------- DBLP
def from_dblp(venue, year, max_pages=10):
    """All proceedings papers for venue+year via DBLP publ search (skips CoRR/informal)."""
    papers, first, page = {}, 0, 200
    for _ in range(max_pages):
        q = f"venue:{venue}: year:{year}: type:Conference_and_Workshop_Papers:"
        url = (f"https://dblp.org/search/publ/api?format=json&h={page}&f={first}&q="
               + urllib.parse.quote(q))
        try:
            hits = json.loads(_get(url))["result"]["hits"]
        except Exception as exc:
            print(f"  dblp page f={first} failed: {exc}", file=sys.stderr)
            break
        rows = hits.get("hit", [])
        for h in rows:
            info = h.get("info", {})
            v = (info.get("venue") or "")
            v = v if isinstance(v, str) else " ".join(v)
            if str(info.get("year")) != str(year):
                continue
            if venue.upper() not in v.upper():
                continue  # DBLP sometimes returns adjacent venues
            title = re.sub(r"\s+", " ", info.get("title", "")).strip().rstrip(".")
            if title:
                papers[title.lower()] = {"title": title, "abstract": "",
                                         "venue": v, "year": year,
                                         "url": info.get("ee") or info.get("url")}
        if len(rows) < page:
            break
        first += page
        time.sleep(2)
    return list(papers.values())


# --------------------------------------------------------------------------- OpenReview
_ACCEPT_HINT = re.compile(r"oral|spotlight|poster|accept|proceed|main|findings", re.I)
_REJECT_HINT = re.compile(r"reject|withdraw|desk|submitted|under review", re.I)


def from_openreview(venue, year, max_pages=20):
    """Accepted papers for a venue+year from OpenReview API v2 (best-effort)."""
    papers, offset, page = {}, 0, 200
    query = f"{venue} {year}"
    for _ in range(max_pages):
        url = ("https://api2.openreview.net/notes/search?"
               + urllib.parse.urlencode({"term": query, "content": "all",
                                         "limit": page, "offset": offset}))
        try:
            notes = json.loads(_get(url)).get("notes", [])
        except Exception as exc:
            print(f"  openreview offset={offset} failed: {exc}", file=sys.stderr)
            break
        if not notes:
            break
        for n in notes:
            c = n.get("content", {})
            def val(k):  # v2 wraps values as {"value": ...}
                x = c.get(k)
                return x.get("value") if isinstance(x, dict) else x
            vfield = str(val("venue") or "")
            if str(year) not in vfield or venue.upper() not in vfield.upper():
                continue
            if _REJECT_HINT.search(vfield) and not _ACCEPT_HINT.search(vfield):
                continue  # skip rejected/withdrawn/submitted
            title = re.sub(r"\s+", " ", str(val("title") or "")).strip()
            if title:
                papers[title.lower()] = {
                    "title": title, "abstract": str(val("abstract") or "")[:1500],
                    "venue": vfield, "year": year,
                    "url": f"https://openreview.net/forum?id={n.get('id')}"}
        offset += page
        time.sleep(1)
    return list(papers.values())


def pick_source(source, venue):
    if source != "auto":
        return source
    return "openreview" if venue.upper() in ML_VENUES else "dblp"


# --------------------------------------------------------------------------- digest
PAPER_CATS = {"data-preparation", "nl2sql", "table-reasoning", "table-curation",
              "data-analysis", "data-science", "db-operations", "memory", "foundations"}


def digest(keepers, venue, year, source):
    """keepers: list of dicts with title, category, decision, reason, arxiv, url."""
    n = len(keepers)
    lines = [f"Venue scan: **{venue} {year}** (via {source}) — "
             f"**{n}** relevant paper(s) after filtering.",
             "",
             "Papers with an arXiv match have a ready `add_paper.py` command (the venue "
             "is grounded automatically). Papers without one are linked to their venue "
             "page — add those by hand or via the issue form.",
             ""]
    from curate import CATEGORY_TITLES
    by_cat = {}
    for k in keepers:
        by_cat.setdefault(k.get("category") or "", []).append(k)
    for cat in CATEGORY_TITLES:
        group = by_cat.get(cat)
        if not group:
            continue
        lines.append(f"## {CATEGORY_TITLES[cat]}\n")
        for k in group:
            reason = f" — {k['reason']}" if k.get("reason") else ""
            if k.get("arxiv"):
                lines.append(f"- [ ] **[{k['title']}](https://arxiv.org/abs/{k['arxiv']})**{reason}")
                add_cat = cat if cat in PAPER_CATS else "TODO"
                lines.append(f"      <sub>`python scripts/add_paper.py {k['arxiv']} "
                             f"-c {add_cat} --venue \"{venue}'{str(year)[2:]}\"`</sub>")
            else:
                url = k.get("url") or ""
                lines.append(f"- [ ] **[{k['title']}]({url})**{reason} "
                             f"<sub>(no arXiv match — add by hand)</sub>")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venue", required=True, help="venue acronym, e.g. ICDE, VLDB, ICLR")
    p.add_argument("--year", required=True, type=int, help="year, e.g. 2026")
    p.add_argument("--source", choices=["auto", "dblp", "openreview"], default="auto")
    p.add_argument("--no-triage", action="store_true",
                   help="skip the LLM; list everything that passes the keyword prefilter")
    p.add_argument("--out", help="write digest here instead of stdout")
    args = p.parse_args()

    source = pick_source(args.source, args.venue)
    print(f"scanning {args.venue} {args.year} via {source} …", file=sys.stderr)
    fetch = from_dblp if source == "dblp" else from_openreview
    papers = fetch(args.venue, args.year)
    print(f"  {len(papers)} papers in venue listing", file=sys.stderr)

    known = arxiv_scan.known_ids()
    candidates = [pp for pp in papers if is_relevant_title(pp["title"])]
    print(f"  {len(candidates)} pass the topic prefilter", file=sys.stderr)
    if not candidates:
        print("nothing relevant found", file=sys.stderr)
        return 1

    # relevance + category via the shared rubric (unless --no-triage)
    if args.no_triage:
        keepers = [{**c, "category": "", "decision": "maybe", "reason": ""}
                   for c in candidates]
    else:
        from llm_triage import triage
        records = [{"id": f"v{i}", "title": c["title"], "abstract": c.get("abstract", ""),
                    "authors": [], "affils": [], "comment": "", "published": str(args.year),
                    "citations": None} for i, c in enumerate(candidates)]
        verdicts = triage(records)
        keepers = []
        for c, v in zip(candidates, verdicts):
            if v["decision"] in ("keep", "maybe"):
                keepers.append({**c, "category": v.get("category", ""),
                                "decision": v["decision"], "reason": v.get("reason", "")})
    print(f"  {len(keepers)} kept after triage", file=sys.stderr)

    # resolve each keeper's title -> arXiv id (skip ones already in the list)
    for k in keepers:
        try:
            m = resolve_title(k["title"])
        except Exception:
            m = None
        if m and m["score"] >= 0.82 and m["id"] not in known:
            k["arxiv"] = m["id"]
        elif m and m["id"] in known:
            k["arxiv"], k["reason"] = m["id"], (k.get("reason", "") + " [already in list]").strip()
        time.sleep(1)

    text = digest(keepers, args.venue, args.year, source)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote digest to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
