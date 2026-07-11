#!/usr/bin/env python3
"""Enrich arXiv candidates with metadata used by the curation triage.

For each arXiv id we collect:
  - title, authors, full abstract, submission date         (arXiv API)
  - arXiv `comment` field — often states "Accepted at ..."  (arXiv API)
  - citationCount and author affiliations, where available  (Semantic Scholar)

Usage:
    python scripts/enrich.py 2504.01234 2505.06789          # ids as args
    python scripts/enrich.py --ids-file ids.txt             # one id per line
    echo 2504.01234 | python scripts/enrich.py --stdin
    # writes JSONL to stdout, or to --out FILE

Importable:
    from enrich import enrich
    records = enrich(["2504.01234", ...])
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def fetch(url, data=None, headers=None, retries=6, timeout=45):
    """GET/POST with exponential backoff — arXiv and S2 both flake on SSL EOF."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))


def _arxiv_meta(ids):
    """arXiv id_list batch (<=100 per call): title/authors/abstract/comment/date."""
    rec = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(batch)
               + "&max_results=100")
        root = ET.fromstring(fetch(url))
        for e in root.findall(f"{ATOM}entry"):
            m = re.search(r"(\d{4}\.\d{4,5})", e.find(f"{ATOM}id").text)
            if not m:
                continue
            aid = m.group(1)
            comment = e.find(f"{ARXIV_NS}comment")
            summ = e.find(f"{ATOM}summary")
            rec[aid] = {
                "id": aid,
                "title": re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip(),
                "authors": [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")],
                "abstract": re.sub(r"\s+", " ", summ.text).strip() if summ is not None else "",
                "comment": re.sub(r"\s+", " ", comment.text).strip() if comment is not None else "",
                "published": e.find(f"{ATOM}published").text[:10],
                "citations": None,
                "affils": [],
            }
        time.sleep(3)  # arXiv rate-limit etiquette
    return rec


def _s2_enrich(rec):
    """Semantic Scholar batch: add citationCount + author affiliations (best effort)."""
    ids = list(rec)
    for i in range(0, len(ids), 200):
        batch = [f"arXiv:{a}" for a in ids[i:i + 200]]
        url = ("https://api.semanticscholar.org/graph/v1/paper/batch?fields="
               "citationCount,authors.affiliations")
        try:
            resp = json.loads(fetch(url, data=json.dumps({"ids": batch}).encode(),
                                    headers={"Content-Type": "application/json"}))
        except Exception as exc:  # S2 is a bonus signal; never fail the whole run on it
            print(f"warning: Semantic Scholar batch failed: {exc}", file=sys.stderr)
            continue
        for s2id, d in zip(batch, resp):
            aid = s2id.split(":", 1)[1]
            if d and aid in rec:
                rec[aid]["citations"] = d.get("citationCount")
                affs = []
                for au in (d.get("authors") or []):
                    affs += au.get("affiliations") or []
                rec[aid]["affils"] = sorted(set(affs))
        time.sleep(2)
    return rec


def enrich(ids, use_s2=True):
    """Return a list of enriched records, in the input id order."""
    ids = list(dict.fromkeys(ids))  # dedup, keep order
    rec = _arxiv_meta(ids)
    if use_s2:
        _s2_enrich(rec)
    return [rec[a] for a in ids if a in rec]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ids", nargs="*", help="arXiv ids (e.g. 2504.01234)")
    p.add_argument("--ids-file", help="file with one arXiv id per line")
    p.add_argument("--stdin", action="store_true", help="read ids from stdin")
    p.add_argument("--no-s2", action="store_true", help="skip Semantic Scholar enrichment")
    p.add_argument("--out", help="write JSONL here instead of stdout")
    args = p.parse_args()

    ids = list(args.ids)
    if args.ids_file:
        ids += [l.strip() for l in open(args.ids_file) if l.strip()]
    if args.stdin:
        ids += [l.strip() for l in sys.stdin if l.strip()]
    ids = [re.search(r"(\d{4}\.\d{4,5})", x).group(1) for x in ids
           if re.search(r"(\d{4}\.\d{4,5})", x)]
    if not ids:
        p.error("no arXiv ids given")

    records = enrich(ids, use_s2=not args.no_s2)
    out = open(args.out, "w") if args.out else sys.stdout
    for r in records:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.out:
        out.close()
        print(f"wrote {len(records)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
