#!/usr/bin/env python3
"""Harvest candidate papers from OTHER curated lists — the way to backfill the canon.

The weekly scan (curate.py) and venue scan (scan_venue.py) both look *forward*; neither
recovers foundational older papers (DIN-SQL, MAC-SQL, CHESS, ...) that predate our
coverage. Those already live in sibling awesome-lists. This mines one or more such lists:
pull the README, extract every arXiv id, drop the ones we already have, enrich + LLM-triage
the rest against our rubric, and emit a digest with ready `add_paper.py` commands.

A `source` can be:
    owner/name                         a GitHub repo (its default README)
    https://github.com/owner/name      same
    https://…/README.md (raw)          a raw markdown URL
    ./path/to/README.md                a local file

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/harvest_list.py HKUSTDial/NL2SQL_Handbook
    python scripts/harvest_list.py eosphoros-ai/Awesome-Text2SQL SpursGoZmy/Awesome-Tabular-LLMs
    python scripts/harvest_list.py HKUSTDial/NL2SQL_Handbook --no-triage --limit 60

Exit code 1 when nothing new is found. Runs best in CI (LLM key + clean network).
"""
import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arxiv_scan  # noqa: E402
from enrich import enrich  # noqa: E402

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
# a markdown link whose URL contains an arXiv id — capture the link text as a title hint
LINK_RE = re.compile(r"\[([^\]]{4,200})\]\((?:https?://)?(?:www\.)?arxiv\.org/[^)]*?(\d{4}\.\d{4,5})[^)]*\)", re.I)


def _fetch(url, timeout=45, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                       "Accept": "application/vnd.github.raw"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))


def read_source(source):
    """Return the markdown text for a source (repo / url / local path)."""
    if Path(source).exists():
        return Path(source).read_text()
    m = re.match(r"(?:https?://github\.com/)?([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$", source)
    if m and "arxiv.org" not in source:
        owner, name = m.group(1), m.group(2)
        return _fetch(f"https://api.github.com/repos/{owner}/{name}/readme")
    return _fetch(source)  # a raw URL


def extract_candidates(text):
    """Map arXiv id -> best title hint found near it (or '')."""
    cands = {}
    for title, aid in LINK_RE.findall(text):
        t = re.sub(r"\s+", " ", title).strip()
        # keep the longest / most title-like hint per id
        if aid not in cands or len(t) > len(cands[aid]):
            cands[aid] = t
    for aid in ARXIV_RE.findall(text):  # ids not wrapped in a titled link
        cands.setdefault(aid, "")
    return cands


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", help="sibling lists (owner/name, url, or file)")
    p.add_argument("--no-triage", action="store_true",
                   help="skip the LLM; list all fresh candidates untriaged")
    p.add_argument("--limit", type=int, default=120, help="max candidates to enrich/triage")
    p.add_argument("--out", help="write the digest here instead of stdout")
    args = p.parse_args()

    known = arxiv_scan.known_ids()
    found = {}
    for src in args.sources:
        try:
            text = read_source(src)
        except Exception as exc:
            print(f"warning: could not read {src!r}: {exc}", file=sys.stderr)
            continue
        c = extract_candidates(text)
        fresh = {aid: t for aid, t in c.items() if aid not in known}
        print(f"  {src}: {len(c)} arXiv ids, {len(fresh)} not already in our list",
              file=sys.stderr)
        for aid, t in fresh.items():
            if aid not in found or len(t) > len(found[aid]):
                found[aid] = t

    ids = list(found)[:args.limit]
    if not ids:
        print("nothing new found", file=sys.stderr)
        return 1
    if len(found) > args.limit:
        print(f"note: {len(found)} candidates found; enriching the first {args.limit} "
              f"(raise --limit for more)", file=sys.stderr)

    records = enrich(ids)
    print(f"enriched {len(records)}/{len(ids)}", file=sys.stderr)

    if args.no_triage:
        from curate import digest
        verdicts = [{**r, "decision": "maybe", "category": "", "gates": "",
                     "reason": (found.get(r["id"], "") or "")[:80]} for r in records]
    else:
        from llm_triage import triage
        verdicts = triage(records)
    from curate import digest

    span = f"{len(records)} candidates from {len(args.sources)} sibling list(s)"
    text = digest(verdicts, span)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote digest to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    main()
