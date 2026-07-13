#!/usr/bin/env python3
"""Ground a paper's publication venue from multiple sources.

`add_paper.py` used to hard-code every new paper as `arXiv'YY.MM`, so freshly
*accepted* papers (whose acceptance is known but not yet on arXiv-as-preprint) lost
their real venue. This module grounds the venue from several channels, first hit wins:

  1. arXiv `comment` field         e.g. "Accepted to KDD 2026"      — keyless
  2. Semantic Scholar publicationVenue / venue                      — keyless (429-prone)
  3. DBLP title search (ignores CoRR — that record *is* the arXiv preprint)  — keyless
  4. LLM + web search: Tavily/Serper fetch snippets, DeepSeek extracts the venue
     — needs a search key (TAVILY_API_KEY or SERPER_API_KEY) + an LLM key
       (DEEPSEEK_API_KEY / LLM_API_KEY). This is the channel that catches brand-new
       acceptances (e.g. LEAF-SQL = ICDE'26) that no structured API knows yet.

All channels return a normalized `ACRONYM'YY` string (repo style, e.g. "KDD'26",
"VLDB'25") or None. If nothing grounds a real (non-preprint) venue, resolve_venue
returns None and the caller falls back to arXiv'YY.MM with a TODO marker.

CLI (handy for spot-checking):
    python scripts/venue.py 2602.16720                 # by arXiv id
    python scripts/venue.py --title "LEAF-SQL: ..."    # title only
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# Canonical short forms we render in the list. Keys are UPPERCASE match tokens;
# values are the exact casing to emit. Longer/aliased names map onto the same value.
CANON = {
    # databases
    "SIGMOD": "SIGMOD", "VLDB": "VLDB", "PVLDB": "VLDB", "ICDE": "ICDE", "PODS": "PODS",
    "CIDR": "CIDR", "EDBT": "EDBT", "TKDE": "TKDE", "TODS": "TODS", "DASFAA": "DASFAA",
    # ML / AI
    "NEURIPS": "NeurIPS", "NIPS": "NeurIPS", "ICML": "ICML", "ICLR": "ICLR",
    "AAAI": "AAAI", "IJCAI": "IJCAI", "COLM": "COLM", "TMLR": "TMLR", "JMLR": "JMLR",
    "AISTATS": "AISTATS", "UAI": "UAI",
    # NLP
    "ACL": "ACL", "EMNLP": "EMNLP", "NAACL": "NAACL", "COLING": "COLING",
    "EACL": "EACL", "TACL": "TACL", "FINDINGS": "ACL-Findings",
    # data mining / IR / web
    "KDD": "KDD", "SIGIR": "SIGIR", "WWW": "WWW", "WSDM": "WSDM", "CIKM": "CIKM",
    "ICDM": "ICDM", "RECSYS": "RecSys",
    # systems
    "OSDI": "OSDI", "SOSP": "SOSP", "NSDI": "NSDI", "ATC": "ATC", "EUROSYS": "EuroSys",
    # vision (rare here but harmless)
    "CVPR": "CVPR", "ICCV": "ICCV", "ECCV": "ECCV",
}
# multi-word aliases collapsed before token matching
ALIASES = [
    (r"the web conference|world wide web", "WWW"),
    (r"neural information processing systems", "NeurIPS"),
    (r"very large data ?bases", "VLDB"),
    (r"knowledge discovery and data mining", "KDD"),
    (r"empirical methods in natural language processing", "EMNLP"),
    (r"association for computational linguistics", "ACL"),
    (r"data engineering", "ICDE"),  # "International Conference on Data Engineering"
]
# venues we explicitly treat as "no real venue" (preprint servers)
PREPRINT = {"ARXIV", "CORR", "PREPRINT"}


def _year2(raw):
    """Extract a 2-digit year from a string ('2026'->'26', "'26"->'26')."""
    m = re.search(r"\b(20\d{2})\b", raw)
    if m:
        return m.group(1)[2:]
    m = re.search(r"'(\d{2})\b", raw)
    return m.group(1) if m else None


def normalize_venue(raw, default_year=None):
    """Map a free-text venue string to 'ACRONYM'YY', or None if unrecognized/preprint."""
    if not raw:
        return None
    s = raw.strip()
    up = s.upper()
    if up in PREPRINT or "ARXIV" in up or up.startswith("CORR"):
        return None
    lo = s.lower()
    for pat, canon in ALIASES:
        if re.search(pat, lo):
            yy = _year2(s) or default_year
            return f"{canon}'{yy}" if yy else canon
    # token match on acronyms (word-boundary, so 'ACL' won't hit 'oracle')
    for token in re.findall(r"[A-Za-z][A-Za-z\-]+", s):
        canon = CANON.get(token.upper())
        if canon:
            yy = _year2(s) or default_year
            return f"{canon}'{yy}" if yy else canon
    return None


# --------------------------------------------------------------------------- channels
def fetch(url, data=None, headers=None, retries=4, timeout=30):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _from_comment(comment, default_year=None):
    """arXiv comment often says 'Accepted to/at/by <VENUE> <YEAR>' — or just '<VENUE> <YEAR>'."""
    if not comment:
        return None
    # Prefer an explicit acceptance clause; fall back to scanning the whole comment.
    m = re.search(r"(?:accepted|to appear|camera[- ]ready|published)\b.{0,80}", comment, re.I)
    span = m.group(0) if m else comment
    return normalize_venue(span, default_year) or normalize_venue(comment, default_year)


def _from_s2(aid, default_year=None):
    """Semantic Scholar publicationVenue — keyless but frequently 429s; treat failure as miss."""
    if not aid:
        return None
    url = (f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}"
           "?fields=venue,publicationVenue,year")
    try:
        d = json.loads(fetch(url, retries=2, timeout=20))
    except Exception:
        return None
    yy = str(d.get("year"))[2:] if d.get("year") else default_year
    for cand in [(d.get("publicationVenue") or {}).get("name"), d.get("venue")]:
        v = normalize_venue(cand, yy)
        if v:
            return v
    return None


def _from_dblp(title, default_year=None):
    """DBLP title search. A 'CoRR' hit is just the arXiv preprint, so it's ignored."""
    if not title:
        return None
    url = ("https://dblp.org/search/publ/api?format=json&h=5&q="
           + urllib.parse.quote(title))
    try:
        hits = json.loads(fetch(url, headers={"User-Agent": "Mozilla/5.0"},
                                retries=2, timeout=20))["result"]["hits"].get("hit", [])
    except Exception:
        return None
    t_lo = title.lower()[:40]
    for h in hits:
        info = h.get("info", {})
        if (info.get("title", "").lower()[:40]) != t_lo:
            continue  # different paper
        v = normalize_venue(info.get("venue", ""), str(info.get("year", ""))[2:] or default_year)
        if v:  # normalize_venue already drops CoRR
            return v
    return None


def _web_search(query):
    """Return a blob of web snippets for the query, using whichever search key is set."""
    tav = os.environ.get("TAVILY_API_KEY")
    ser = os.environ.get("SERPER_API_KEY")
    try:
        if tav:
            body = json.dumps({"api_key": tav, "query": query, "max_results": 6,
                               "include_answer": False}).encode()
            data = json.loads(fetch("https://api.tavily.com/search", data=body,
                                    headers={"Content-Type": "application/json"},
                                    retries=2, timeout=30))
            return "\n".join(f"{r.get('title','')} — {r.get('content','')}"
                             for r in data.get("results", []))
        if ser:
            body = json.dumps({"q": query, "num": 6}).encode()
            data = json.loads(fetch("https://google.serper.dev/search", data=body,
                                    headers={"Content-Type": "application/json",
                                             "X-API-KEY": ser}, retries=2, timeout=30))
            return "\n".join(f"{r.get('title','')} — {r.get('snippet','')}"
                             for r in data.get("organic", []))
    except Exception as exc:
        print(f"  venue web-search failed: {exc}", file=sys.stderr)
    return None


def _from_web_llm(title, default_year=None):
    """Search the web for the acceptance venue and have the LLM extract it (grounded)."""
    if not title:
        return None
    if not (os.environ.get("TAVILY_API_KEY") or os.environ.get("SERPER_API_KEY")):
        return None  # no search provider configured
    snippets = _web_search(f'"{title}" paper accepted venue conference')
    if not snippets:
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llm_triage import _call, load_config
        cfg = load_config()  # raises SystemExit if no LLM key
    except SystemExit:
        return None
    except Exception:
        return None
    prompt = (
        "From the web search results below, determine the PEER-REVIEWED venue and year "
        f'the paper titled "{title}" was ACCEPTED to (e.g. SIGMOD, VLDB, ICDE, NeurIPS, '
        "ICLR, ACL, KDD, ...). Only count a real acceptance stated in the results — NOT "
        '"submitted", "under review", or the arXiv/CoRR preprint. '
        "Reply with ONLY the venue and year like `KDD 2026`, or exactly `NONE` if the "
        "results don't clearly state an acceptance.\n\n=== RESULTS ===\n" + snippets[:4000])
    try:
        ans = _call([{"role": "user", "content": prompt}], cfg).strip()
    except Exception:
        return None
    if "NONE" in ans.upper():
        return None
    return normalize_venue(ans, default_year)


# --------------------------------------------------------------------------- entry point
def resolve_venue(meta, use_web=True):
    """Best real venue for a paper, or None. `meta` needs id/title/comment/published.

    Returns (venue_str, source) — source is one of comment|s2|dblp|web, or (None, None).
    """
    aid = meta.get("id")
    title = meta.get("title")
    comment = meta.get("comment")
    yy = (meta.get("published") or "")[2:4] or None  # 'YYYY-..' -> 'YY'

    for name, fn in (("comment", lambda: _from_comment(comment, yy)),
                     ("s2", lambda: _from_s2(aid, yy)),
                     ("dblp", lambda: _from_dblp(title, yy))):
        v = fn()
        if v:
            return v, name
    if use_web:
        v = _from_web_llm(title, yy)
        if v:
            return v, "web"
    return None, None


def _arxiv_meta(aid):
    url = f"https://export.arxiv.org/api/query?id_list={aid}"
    e = ET.fromstring(fetch(url)).find(f"{ATOM}entry")
    if e is None:
        return None
    c = e.find(f"{ARXIV_NS}comment")
    return {"id": aid,
            "title": re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip(),
            "comment": re.sub(r"\s+", " ", c.text).strip() if c is not None else "",
            "published": e.find(f"{ATOM}published").text[:10]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arxiv", nargs="?", help="arXiv id or url")
    p.add_argument("--title", help="resolve by title only (no arXiv metadata)")
    p.add_argument("--no-web", action="store_true", help="skip the web+LLM channel")
    args = p.parse_args()

    if args.arxiv:
        m = re.search(r"(\d{4}\.\d{4,5})", args.arxiv)
        meta = _arxiv_meta(m.group(1)) if m else None
        if not meta:
            sys.exit("could not fetch arXiv metadata")
    elif args.title:
        meta = {"title": args.title}
    else:
        p.error("give an arXiv id/url or --title")

    v, src = resolve_venue(meta, use_web=not args.no_web)
    print(f"{v or '(none)'}   [source: {src or 'unresolved'}]")


if __name__ == "__main__":
    main()
