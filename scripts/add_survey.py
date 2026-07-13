#!/usr/bin/env python3
"""Add a survey / vision / position paper to data/surveys.yaml from an arXiv link.

surveys.yaml is a flat list of entries {title, venue, year, paper, corresponding?, note?}.
This is the survey analogue of add_paper.py: fetch arXiv metadata, ground the venue (most
surveys are preprints, so it falls back to arXiv'YY.MM), and append an entry — preserving
the file's comments by writing text (no full YAML re-dump).

Usage:
    python scripts/add_survey.py 2510.23587
    python scripts/add_survey.py 2507.01599 --corresponding "Guoliang Li" --note "vision paper"
    python scripts/add_survey.py 2510.23587 --venue "SIGMOD'26"     # override venue
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_paper import fetch, arxiv_id  # noqa: E402
from venue import resolve_venue, normalize_venue  # noqa: E402

SURVEYS = HERE.parent / "data" / "surveys.yaml"


def insert_survey(aid, corresponding=None, note=None, venue=None, use_web=True):
    """Append one survey to data/surveys.yaml. Returns a summary dict."""
    if aid in SURVEYS.read_text():
        raise ValueError(f"arXiv id {aid} already exists in {SURVEYS.name}")

    meta = fetch(aid)
    year, month = meta["published"][:4], meta["published"][5:7]
    fallback = f"arXiv'{year[2:]}.{month}"

    venue_todo = False
    venue_source = "manual"
    if venue:
        venue = normalize_venue(venue, year[2:]) or venue
    else:
        venue, venue_source = resolve_venue(meta, use_web=use_web)
        if not venue:                       # surveys are usually preprints — this is normal
            venue, venue_source, venue_todo = fallback, "arxiv-fallback", False

    lines = [
        "",
        f'- title: "{meta["title"]}"',
        f"  venue: {venue}" + ("  # TODO verify venue" if venue_todo else ""),
        f"  year: {year}",
    ]
    if corresponding:
        lines.append(f"  corresponding: {corresponding}")
    lines.append(f"  paper: https://arxiv.org/pdf/{aid}")
    if note:
        lines.append(f"  note: {note}")

    with SURVEYS.open("a") as f:
        f.write("\n".join(lines) + "\n")

    return {"id": aid, "title": meta["title"], "venue": venue,
            "venue_source": venue_source, "venue_todo": venue_todo,
            "corresponding": corresponding, "yaml": "\n".join(lines)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arxiv", help="arXiv id or url")
    p.add_argument("--corresponding", help="corresponding author (optional)")
    p.add_argument("--note", help="short note, e.g. 'vision paper' (optional)")
    p.add_argument("--venue", help="force the venue (else grounded; falls back to arXiv'YY.MM)")
    p.add_argument("--no-web", action="store_true", help="skip the web+LLM venue channel")
    args = p.parse_args()

    try:
        res = insert_survey(arxiv_id(args.arxiv), corresponding=args.corresponding,
                            note=args.note, venue=args.venue, use_web=not args.no_web)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    print(res["yaml"])
    print(f"\nvenue: {res['venue']} [via {res['venue_source']}]")
    print(f"Appended to {SURVEYS.name}. Now run: python scripts/generate_readme.py")


if __name__ == "__main__":
    main()
