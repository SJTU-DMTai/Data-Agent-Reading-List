#!/usr/bin/env python3
"""Re-ground the venue of existing papers whose venue is still a bare `arXiv'YY.MM`.

Many entries were added before venue grounding existed, so a paper that has since been
accepted (or always had its acceptance in the arXiv comment) still shows as arXiv. This
re-checks each such entry through scripts/venue.py and, when a real venue is found,
rewrites just that `venue:` line — comments and formatting elsewhere are untouched.

    python scripts/backfill_venues.py            # dry-run: show what would change
    python scripts/backfill_venues.py --apply    # write the changes
    python scripts/backfill_venues.py --all      # reconsider every entry, not just arXiv'
    python scripts/backfill_venues.py --no-web    # keyless channels only

Set DEEPSEEK_API_KEY + TAVILY_API_KEY (or SERPER_API_KEY) to enable the web channel that
catches brand-new acceptances no structured API knows yet.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from venue import resolve_venue, _arxiv_meta  # noqa: E402

PAPERS = Path(__file__).resolve().parent.parent / "data" / "papers.yaml"
ENTRY_RE = re.compile(r"^- title:", re.M)


def split_entries(text):
    """Yield (start, end, block) for each `- title:` entry in the file."""
    starts = [m.start() for m in ENTRY_RE.finditer(text)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        yield s, e, text[s:e]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--all", action="store_true", help="reconsider all entries, not just arXiv'")
    ap.add_argument("--no-web", action="store_true", help="skip the web+LLM venue channel")
    args = ap.parse_args()

    text = PAPERS.read_text()
    out, changes = [], []
    last = 0
    for s, e, block in split_entries(text):
        out.append(text[last:s])
        last = e
        vm = re.search(r"^(\s*venue:\s*)([^\n#]+?)(\s*(?:#.*)?)$", block, re.M)
        pm = re.search(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})", block)
        tm = re.search(r'^- title:\s*"?(.+?)"?\s*$', block, re.M)
        cur_venue = vm.group(2).strip() if vm else ""
        needs = args.all or cur_venue.lower().startswith("arxiv")
        if not (vm and pm and needs):
            out.append(block)
            continue
        aid = pm.group(1)
        meta = _arxiv_meta(aid) or {"id": aid, "title": tm.group(1) if tm else ""}
        time.sleep(3)  # arXiv etiquette
        new_venue, src = resolve_venue(meta, use_web=not args.no_web)
        if new_venue and new_venue != cur_venue:
            # rewrite only the venue line; drop any stale "# TODO verify venue" note
            block = block[:vm.start()] + f"{vm.group(1)}{new_venue}" + block[vm.end():]
            changes.append((tm.group(1)[:60] if tm else aid, cur_venue, new_venue, src))
        out.append(block)
    out.append(text[last:])

    if not changes:
        print("no venues to backfill.")
        return
    print(f"{'APPLIED' if args.apply else 'DRY-RUN'} — {len(changes)} venue(s):\n")
    for title, old, new, src in changes:
        print(f"  {old:14} -> {new:12} [{src}]  {title}")
    if args.apply:
        PAPERS.write_text("".join(out))
        print(f"\nwrote {PAPERS}. Now run: python scripts/generate_readme.py")
    else:
        print("\n(dry-run — re-run with --apply to write)")


if __name__ == "__main__":
    main()
