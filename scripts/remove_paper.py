#!/usr/bin/env python3
"""Remove a paper from data/papers.yaml by arXiv id or title — the inverse of add_paper.py.

Finds the matching entry (by arXiv id, or a case-insensitive title substring), deletes just
that entry (comment-preserving text surgery — section headers and neighbours are untouched),
and regenerates README.md.

    python scripts/remove_paper.py 2304.11015                 # by arXiv id / url
    python scripts/remove_paper.py "MAC-SQL"                  # by title substring
    python scripts/remove_paper.py "MAC-SQL" --dry-run        # preview, don't write
    python scripts/remove_paper.py 2304.11015 --no-readme     # skip README regen

Refuses to act when a title query matches more than one paper (unless --all) so you never
delete the wrong entry. arXiv ids are unique, so they always target exactly one.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE.parent / "data" / "papers.yaml"
TITLE_RE = re.compile(r'^- title:\s*"?(.*?)"?\s*$')


def find_entries(lines, query):
    """Yield (start, end, title) for every entry whose id/title matches `query`."""
    q = query.lower().strip()
    aid = re.search(r"(\d{4}\.\d{4,5})", query)
    aid = aid.group(1) if aid else None
    starts = [i for i, l in enumerate(lines) if l.startswith("- title:")]
    for k, i in enumerate(starts):
        # entry body = the `- title:` line plus following indented field lines
        j = i + 1
        while j < len(lines) and (lines[j].startswith((" ", "\t"))):
            j += 1
        block = "".join(lines[i:j])
        title = TITLE_RE.match(lines[i]).group(1) if TITLE_RE.match(lines[i]) else lines[i]
        hit = (aid and aid in block) or (not aid and q in title.lower())
        if hit:
            end = j
            if end < len(lines) and lines[end].strip() == "":  # eat one trailing blank
                end += 1
            yield i, end, title


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help="arXiv id / url, or a title substring")
    p.add_argument("--all", action="store_true", help="remove ALL matches (title queries)")
    p.add_argument("--dry-run", action="store_true", help="show what would be removed only")
    p.add_argument("--no-readme", action="store_true", help="don't regenerate README")
    args = p.parse_args()

    lines = PAPERS.read_text().splitlines(keepends=True)
    matches = list(find_entries(lines, args.query))
    if not matches:
        sys.exit(f"no paper matching {args.query!r} in {PAPERS.name}")
    if len(matches) > 1 and not args.all:
        print(f"{len(matches)} papers match {args.query!r} — narrow it down, "
              f"or pass --all:", file=sys.stderr)
        for _, _, t in matches:
            print(f"  - {t}", file=sys.stderr)
        sys.exit(1)

    for _, _, t in matches:
        print(("would remove: " if args.dry_run else "removing: ") + t)
    if args.dry_run:
        return

    for start, end, _ in sorted(matches, reverse=True):  # delete bottom-up to keep indices valid
        del lines[start:end]
    PAPERS.write_text("".join(lines))
    print(f"removed {len(matches)} paper(s) from {PAPERS.name}.")

    if not args.no_readme:
        subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)


if __name__ == "__main__":
    main()
