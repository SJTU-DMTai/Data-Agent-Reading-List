#!/usr/bin/env python3
"""Remove a paper from a GitHub "Remove a paper" issue-form submission.

The workflow (.github/workflows/remove-paper.yml) passes the rendered issue body via the
ISSUE_BODY env var. We parse the form, locate the paper (by arXiv id or title substring),
delete just that entry, and regenerate README.md. Results go to $GITHUB_OUTPUT
(status=removed|not_found|ambiguous|error + a summary the workflow comments back).

For safety a title that matches more than one paper is REFUSED (the user is asked to use
the arXiv id) — the issue flow never bulk-deletes.

Local dry-run:
    ISSUE_BODY="$(cat issue.md)" python scripts/remove_from_issue.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_from_issue import parse_form, _get, set_output  # noqa: E402
from remove_paper import (find_entries, find_benchmark_items,  # noqa: E402
                          PAPERS, SURVEYS, BENCH)

# (file, kind label, finder) — searched in order; a hit can be in any of them.
TARGETS = [(PAPERS, "paper", find_entries),
           (SURVEYS, "survey", find_entries),
           (BENCH, "benchmark", find_benchmark_items)]


def main():
    body = os.environ.get("ISSUE_BODY", "")
    if "--body-file" in sys.argv:
        body = Path(sys.argv[sys.argv.index("--body-file") + 1]).read_text()
    if not body.strip():
        set_output("error", "empty issue body — nothing to remove.")
        return 1

    fields = parse_form(body)
    ref = _get(fields, "arxiv id or paper title", "arxiv link or paper title",
               "paper link", "paper")
    if not ref:
        set_output("error", "no arXiv id / title found in the form.")
        return 1

    # search papers, surveys, and benchmark items; collect matches across all files
    file_lines, matches = {}, []
    for path, kind, finder in TARGETS:
        if not path.exists():
            continue
        lines = path.read_text().splitlines(keepends=True)
        file_lines[path] = lines
        for start, end, label in finder(lines, ref):
            matches.append((path, kind, start, end, label))

    if not matches:
        set_output("not_found", f"no paper, survey, or benchmark matching **{ref}** in "
                                "`data/`. Double-check the arXiv id or exact title.")
        return 0
    if len(matches) > 1:
        listed = "\n".join(f"- ({k}) {t}" for _, k, _, _, t in matches)
        set_output("ambiguous",
                   f"**{ref}** matches {len(matches)} entries — please re-open with the "
                   f"exact **arXiv id** so only one is removed:\n{listed}")
        return 0

    path, kind, start, end, label = matches[0]
    lines = file_lines[path]
    del lines[start:end]
    path.write_text("".join(lines))
    subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)
    set_output("removed", f"🗑️ Removed {kind} **{label}** (from `data/{path.name}`) and "
                          "regenerated the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
