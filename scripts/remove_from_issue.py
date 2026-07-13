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
from remove_paper import find_entries, PAPERS  # noqa: E402


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

    lines = PAPERS.read_text().splitlines(keepends=True)
    matches = list(find_entries(lines, ref))
    if not matches:
        set_output("not_found", f"no paper matching **{ref}** in `data/papers.yaml`. "
                                "(Only papers.yaml is supported — benchmarks/surveys are "
                                "still removed by hand.)")
        return 0
    if len(matches) > 1:
        listed = "\n".join(f"- {t}" for _, _, t in matches)
        set_output("ambiguous",
                   f"**{ref}** matches {len(matches)} papers — please re-open with the "
                   f"exact **arXiv id** so only one is removed:\n{listed}")
        return 0

    start, end, title = matches[0]
    del lines[start:end]
    PAPERS.write_text("".join(lines))
    subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)
    set_output("removed", f"🗑️ Removed **{title}** from the list and regenerated the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
