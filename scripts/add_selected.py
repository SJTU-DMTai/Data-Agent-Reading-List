#!/usr/bin/env python3
"""Batch-add the checked papers from a digest issue (venue scan OR weekly digest).

Both digests render each candidate as a task-list checkbox followed by a `<sub>` line
holding the exact command that would add it, e.g.

    - [x] **[APEX-SQL: ...](https://arxiv.org/abs/2602.16720)** — reason
          <sub>`python scripts/add_paper.py 2602.16720 -c nl2sql --venue "KDD'26"`</sub>

The user ticks the boxes they want and applies the `add-selected` label; the workflow
(.github/workflows/add-selected.yml) runs this. For every *checked* entry that isn't
already marked done, we parse its command (arXiv id + category + venue, or a benchmark
group), add it via the same insert_paper / insert_benchmark used by the issue form, then
rewrite that line with a ✅ / ⚠️ / ❌ marker so re-running skips it.

Outputs to $GITHUB_OUTPUT: status=added|noop|error + a summary the workflow comments back.
The rewritten issue body is written to $UPDATED_BODY (default updated_body.md) for the
workflow to push back with `gh issue edit --body-file`.

Local dry-run:
    ISSUE_BODY="$(cat digest.md)" python scripts/add_selected.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_paper import insert_paper  # noqa: E402
from add_from_issue import set_output, classify_category  # noqa: E402
from add_benchmark import insert_benchmark, DEFAULT_GROUP  # noqa: E402
from scan_venue import PAPER_CATS  # noqa: E402

CHECKBOX_RE = re.compile(r"^(\s*[-*]\s*)\[([ xX])\]\s*(.*)$")
CMD_RE = re.compile(r"add_(paper|benchmark)\.py\s+(\d{4}\.\d{4,5})(.*)")
TITLE_RE = re.compile(r"\*\*\[(.*?)\]")
DONE_MARKS = ("✅", "⚠️", "❌")


def _flag(rest, *names):
    """Pull a `--flag value` / `--flag "quoted value"` out of a command tail."""
    for n in names:
        m = re.search(rf'{re.escape(n)}\s+"([^"]+)"', rest) or \
            re.search(rf"{re.escape(n)}\s+(\S+)", rest)
        if m:
            return m.group(1)
    return None


def parse_selected(lines):
    """Yield (line_index, kind, aid, title, category, venue, group) for each checked,
    not-yet-done entry that has an add_* command. line_index points at its checkbox line."""
    for i, line in enumerate(lines):
        m = CMD_RE.search(line)
        if not m:
            continue
        # walk back to the checkbox this command belongs to
        j = i
        while j >= 0 and not CHECKBOX_RE.match(lines[j]):
            j -= 1
        if j < 0:
            continue
        cb = CHECKBOX_RE.match(lines[j])
        if cb.group(2) == " ":
            continue  # unchecked
        if any(mark in lines[j] for mark in DONE_MARKS):
            continue  # already processed on a previous run
        kind, aid, rest = m.group(1), m.group(2), m.group(3)
        tm = TITLE_RE.search(cb.group(3))
        title = tm.group(1) if tm else aid
        yield {
            "line": j, "kind": kind, "aid": aid, "title": title,
            "category": _flag(rest, "-c", "--category"),
            "venue": _flag(rest, "--venue"),
            "group": _flag(rest, "-g", "--group") or DEFAULT_GROUP,
        }


def _add_one(e):
    """Add a single entry. Returns (marker, note) where marker is ✅ / ⚠️ / ❌."""
    if e["kind"] == "benchmark":
        try:
            b = insert_benchmark(e["aid"], group=e["group"])
            return "✅", f"added benchmark to *{b['group']}*"
        except ValueError as exc:
            return ("⚠️", "already in list") if "already" in str(exc) else ("❌", str(exc))

    category = e["category"]
    if not category or category == "TODO" or category not in PAPER_CATS:
        try:
            category = classify_category(e["aid"])
        except SystemExit:
            return "❌", "no category and auto-classify needs DEEPSEEK_API_KEY — edit the `-c` value and re-tick"
    try:
        res = insert_paper(e["aid"], category, venue=e["venue"] or None)
    except ValueError as exc:
        return ("⚠️", "already in list") if "already exists" in str(exc) else ("❌", str(exc))
    v = res["venue"] + ("  ⚠️ verify venue" if res["venue_todo"] else "")
    return "✅", f"`{category}` · {v}"


def main():
    body = os.environ.get("ISSUE_BODY", "")
    if "--body-file" in sys.argv:
        body = Path(sys.argv[sys.argv.index("--body-file") + 1]).read_text()
    if not body.strip():
        set_output("error", "empty issue body — nothing to add.")
        return 1

    lines = body.splitlines()
    todo = list(parse_selected(lines))
    if not todo:
        set_output("noop", "No newly-checked papers to add. Tick the boxes you want, keep "
                           "the `add-selected` label, and I'll add them.")
        return 0

    added, dup, failed = [], [], []
    for e in todo:
        marker, note = _add_one(e)
        lines[e["line"]] = lines[e["line"]].rstrip() + f"  — {marker} {note}"
        entry = f"**{e['title']}** — {note}"
        (added if marker == "✅" else dup if marker == "⚠️" else failed).append(entry)

    if added:
        subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)

    # write the marked-up body back for the workflow to push to the issue
    Path(os.environ.get("UPDATED_BODY", "updated_body.md")).write_text("\n".join(lines) + "\n")

    parts = []
    if added:
        parts.append(f"### ✅ Added {len(added)}\n" + "\n".join(f"- {x}" for x in added))
    if dup:
        parts.append(f"### ⚠️ Already in list ({len(dup)})\n" + "\n".join(f"- {x}" for x in dup))
    if failed:
        parts.append(f"### ❌ Failed ({len(failed)})\n" + "\n".join(f"- {x}" for x in failed))
    parts.append("_Re-tick more boxes and re-apply the `add-selected` label to add the rest; "
                 "already-processed lines are skipped._")
    set_output("added" if added else "noop", "\n\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
