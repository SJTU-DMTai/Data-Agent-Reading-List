#!/usr/bin/env python3
"""Add a paper from a GitHub "Add a paper" issue-form submission.

The workflow (.github/workflows/add-paper.yml) passes the rendered issue body via the
ISSUE_BODY env var (or --body-file). We parse the form fields, resolve the paper
(arXiv id/url *or* a bare title), classify it if the user chose Auto, ground the venue,
append it to data/papers.yaml, and regenerate README.md.

Results are written to $GITHUB_OUTPUT (status=added|duplicate|error, and a summary
block the workflow posts back as an issue comment).

Local dry-run:
    ISSUE_BODY="$(cat issue.md)" python scripts/add_from_issue.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_paper import insert_paper, arxiv_id  # noqa: E402

# Human dropdown label (issue form) -> papers.yaml category key. None => auto-classify.
CATEGORY_MAP = {
    "auto (let the bot classify)": None,
    "data preparation & integration": "data-preparation",
    "nl2sql (text-to-sql)": "nl2sql",
    "table understanding & reasoning": "table-reasoning",
    "table generation, curation & synthesis": "table-curation",
    "data analysis & insight discovery": "data-analysis",
    "data science & machine learning agents": "data-science",
    "database operations & diagnosis": "db-operations",
    "agent memory & context engineering": "memory",
    "general agent techniques": "foundations",
}
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def parse_form(body: str) -> dict:
    """Extract values keyed by their '### Header' from a rendered issue-form body."""
    fields, key, buf = {}, None, []
    for line in body.splitlines():
        h = re.match(r"^#{2,3}\s+(.*)", line.strip())
        if h:
            if key is not None:
                fields[key] = "\n".join(buf).strip()
            key, buf = h.group(1).strip().lower(), []
        elif key is not None:
            buf.append(line)
    if key is not None:
        fields[key] = "\n".join(buf).strip()
    # normalize GitHub's "no response" placeholder to empty
    return {k: ("" if v.strip().lower() in ("_no response_", "none", "") else v.strip())
            for k, v in fields.items()}


def _get(fields, *names):
    for n in names:
        if n in fields and fields[n]:
            return fields[n]
    return ""


def set_output(status: str, summary: str, title: str = ""):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"status={status}\n")
            if title:
                f.write("title<<__EOF__\n" + title + "\n__EOF__\n")
            f.write("summary<<__EOF__\n" + summary + "\n__EOF__\n")
    print(f"[{status}]\n{summary}")


def resolve_input(paper_ref: str):
    """Return an arXiv id from either an id/url or a bare title (via title resolver)."""
    m = ARXIV_RE.search(paper_ref)
    if m:
        return m.group(1), None
    # treat as a title
    from add_by_title import resolve, ACCEPT_THRESHOLD
    match = resolve(paper_ref)
    if not match:
        return None, f"could not resolve a paper from: {paper_ref!r}"
    # Guard against a confident-looking but WRONG match: resolve() returns the best
    # candidate regardless of quality, so a low similarity means we probably matched a
    # *different* paper. Refuse it (the workflow won't close the issue) and ask for the
    # arXiv link, rather than silently adding the wrong paper.
    if match["score"] < ACCEPT_THRESHOLD:
        return None, (f"couldn't confidently match the title {paper_ref!r} — the closest "
                      f"arXiv paper was **{match['title']}** (arXiv:{match['id']}, similarity "
                      f"{match['score']:.2f}, below the {ACCEPT_THRESHOLD} threshold), which is "
                      f"likely a *different* paper. Please edit this issue and paste the arXiv "
                      f"link or id so I add the right one.")
    return match["id"], f"resolved title -> arXiv:{match['id']} " \
                        f"({match['title']}, similarity {match['score']:.2f})"


def classify_category(aid: str):
    from enrich import enrich
    from llm_triage import classify
    rec = enrich([aid], use_s2=False)
    if not rec:
        return "foundations"
    return classify(rec[0])


def classify_artifact(aid: str):
    """Auto-triage what KIND of thing this is, so the form can route it correctly.

    Returns (kind, category): kind is 'paper' | 'benchmark' | 'survey'; category is the
    paper subcategory when kind == 'paper' (empty otherwise). This is what lets an
    Auto-classified benchmark land in benchmarks.yaml and a survey in surveys.yaml
    instead of being forced into a paper section of papers.yaml.
    """
    from enrich import enrich
    from llm_triage import classify, classify_kind
    rec = enrich([aid], use_s2=False)
    if not rec:
        return "paper", "foundations"
    kind = classify_kind(rec[0])
    if kind == "paper":
        return "paper", classify(rec[0])
    return kind, ""


def main():
    body = os.environ.get("ISSUE_BODY", "")
    if "--body-file" in sys.argv:
        body = Path(sys.argv[sys.argv.index("--body-file") + 1]).read_text()
    if not body.strip():
        set_output("error", "empty issue body — nothing to add.")
        return 1

    fields = parse_form(body)
    paper_ref = _get(fields, "arxiv link or paper title", "paper link", "paper")
    cat_label = _get(fields, "category").lower()
    venue_override = _get(fields, "venue (optional)", "venue")
    bench_group = _get(fields, "benchmark group (benchmarks only)", "benchmark group")
    author = _get(fields, "corresponding author (benchmarks & surveys, optional)",
                  "corresponding author", "author")

    if not paper_ref:
        set_output("error", "no paper link/title found in the form.")
        return 1

    aid, note = resolve_input(paper_ref)
    if not aid:
        set_output("error", note)
        return 1

    # Decide the artifact KIND up-front so the Auto path can route benchmarks and
    # surveys to their own files instead of forcing everything into papers.yaml.
    auto_routed = False
    category = None
    cat_note = ""
    if cat_label == "survey":
        kind = "survey"
    elif cat_label == "benchmark":
        kind = "benchmark"
    elif cat_label in CATEGORY_MAP and CATEGORY_MAP[cat_label]:
        kind, category = "paper", CATEGORY_MAP[cat_label]
        cat_note = f"category: `{category}` (from form)"
    else:
        try:
            kind, category = classify_artifact(aid)
        except SystemExit:
            set_output("error", "auto-classify needs an LLM key (DEEPSEEK_API_KEY secret); "
                                "please pick a category in the form and re-open.")
            return 1
        auto_routed = kind != "paper"
        cat_note = (f"category: `{category}` (auto-classified)" if kind == "paper"
                    else f"auto-detected as a **{kind}**")

    # surveys live in surveys.yaml (flat title/venue/year/paper schema)
    if kind == "survey":
        from add_survey import insert_survey
        try:
            s = insert_survey(aid, corresponding=author or None,
                              venue=venue_override or None)
        except ValueError as exc:
            status = "duplicate" if "already" in str(exc) else "error"
            set_output(status, f"not added: {exc}")
            return 0 if status == "duplicate" else 1
        subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)
        lines = [f"✅ Added survey **{s['title']}**", "",
                 (note + "\n" if note else "") + f"venue: **{s['venue']}**"
                 + (f"  ·  corresponding: {s['corresponding']}" if s['corresponding'] else ""),
                 "```yaml", s["yaml"].strip(), "```"]
        if auto_routed:
            lines.append("_Auto-detected as a **survey** and added to `surveys.yaml`. "
                         "If that's wrong, remove it and re-add with an explicit category._")
        set_output("added", "\n".join(lines), title=s["title"])
        return 0

    # benchmarks live in benchmarks.yaml with their own group/tldr schema
    if kind == "benchmark":
        from add_benchmark import insert_benchmark, DEFAULT_GROUP
        group = DEFAULT_GROUP
        if bench_group and not bench_group.lower().startswith("auto"):
            group = bench_group
        try:
            b = insert_benchmark(aid, group=group, label=author or None)
        except ValueError as exc:
            status = "duplicate" if "already" in str(exc) else "error"
            set_output(status, f"not added: {exc}")
            return 0 if status == "duplicate" else 1
        subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)
        lines = [f"✅ Added benchmark **{b['name']}** to group *{b['group']}*",
                 "",
                 (note + "\n" if note else "") +
                 f"label: {b['label']}  ·  year: {b['year']}"
                 + (f"  ·  tldr drafted by LLM" if b.get("tldr") else ""),
                 "```yaml", b["block"].strip(), "```",
                 "_Group and author come from the form (defaults to \"Data Agent "
                 "Benchmarks\" / last author). Tweak the tldr in `data/benchmarks.yaml` "
                 "if needed._"]
        if auto_routed:
            lines.append("_Auto-detected as a **benchmark** and routed to `benchmarks.yaml`. "
                         "If that's wrong, remove it and re-add with an explicit category._")
        set_output("added", "\n".join(lines), title=b["name"])
        return 0

    try:
        res = insert_paper(aid, category, venue=venue_override or None)
    except ValueError as exc:
        status = "duplicate" if "already exists" in str(exc) else "error"
        set_output(status, f"not added: {exc}")
        return 0 if status == "duplicate" else 1

    subprocess.run([sys.executable, str(HERE / "generate_readme.py")], check=True)

    venue_line = (f"venue: **{res['venue']}**"
                  + (" ⚠️ *could not ground a real venue — please verify*"
                     if res["venue_todo"] else f" (via {res['venue_source']})"))
    lines = [
        f"✅ Added **{res['title']}**",
        "",
        (note + "\n" if note else "") + cat_note,
        venue_line,
        "",
        "```yaml",
        res["yaml"].strip(),
        "```",
        "",
        "_Corresponding author was set to the last author — please double-check._",
    ]
    set_output("added", "\n".join(lines), title=res["title"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
