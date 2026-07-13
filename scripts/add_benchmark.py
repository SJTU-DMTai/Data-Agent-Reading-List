#!/usr/bin/env python3
"""Add a benchmark to data/benchmarks.yaml from an arXiv link — the benchmark analogue
of add_paper.py (benchmarks were manual-only until now).

benchmarks.yaml is a list of GROUPS, each with items {name, label, year, paper, code?,
tldr?}. `label` means the corresponding author in the "Corresp. Author" groups and the
task type in the "Task Type" groups. We fetch metadata from arXiv, ground the year,
sniff a code repo, optionally draft a house-style `tldr` via the LLM, and insert the
item into the chosen group — preserving the file's comments and folded-scalar tldrs by
editing the text (no full YAML re-dump).

Usage:
    python scripts/add_benchmark.py 2602.13812                       # -> "Data Agent Benchmarks"
    python scripts/add_benchmark.py 2602.13812 -g "Latest Benchmarks for Tabular Data (Since 2024)"
    python scripts/add_benchmark.py 2602.13812 --name DTBench --label "Table QA" --no-tldr

List the available groups:
    python scripts/add_benchmark.py --list-groups
"""
import argparse
import re
import sys
import textwrap
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_paper import fetch as arxiv_fetch, arxiv_id  # noqa: E402

BENCH = HERE.parent / "data" / "benchmarks.yaml"
DEFAULT_GROUP = "Data Agent Benchmarks"


def groups():
    return yaml.safe_load(BENCH.read_text())


def group_names():
    return [g["name"] for g in groups()]


def _derive_name(title):
    """Benchmark short name — usually the token before the first colon (e.g. 'DTBench: ...')."""
    return title.split(":", 1)[0].strip() if ":" in title else title.strip()


def _make_tldr(meta):
    """Draft a house-style tldr via the LLM; return None if no LLM key / on failure."""
    try:
        from enrich import enrich
        from llm_triage import _call, load_config
        cfg = load_config()
    except SystemExit:
        return None
    except Exception:
        return None
    rec = enrich([meta["id"]], use_s2=False)
    abstract = rec[0]["abstract"] if rec else ""
    prompt = (
        "Write a ONE-sentence TLDR (about 20-30 words, max 35) for this benchmark: what "
        "it evaluates and over what kind of data. Plain sentence, no markdown, no line "
        "breaks, no 'Sub-tasks/Input/Output/Metrics' structure.\n\n"
        f"Title: {meta['title']}\nAbstract: {abstract[:1600]}")
    try:
        out = _call([{"role": "user", "content": prompt}], cfg).strip()
    except Exception:
        return None
    return re.sub(r"\s+", " ", out) or None


def _item_block(name, label, year, paper, code, tldr, label_is_author):
    """Render one benchmark item at the file's 4/6/8-space indentation."""
    note = "  # TODO verify corresponding author" if label_is_author else "  # TODO set task type"
    lines = [f"    - name: {name}",
             f"      label: {label}{note}",
             f"      year: {year}",
             f"      paper: {paper}"]
    if code:
        lines.append(f"      code: {code}  # auto-detected from abstract, please verify")
    if tldr:
        lines.append("      tldr: >-")
        for wrapped in textwrap.wrap(tldr, width=96):
            lines.append("        " + wrapped)
    elif tldr is None:
        lines.append("      # tldr: >-   # TODO add a short description")
    return "\n".join(lines) + "\n"


def insert_benchmark(aid, group=DEFAULT_GROUP, name=None, label=None, code=None, make_tldr=True):
    """Insert a benchmark item into `group`. Returns a summary dict."""
    names = group_names()
    if group not in names:
        raise ValueError(f"unknown group {group!r}; choose from {names}")
    text = BENCH.read_text()
    if f"/pdf/{aid}" in text or f"/abs/{aid}" in text:
        raise ValueError(f"arXiv id {aid} already in {BENCH.name}")

    meta = arxiv_fetch(aid)
    year = meta["published"][:4]
    gmeta = next(g for g in groups() if g["name"] == group)
    label_is_author = "author" in (gmeta.get("label_column", "").lower())
    if not label:
        label = meta["authors"][-1] if label_is_author else "TODO"
    code = code or meta["code"]
    tldr = _make_tldr(meta) if (make_tldr and gmeta.get("show_tldr")) else (
        None if gmeta.get("show_tldr") else False)  # False => group has no tldr column

    block = _item_block(name or _derive_name(meta["title"]), label, year,
                        f"https://arxiv.org/pdf/{aid}", code,
                        tldr if tldr else None, label_is_author)

    # locate the group header, then the start of the next top-level group (or EOF)
    lines = text.splitlines(keepends=True)
    gi = next(i for i, l in enumerate(lines) if re.match(rf"^- name: {re.escape(group)}\s*$", l))
    ni = next((i for i in range(gi + 1, len(lines)) if re.match(r"^- name: ", lines[i])), len(lines))
    while ni - 1 > gi and lines[ni - 1].strip() == "":  # insert before trailing blank lines
        ni -= 1
    lines.insert(ni, block)
    BENCH.write_text("".join(lines))

    return {"id": aid, "name": name or _derive_name(meta["title"]), "group": group,
            "label": label, "year": year, "code": code,
            "tldr": (tldr if isinstance(tldr, str) else None), "block": block}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arxiv", nargs="?", help="arXiv id or url")
    p.add_argument("-g", "--group", default=DEFAULT_GROUP)
    p.add_argument("--name", help="short benchmark name (else derived from the title)")
    p.add_argument("--label", help="corresponding author or task type (else auto)")
    p.add_argument("--code", help="code repo URL")
    p.add_argument("--no-tldr", action="store_true", help="don't draft a tldr with the LLM")
    p.add_argument("--list-groups", action="store_true")
    args = p.parse_args()

    if args.list_groups:
        for g in groups():
            print(f"- {g['name']}  (label: {g.get('label_column')}, "
                  f"tldr: {g.get('show_tldr')}, {len(g['items'])} items)")
        return
    if not args.arxiv:
        p.error("give an arXiv id/url (or --list-groups)")

    try:
        res = insert_benchmark(arxiv_id(args.arxiv), group=args.group, name=args.name,
                               label=args.label, code=args.code, make_tldr=not args.no_tldr)
    except (ValueError, StopIteration) as exc:
        sys.exit(f"error: {exc}")
    print(res["block"])
    print(f"Added '{res['name']}' to group '{res['group']}'. "
          f"Now run: python scripts/generate_readme.py")


if __name__ == "__main__":
    main()
