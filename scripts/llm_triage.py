#!/usr/bin/env python3
"""Triage enriched paper candidates against scripts/curation_rubric.md using an LLM.

Talks to any OpenAI-compatible chat API. Defaults to DeepSeek (cheap, good enough for
this classification task). Configure via environment variables:

    LLM_API_KEY    (required)  e.g. your DeepSeek key   — or DEEPSEEK_API_KEY
    LLM_BASE_URL   default: https://api.deepseek.com
    LLM_MODEL      default: deepseek-chat

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/llm_triage.py enriched.jsonl                 # -> verdicts.jsonl on stdout
    python scripts/llm_triage.py enriched.jsonl --out verdicts.jsonl

Each output record: {id, decision: keep|maybe|drop, category, gates, reason, ...enriched}.

Importable:
    from llm_triage import triage
    verdicts = triage(records, rubric_text)
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = ROOT / "scripts" / "curation_rubric.md"
CATEGORIES = ["data-preparation", "nl2sql", "table-reasoning", "table-curation", "data-analysis",
              "data-science", "db-operations", "memory", "foundations",
              "benchmark", "survey"]

SYSTEM = """You triage paper candidates for a CURATED awesome-list on LLM-based data \
agents (github.com/SJTU-DMTai/Awesome-Data-Agent-Papers). Apply the rubric below EXACTLY.

A paper is KEEP if it clears ANY ONE gate:
  Gate A (Substance): a named system/method/benchmark/dataset squarely in one of our
    categories with a novel mechanism. Lenient for the core directions (nl2sql,
    data-analysis, data-science, db-operations, table-reasoning, table-curation,
    data-preparation);
    STRICT for memory and foundations (require a named system/OS/benchmark/survey or a
    clearly novel mechanism; drop domain-specific applications).
  Gate B (Institution): led by a whitelisted top university or company. Affiliations are
    often missing from the data, so use your own knowledge of who the authors are —
    especially the last/corresponding author. Only credit Gate B when you actually
    recognize the author or an affiliation string confirms it; never guess wildly.
  Gate C (Top venue): the COMMENT field (or abstract) states acceptance at a CCF-A venue
    (SIGMOD/VLDB/ICDE/PODS, NeurIPS/ICML/ICLR/AAAI/IJCAI, ACL/KDD/SIGIR/WWW, OSDI/SOSP)
    or a field-leading CCF-B venue (EMNLP/NAACL/WSDM/CIKM/COLING).
Otherwise DROP. Use MAYBE only when genuinely borderline. Always DROP off-topic work
(robotics, pure vision, medical/clinical, social simulation) and thin position pieces.

--- FULL RUBRIC ---
%s
--- END RUBRIC ---

Reply with ONLY a JSON array, one object per input paper, in the same order:
[{"id": "<arxiv-id>", "decision": "keep|maybe|drop",
  "category": "<one of: %s>",
  "gates": "<subset of A,B,C that triggered, or empty for drop>",
  "reason": "<<=20 words; name the system and, for Gate B, the institution/author>"}]
No prose, no markdown fences.""" % ("%s", ", ".join(CATEGORIES))


def _endpoint(base):
    base = base.rstrip("/")
    return base if base.endswith(("/chat/completions",)) else base + "/chat/completions"


def _call(messages, cfg, retries=4):
    body = json.dumps({"model": cfg["model"], "messages": messages,
                       "temperature": 0, "stream": False}).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {cfg['key']}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(_endpoint(cfg["base"]), data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
                data = json.load(r)
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  LLM call retry {attempt + 1}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def _parse_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()  # strip code fences
    m = re.search(r"\[.*\]", text, re.S)  # tolerate leading/trailing chatter
    return json.loads(m.group(0) if m else text)


def _render(r):
    lines = [f"id: {r['id']}  (published {r.get('published', '?')}, "
             f"citations={r.get('citations')})",
             f"title: {r['title']}",
             "authors: " + ", ".join(r.get("authors", [])[:12])]
    if r.get("affils"):
        lines.append("affiliations: " + " ; ".join(r["affils"][:8]))
    if r.get("comment"):
        lines.append("comment: " + r["comment"][:220])
    lines.append("abstract: " + r.get("abstract", "")[:800])
    return "\n".join(lines)


def triage(records, rubric_text=None, cfg=None, batch_size=12):
    """Classify records; returns each record augmented with decision/category/gates/reason."""
    if rubric_text is None:
        rubric_text = RUBRIC.read_text()
    if cfg is None:
        cfg = load_config()
    system = SYSTEM % rubric_text
    out = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        user = "Triage these %d papers:\n\n%s" % (
            len(batch), "\n\n---\n\n".join(_render(r) for r in batch))
        content = _call([{"role": "system", "content": system},
                         {"role": "user", "content": user}], cfg)
        try:
            verdicts = _parse_json_array(content)
        except Exception as exc:
            print(f"  batch {i // batch_size}: unparseable response ({exc}); "
                  "marking as maybe", file=sys.stderr)
            verdicts = []
        by_id = {str(v.get("id")): v for v in verdicts if isinstance(v, dict)}
        for r in batch:
            v = by_id.get(r["id"], {})
            out.append({**r,
                        "decision": (v.get("decision") or "maybe").lower(),
                        "category": v.get("category", ""),
                        "gates": v.get("gates", ""),
                        "reason": v.get("reason", "")})
        print(f"  triaged {min(i + batch_size, len(records))}/{len(records)}",
              file=sys.stderr)
    return out


PAPER_CATEGORIES = ["data-preparation", "nl2sql", "table-reasoning", "table-curation", "data-analysis",
                    "data-science", "db-operations", "memory", "foundations"]

CLASSIFY_SYSTEM = """You assign a single category to a paper for an awesome-list on \
LLM-based data agents. Pick the ONE best-fitting key from:
- data-preparation: cleaning/transforming/imputing/integrating data
- nl2sql: text-to-SQL / natural language to SQL
- table-reasoning: QA and reasoning over tables / spreadsheets / semi-structured data
- table-curation: generating / synthesizing / augmenting tables and tabular data, schema & column expansion, dataset construction
- data-analysis: EDA, BI, insight discovery, semantic operators, report/visualization, time-series analytics
- data-science: autonomous DS/ML/AutoML/Kaggle-style agents, model building
- db-operations: LLM agents for DBA work (diagnosis, tuning, config, vector-DB ops)
- memory: agent memory systems, memory OS, context engineering/folding, experience-driven self-evolution
- foundations: general agent building blocks (planning, workflow, multi-agent, RAG, skills)
Reply with ONLY the key, nothing else."""


def classify(record, cfg=None):
    """Return the single best category key for one enriched paper record."""
    if cfg is None:
        cfg = load_config()
    user = (f"Title: {record['title']}\n\n"
            f"Abstract: {record.get('abstract', '')[:1200]}")
    out = _call([{"role": "system", "content": CLASSIFY_SYSTEM},
                 {"role": "user", "content": user}], cfg).strip().lower()
    for key in PAPER_CATEGORIES:  # tolerate extra words in the reply
        if key in out:
            return key
    return "foundations"  # safe default; caller can override


def load_config():
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("error: set LLM_API_KEY (or DEEPSEEK_API_KEY) in the environment")
    return {
        "key": key,
        "base": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
        "timeout": int(os.environ.get("LLM_TIMEOUT", "120")),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("enriched", help="enriched JSONL from scripts/enrich.py ('-' for stdin)")
    p.add_argument("--out", help="write verdict JSONL here instead of stdout")
    p.add_argument("--batch-size", type=int, default=12)
    args = p.parse_args()

    src = sys.stdin if args.enriched == "-" else open(args.enriched)
    records = [json.loads(l) for l in src if l.strip()]
    verdicts = triage(records, batch_size=args.batch_size)

    out = open(args.out, "w") if args.out else sys.stdout
    for v in verdicts:
        out.write(json.dumps(v, ensure_ascii=False) + "\n")
    if args.out:
        out.close()
        kept = sum(1 for v in verdicts if v["decision"] == "keep")
        maybe = sum(1 for v in verdicts if v["decision"] == "maybe")
        print(f"wrote {len(verdicts)} verdicts to {args.out} "
              f"(keep={kept} maybe={maybe})", file=sys.stderr)


if __name__ == "__main__":
    main()
