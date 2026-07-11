# Curation rubric — how a paper earns a place on this list

This list is **curated**, not a keyword dump. A candidate is included only if it clears
the bar below. The final call always rests with a human maintainer; the rubric exists to
keep triage consistent (and to guide the automated triage of the weekly arXiv scan).

## Include if it clears **any one** of these gates

### Gate A — Substance
Proposes a **named system / method / benchmark / dataset** that is squarely about one of
our categories, with a **novel mechanism** (not "we apply an off-the-shelf LLM to task Y").
Releasing **code** is a strong plus.
- Core directions (`nl2sql`, `data-analysis`, `data-science`, `db-operations`,
  `table-reasoning`, `data-preparation`): apply this gate **leniently**.
- `memory` and `foundations`: apply **strictly** — the volume is huge, so require a named
  system / memory-OS / benchmark / survey or a clearly novel mechanism, and drop
  domain-specific applications (medical dialogue memory, role-play chatbots, etc.).

### Gate B — Institution whitelist
Authored (ideally led — check the corresponding / last author) by a **top institution**:

**Universities.** SJTU, Tsinghua (THU), Peking (PKU), Fudan, Zhejiang (ZJU), USTC,
Renmin (RUC), Nanjing (NJU), HKU, HKUST, CUHK, NUS, NTU, KAIST; MIT, Stanford,
UC Berkeley (UCB), CMU, Harvard, Princeton, Yale, Caltech, UIUC, Georgia Tech, UW,
UCLA, UCSD, Cornell, Columbia, Michigan, UT Austin, Wisconsin–Madison, NYU, UMass,
Waterloo, Toronto, Mila/Montreal; Oxford, Cambridge, ETH Zürich, EPFL, Edinburgh,
Max Planck, TU Munich, Tel Aviv.

**Companies / industry labs.** OpenAI, Anthropic, Google / Google DeepMind, Meta / FAIR,
Microsoft / Microsoft Research, Amazon / AWS, Apple, NVIDIA, Salesforce Research,
Databricks, Snowflake, IBM Research, Adobe Research, ByteDance / Seed, Alibaba (DAMO /
Qwen), Tencent, Baidu, Huawei / Noah's Ark, Ant Group, Zhipu AI, Moonshot, DeepSeek,
Xiaomi, Kuaishou.

Small / unknown institutions do **not** clear this gate on their own — they must clear
Gate A or C instead.

### Gate C — Top venue acceptance
Accepted at a **CCF-A** venue or a **field-leading CCF-B** venue:
- CCF-A: SIGMOD, VLDB, ICDE, PODS; NeurIPS, ICML, ICLR, AAAI, IJCAI; ACL, KDD, SIGIR,
  WWW/TheWebConf; OSDI, SOSP.
- Top CCF-B: EMNLP, NAACL, WSDM, CIKM, COLING (findings-only tracks are weaker — treat
  as borderline, not an automatic include).

## Always drop
Off-topic (robotics control, pure vision, medical/clinical, social simulation), papers
that only mention agents/memory/data in passing, thin position pieces with no artifact,
and anything already in `data/*.yaml`.

## Signals used during triage
- **arXiv `comment` field** — often states "Accepted at EMNLP 2026" → Gate C.
- **Semantic Scholar `citationCount`** — weak for <3-month-old papers, but an early
  citation count that is already high is a positive signal.
- **Author affiliations** (S2, where available) + **maintainer/model knowledge of who the
  well-known authors are** — for Gate B. Affiliation metadata is sparse for fresh
  preprints, so recognising the corresponding author's lab is often the practical route.
