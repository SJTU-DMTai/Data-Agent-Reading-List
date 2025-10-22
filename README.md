# Awesome Data Agent and Table Reasoning Paper

## Benchmark

### Data Agent Benchmark

| Benchmark Name | Corresp. Author     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| InfiAgent-DABench | Jingjing Xu | 2024 | [Link](https://arxiv.org/pdf/2401.05507) [Repo](https://github.com/InfiAgent/InfiAgent)| Benchmark from ICML'24 Paper. Input: NL data-analysis queries plus linked CSVs (executed in a constrained Python sandbox). Output: Closed-form answers in a fixed format (e.g., @answer_name[answer]). Metrics: Exact-match scoring on the canonicalized answers; aggregate accuracy over sub-tasks.
| CRAG | Xiao Yang | 2024 | [Link](https://arxiv.org/pdf/2406.04744) [Repo](https://github.com/facebookresearch/CRAG/)| Benchmark from NIPS'24 D&B track. Input: Factoid questions with access to provided corpora and mock retrieval/APIs. Output: Final textual answer (optionally with cited evidence). Metrics: Automatic correctness categories mapped to a score (correct/acceptable vs. missing/incorrect); report overall accuracy.
| FDABench | Gao Cong | 2025 | [Link](https://arxiv.org/pdf/2509.02473) [Repo](https://github.com/fdabench/FDAbench)| Input: Cross-source analytical queries spanning structured tables and unstructured sources. Output: Multiple-choice selections or a short analysis/report when required. Metrics: Exact-match for MC/checkbox tasks; text tasks by overlap metrics (e.g., ROUGE) plus tool-use success/recall; also reports efficiency (latency, model/tool calls, token/cost).
| DA-Bench | Yunjun Gao | 2025 | [Link](https://arxiv.org/pdf/2503.13269) [Repo](https://github.com/ZJU-LLMs/DAgent)| Input: Realistic analyst tasks against a live data stack/tools. Output: Final answers or produced charts/tables. Metrics: Per-task 0–5 rubric with penalties for extra user interaction, corrections, or hallucinations; summarize overall score and hallucination rate.
| DSBench | Dong Yu | 2025 | [Link](https://arxiv.org/pdf/2409.07703) [Repo](https://github.com/LiqiangJing/DSBench)| Benchmark from ICLR'25 Paper. Input: Realistic analyst tasks against a live data stack/tools. Output: Final answers or produced charts/tables. Metrics: Per-task 0–5 rubric with penalties for extra user interaction, corrections, or hallucinations; summarize overall score and hallucination rate.
| DABStep | Thomas Wolf | 2025 | [Link](https://arxiv.org/pdf/2506.23719) [Repo](https://huggingface.co/spaces/adyen/DABstep)| Input: Multi-step data-analysis tasks requiring iterative code execution over structured tables (CSV/JSON) and cross-referencing unstructured domain manuals and documentation; Output: single factoid answers (string or numeric) formatted per task spec; 
| DataSciBench | Yisong Yue | 2025 | [Link](https://arxiv.org/pdf/2502.13897) [Repo](https://github.com/THUDM/DataSciBench/)| 
| DSEval | Kan Ren | 2024 | [Link](https://arxiv.org/pdf/2402.17168) [Repo](https://github.com/MetaCopilot/dseval)| 
| WebDS | Christopher D. Manning | 2025 | [Link](https://arxiv.org/pdf/2508.01222)| 
| PredictiQ | Xiaojun Ma | 2025 |[Link](https://arxiv.org/pdf/2505.17149) [Repo](https://github.com/Cqkkkkkk/PredictiQ)| 
| InsightBench | Issam Hadj Laradji | 2025 | [Paper](https://arxiv.org/pdf/2407.06423) [Repo](https://github.com/ServiceNow/insight-bench)|

### Latest Benchmarks for Tabular Data (Since 2024)

| Benchmark Name | Task Type     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| MDBench        | Reasoning     | 2025 | [Link](https://github.com/jpeper/MDBench) | MDBench introduces a new multi-document reasoning benchmark synthetically generated through knowledge-guided prompting. |
| MMQA           | Reasoning     | 2025 | [Link](https://openreview.net/pdf?id=GGlpykXDCa)  [Repo](https://github.com/WuJian1995/MMQA/issues/2)| MMQA is a multi-table multi-hop question answering dataset with 3,312 tables across 138 domains, evaluating LLMs' capabilities in multi-table retrieval, Text-to-SQL, Table QA, and primary/foreign key selection.
| ToRR           | Reasoning     | 2025 | [Link](https://arxiv.org/pdf/2502.19412)  [Repo](https://github.com/IBM/unitxt/blob/main/prepare/benchmarks/torr.py)| ToRR is a benchmark assessing LLMs' table reasoning and robustness across 10 datasets with diverse table serializations and perturbations, revealing models' brittleness to format variations.
| MMTU           | Comprehensive | 2025 | [Link](https://arxiv.org/pdf/2506.05587)  [Repo](https://github.com/MMTU-Benchmark/MMTU)| MMTU is a massive multi-task table understanding and reasoning benchmark with over 30K questions across 25 real-world table tasks, designed to evaluate models' ability to understand, reason, and manipulate tables.
| RADAR          | Reasoning     | 2025 | [Link](https://kenqgu.com/assets/pdf/RADAR_ARXIV.pdf)  [Repo](https://huggingface.co/datasets/kenqgu/RADAR)| RADAR is a benchmark for evaluating language models' data-aware reasoning on imperfect tabular data with 5 common data artifact types like outlier value or inconsistent format, which ensures that direct calculation on the perturbed table will yield an incorrect answer, forcing the model to handle the artifacts to obtain the correct result.
| Spider2        | Text2SQL      | 2025 | [Link](https://arxiv.org/abs/2411.07763)  [Repo](https://github.com/xlang-ai/Spider2)|
| DataBench      | Reasoning     | 2024 | [Link](https://aclanthology.org/2024.lrec-main.1179.pdf)  [Repo](https://huggingface.co/datasets/cardiffnlp/databench)|
| TableBench     | Reasoning     | 2024 | [Link](https://arxiv.org/abs/2408.09174)  [Repo](https://github.com/TableBench/TableBench)|
| TQA-Bench      | Reasoning     | 2024 | [Link](https://arxiv.org/pdf/2411.19504)  [Repo](https://github.com/Relaxed-System-Lab/TQA-Bench)|
| SpreadsheetBench | Reasoning   | 2024 | [Link](https://arxiv.org/pdf/2406.14991)  [Repo](https://github.com/RUCKBReasoning/SpreadsheetBench/tree/main/data)|

### Selected Classical Reasoning Benchmarks for Tabular Data
| Benchmark Name | Task Type     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| FinQA          | Reasoning     | 2021 | [Link](https://arxiv.org/pdf/2109.00122)  [Repo](https://github.com/czyssrs/FinQA)|
| FeTaQA         | Reasoning     | 2021 | [Link](https://arxiv.org/pdf/2104.00369)  [Repo](https://github.com/Yale-LILY/FeTaQA)|
| HiTab          | Reasoning     | 2022 | [Link](https://aclanthology.org/2022.acl-long.78.pdf) [Repo](https://github.com/microsoft/HiTab)


## Conference Paper (including arxiv)
| Venue       | Paper                                                        | Corresp. Author |                           Links                             |
| :---------- | :----------------------------------------------------------- | :-------------: |:----------------------------------------------------------: |
| SIGMOD'26   | ST-Raptor: LLM-Powered Semi-Structured Table Question Answering | Xuanhe Zhou |  [paper](https://arxiv.org/pdf/2508.18190)   |
| CIDR'25     | AOP: Automated and Interactive LLM Pipeline Orchestration for Answering Complex Queries | Guoliang Li |  [paper](https://www.vldb.org/cidrdb/papers/2025/p32-wang.pdf)   |
| CIDR'25     | Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing | Gerardo Vitagliano |  [paper](https://www.vldb.org/cidrdb/papers/2025/p12-liu.pdf)   |
| IEEE Data Eng. Bull.  | iDataLake: An LLM-Powered Analytics System on Data Lakes | Guoliang Li |  [paper](http://sites.computer.org/debull/A25mar/p57.pdf#:~:text=In%20this%20paper%20we%20highlight%20challenges%20for%20supporting,query%20and%20outputs%20the%20results%20for%20the%20query.)   |
| VLDB'25     | Towards Automated Cross-domain Exploratory Data Analysis through Large Language Models  | Qi Liu | [paper](https://www.vldb.org/pvldb/vol18/p5086-zhu.pdf)   |
| VLDB'25     | AutoPrep: Natural Language Question-Aware Data Preparation with a Multi-Agent Framework | Ju Fan | [paper](https://www.vldb.org/pvldb/vol18/p5086-zhu.pdf)   |
| VLDB'25     | DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing | Eugene Wu | [paper](https://arxiv.org/pdf/2410.12189)   |
| VLDB'25     | Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS | Matei Zaharia | [paper](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf)   |
| ICDE'25     | DataLab: A Unified Platform for LLM-Powered Business Intelligence | Wei Chen | [paper](https://arxiv.org/pdf/2412.02205)   |
| ICML'25     | AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML | Sung Ju Hwang | [paper](https://arxiv.org/pdf/2410.02958)   |
| ICML'25     | Compositional Condition Question Answering in Tabular Understanding | Han-Jia Ye | [paper](https://openreview.net/attachment?id=aXU48nrA2v&name=pdf)   |
| ICML'25     | Are Large Language Models Ready for Multi-Turn Tabular Data Analysis? | Reynold Cheng | [paper](https://openreview.net/attachment?id=flKhxGTBj2&name=pdf) |
| ICML'25     | Agent Workflow Memory | Graham Neubig |  [paper](https://arxiv.org/pdf/2409.07429)   |
| ICLR'25     | SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents | Qianhui Wu | [paper](https://openreview.net/pdf?id=xKDZAW0He39) |
| ICLR'25     | InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation | Issam Hadj Laradji | [paper](https://arxiv.org/pdf/2407.06423) |
| ICLR'25     | Agent-Oriented Planning in Multi-Agent Systems | Yaliang Li | [paper](https://arxiv.org/pdf/2410.02189) |
| ICLR'25 Oral | AFlow: Automating Agentic Workflow Generation | Chenglin Wu | [paper](https://openreview.net/pdf?id=z5uVAKwmjf) |
| NAACL'25    | H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables | Chandan K. Reddy | [paper](https://arxiv.org/pdf/2407.05952) |
| ACL'25 Findings | Data Interpreter: An LLM Agent For Data Science | Bang Liu & Chenglin Wu | [paper](https://arxiv.org/pdf/2402.18679) |
| Neurips'25  | Table as a Modality for Large Language Models | Junbo Zhao | [paper](https://neurips.cc/virtual/2025/poster/116332)  |
| Arxiv/2510  | LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation | Dongge Han | [paper](https://arxiv.org/pdf/2510.04851) |
| Arxiv/2510  | TOOLMEM: Enhancing Multimodal Agents with Learnable Tool Capability Memory | Zora Zhiruo Wang | [paper](https://arxiv.org/pdf/2510.06664) |
| Arxiv/2510  | Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks | Jitao Sang | [paper](https://arxiv.org/abs/2510.12635) |
| Arxiv/2510  | LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science | Tomas Pfister | [paper](https://arxiv.org/pdf/2510.01285) |
| Arxiv/2510  | Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models | Qizheng Zhang & Changran Hu | [paper](https://www.arxiv.org/abs/2510.04618) |
| Arxiv/2509  | Latent learning: episodic memory complements parametric learning by enabling flexible reuse of experiences | Andrew Kyle Lampinen | [paper](https://arxiv.org/pdf/2509.16189) |
| Arxiv/2509  | H2R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents | Chengdong Xu | [paper](https://arxiv.org/pdf/2509.12810) |
| Arxiv/2509  | Mem-α: Learning Memory Construction via Reinforcement Learning | Yu Wang | [paper](https://arxiv.org/pdf/2509.25911) |
| Arxiv/2509  | ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory | Siru Ouyang | [paper](https://arxiv.org/abs/2509.25140) |
| Arxiv/2509  | SGMem: Sentence Graph Memory for Long-Term Conversational Agents | Yaxiong Wu |  [paper](https://arxiv.org/pdf/2509.21212v1)   |
| Arxiv/2509  | TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning | Qi Liu |  [paper](https://arxiv.org/pdf/2509.06278)   |
| Arxiv/2509  | MemGen: Weaving Generative Latent Memory for Self-Evolving Agents | Shuicheng Yan |  [paper](https://arxiv.org/pdf/2509.24704)   |
| Arxiv/2508  | Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models | Zhouhan Lin |  [paper](https://arxiv.org/pdf/2508.09874)   |
| Arxiv/2508  | Memory-R1: Enhancing Large Language Model Agents to Actively Manage and Utilize External Memory | Yunpu Ma |  [paper](https://arxiv.org/pdf/2508.19828)   |
| Arxiv/2508  | Data Agent: A Holistic Architecture for Orchestrating Data+AI Ecosystems | Guoliang Li |  [paper](https://arxiv.org/pdf/2507.01599)   |
| Arxiv/2508  | Memp: Exploring Agent Procedural Memory | Ningyu Zhang |  [paper](https://arxiv.org/pdf/2508.06433)   |
| Arxiv/2508  | Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory | Wei Li |  [paper](https://arxiv.org/pdf/2508.09736)   |
| Arxiv/2508  | AgenticData: An Agentic Data Analytics System for Heterogeneous Data | Yuan Li |  [paper](https://arxiv.org/pdf/2508.50002)   |
| Arxiv/2508  | Multiple Memory Systems for Enhancing the Long-term Memory of Agent | Bo Wang |  [paper](https://arxiv.org/pdf/2508.15294)   |
| Arxiv/2507  | H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents | Shaoning Zeng |  [paper](https://arxiv.org/pdf/2507.22925v1)   |
| Arxiv/2507  | MemOS: A Memory OS for AI System | Siheng Chen, Wentao Zhang, Zhi-Qin John Xu, Feiyu Xiong |  [paper](https://arxiv.org/pdf/2507.07957)   |
| Arxiv/2507  | MIRIX: Multi-Agent Memory System for LLM-Based Agents | Xi Chen |  [paper](https://arxiv.org/pdf/2507.07957)   |
| Arxiv/2507  | MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent | Hao Zhou |  [paper](https://arxiv.org/pdf/2507.02259)   |
| Arxiv/2506  | MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning | Thuy-Trang Vu |  [paper](https://arxiv.org/pdf/2503.13269)   |
| Arxiv/2506  | Memory OS of AIAgent | Ting Bai |  [paper](https://arxiv.org/pdf/2506.06326)   |
| Arxiv/2506  | G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems | Shuicheng Yan |  [paper](https://arxiv.org/pdf/2506.07398)   |
| Arxiv/2506  | AutoMind: Adaptive Knowledgeable Agent for Automated Data Science | Ningyu Zhang |  [paper](https://arxiv.org/pdf/2506.10974)   |
| Arxiv/2505  | DSMentor: Enhancing Data Science Agents with Curriculum Learning and Online Knowledge Accumulation | Patrick Ng |  [paper](https://arxiv.org/pdf/2505.14163)   |
| Arxiv/2505  | TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes | Ju Fan |  [paper](https://arxiv.org/pdf/2505.11270)   |
| Arxiv/2505  | How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior | Zidi Xiong |  [paper](https://arxiv.org/pdf/2505.16067)   |
| Arxiv'2505  | Weaver: Interweaving SQL and LLM for Table Reasoning | Vivek Gupta |  [paper](https://arxiv.org/pdf/2505.18961)    |
| Arxiv/2504  | AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery | Issam H. Laradji |  [paper](https://arxiv.org/pdf/2504.07421v3)   |
| Arxiv/2504  | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | Deshraj Yadav |  [paper](https://arxiv.org/pdf/2504.19413)   |
| Arxiv/2503  | R3Mem: Bridging Memory Retention and Retrieval via Reversible Compression | Yun Zhu |  [paper](https://arxiv.org/pdf/2502.15957v1)   |
| Arxiv/2503  | DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Science | Yu Huang |  [paper](https://arxiv.org/pdf/2503.07044)   |
| Arxiv/2503  | DAgent: A Relational Database-Driven Data Analysis Report Generation Agent | Yunjun Gao |  [paper](https://arxiv.org/pdf/2503.13269)   |
| Arxiv/2502  | A-MEM: Agentic Memory for LLM Agents | Yongfeng Zhang |  [paper](https://arxiv.org/pdf/2502.12110)   |
| Arxiv'2501  | TableMaster: A Recipe to Advance Table Understanding with Language Models | Hanbing Liu |  [paper](https://arxiv.org/pdf/2501.19378)    |
| Arxiv'2410  | AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competition | Wenhao Huang & Ge Zhang | [paper](https://arxiv.org/abs/2410.20424) |
| Arxiv'2407  | LAMBDA: A Large Model Based Data Agent | Yancheng Yuan & Jian Huang | [paper](https://arxiv.org/abs/2407.17535) |
| ICML'24     | DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning | Hechang Chen | [paper](https://arxiv.org/abs/2402.17453) |
| ICLR'24     | OpenTab: Advancing Large Language Models as Open-domain Table Reasoners | Jiani Zhang  |  [paper](https://arxiv.org/pdf/2402.14361)    |
| ICLR'24     | CABINET: Content Relevance based Noise Reduction for Table Question Answering |  Balaji Krishnamurthy |  [paper](https://arxiv.org/pdf/2402.01155)    |
| ICLR'24     | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Hannaneh Hajishirzi  |  [paper](https://arxiv.org/pdf/2401.04398)    |
| ICLR'24     | Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding | Tomas Pfister  |  [paper](https://arxiv.org/pdf/2401.04398)    |
| ICLR'24     | MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework | Chenglin Wu  |  [paper](https://arxiv.org/pdf/2308.00352)    |
| AAAI'24     | ExpeL: LLM Agents Are Experiential Learners | Gao Huang |  [paper](https://arxiv.org/pdf/2308.10144)    |
| VLDB'24     | ReAcTable: Enhancing ReAct for Table Question Answering |  Jignesh M. Patel |  [paper](https://arxiv.org/pdf/2310.00815)    |
| VLDB'24     | AutoTQA: Towards Autonomous Tabular Question Answering through Multi-Agent Large Language Models | Qi Liu |  [paper](https://www.vldb.org/pvldb/vol17/p3920-zhu.pdf)    |
| VLDB'24     | D-Bot: Database Diagnosis System using Large Language Models | Guoliang Li |  [paper](https://arxiv.org/pdf/2312.01454)    |
| NIPS'23     | Augmenting language models with long-term memory | Furu Wei |  [paper](https://arxiv.org/pdf/2306.07174)    |
| NIPS'23     | Reflexion: Language Agents with Verbal Reinforcement Learning | Shunyu Yao |  [paper](https://arxiv.org/pdf/2303.11366)    |
| SIGIR'23    | Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning (Dater) | Yongbin Li |  [paper](https://arxiv.org/pdf/2301.13808)    |
| Arxiv'2310  | MemGPT: Towards LLMs as Operating Systems | Charles Packer |  [paper](https://arxiv.org/pdf/2310.08560)    |

## Survey Paper
[Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions](https://arxiv.org/pdf/2505.11270)
[A Survey on Large Language Model-based Agents for Statistics and Data Science](https://arxiv.org/abs/2412.14222)
[Large Language Model-based Data Science Agent: A Survey](https://arxiv.org/pdf/2508.02744)
[LLM/Agent-as-Data-Analyst: A Survey](https://arxiv.org/pdf/2509.23988)







