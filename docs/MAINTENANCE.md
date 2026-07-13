# Maintenance Guide / 维护指南

How to use and maintain this reading list day to day. Almost everything can be
done from the GitHub website — you do not need to set up anything on your own
computer.

如何日常使用和维护这个论文列表。几乎所有操作都能在 GitHub 网页上完成，不需要在自己电脑上配置任何环境。

---

## 0. The one rule to remember / 必须记住的一条规则

The paper data lives in the `data/` folder as YAML files. The `README.md` is
**generated automatically** from that data:

```
data/*.yaml  ──(scripts/generate_readme.py)──►  README.md
   ▲ you edit here                                ▲ generated — do not edit by hand
```

Every time something is pushed to the `main` branch, a GitHub Action rebuilds
`README.md` and overwrites it. So if you edit `README.md` directly, your change
will be wiped out on the next build. **To change the content or the layout, edit
the YAML files or the generator script — never the README itself.**

论文数据以 YAML 文件的形式存放在 `data/` 文件夹里。`README.md` 是根据这些数据**自动生成**的。

每次有内容推送到 `main` 分支，GitHub Action 都会重新生成 `README.md` 并覆盖它。所以如果你直接改
`README.md`，下次构建时你的改动会被冲掉。**要改内容或版式，请改 YAML 文件或生成脚本，绝不要直接改
README。**

Day to day you mostly touch two things: **issue forms** (to add or remove a
single paper) and **the Actions tab** (to scan many papers at once).

日常你主要用两样东西：**Issue 表单**（增删单篇论文）和 **Actions 标签页**（一次扫很多论文）。

---

## 1. One-time setup: API keys / 一次性设置：API 密钥

This step is optional but recommended. Without these keys the automation still
runs, but it does less: it will not auto-classify papers, auto-fill the venue,
or filter out off-topic papers — it will just list candidates for you to review.

这一步可选，但建议做。不配这些密钥自动化也能跑，只是功能少一些：它不会自动分类论文、不会自动补会议名、
也不会过滤掉不相关的论文——只会把候选列出来让你自己看。

On the GitHub website:

1. Open the repo → **Settings** (top bar) → left menu **Secrets and variables**
   → **Actions**.
2. Click **New repository secret** and add:
   - `DEEPSEEK_API_KEY` — turns on auto-classify, venue lookup by AI, off-topic
     filtering, and auto-drafted benchmark summaries.
   - `TAVILY_API_KEY` **or** `SERPER_API_KEY` — turns on the web-search path for
     finding a paper's venue.
3. Save. Nothing needs to be restarted.

在 GitHub 网页上：

1. 打开仓库 → **Settings**（顶栏）→ 左侧菜单 **Secrets and variables** → **Actions**。
2. 点 **New repository secret**，添加：
   - `DEEPSEEK_API_KEY` —— 开启自动分类、用 AI 查会议名、过滤不相关论文、自动起草 benchmark 摘要。
   - `TAVILY_API_KEY` **或** `SERPER_API_KEY` —— 开启用网页搜索来查论文会议名这条路径。
3. 保存即可，不需要重启任何东西。

> Never paste a key into an issue, a comment, or the code. It only belongs on
> this Secrets page.
>
> 千万别把密钥贴进 issue、评论或代码里。它只应该放在这个 Secrets 页面。

---

## 2. Add a paper / 添加一篇论文

This is the most common task.

这是最常用的操作。

1. Repo → **Issues** → **New issue**.
2. Choose the **📄 Add a paper** template (click **Get started**).
3. Fill in:
   - **arXiv link or paper title** — paste the arXiv link/id (most reliable), or
     just paste the title and let the bot look it up.
   - **Category** — leave it on `Auto (let the bot classify)`, or pick one
     yourself.
   - **Venue (optional)** — leave blank to fill it in automatically. Fill it in
     to override, for example a brand-new acceptance like `KDD'26` that cannot
     be found online yet.
   - **Why (optional)** — a sentence or two for a human to sanity-check.
4. Click **Submit new issue**.

1. 仓库 → **Issues** → **New issue**。
2. 选 **📄 Add a paper** 模板（点 **Get started**）。
3. 填写：
   - **arXiv link or paper title** —— 贴 arXiv 链接/编号（最可靠），或直接贴标题让机器人去查。
   - **Category** —— 保持 `Auto (let the bot classify)` 即可，或自己选一个分类。
   - **Venue (optional)** —— 留空则自动填。想覆盖就填，比如刚被接收、网上还查不到的 `KDD'26`。
   - **Why (optional)** —— 一两句话，方便人工核对。
4. 点 **Submit new issue**。

What happens next: a GitHub Action looks the paper up, fills in the venue,
classifies it, adds it to `data/papers.yaml`, rebuilds the README, pushes, then
comments the result on your issue and closes it.

之后会发生：GitHub Action 查到论文、补上会议名、分类、加进 `data/papers.yaml`、重新生成 README、推送，
然后在你的 issue 里回复结果并关闭它。

- Success: comment `✅ Added …`, issue closed. / 成功：回复 `✅ Added …`，issue 关闭。
- Already there: comment `duplicate`, issue closed (nothing added twice). / 已存在：回复 `duplicate`，issue 关闭（不会重复添加）。
- Failure: the issue **stays open** and the comment explains why → see Section 8. / 失败：issue **保持打开**，回复里说明原因 → 见第 8 节。

---

## 3. Remove a paper / 删除一篇论文

1. **Issues → New issue → 🗑️ Remove a paper**.
2. **arXiv id or paper title** — **prefer the arXiv id.**
3. Submit.

1. **Issues → New issue → 🗑️ Remove a paper**。
2. **arXiv id or paper title** —— **优先填 arXiv 编号。**
3. 提交。

What happens:

- Exactly one match → it is deleted, the README is rebuilt, pushed, and the
  issue is closed.
- The title matches **more than one paper** → the removal is **refused** on
  purpose. The bot lists the candidates and asks you to re-open with the arXiv
  id, so only one paper can ever be deleted at a time.
- No match → comment `not_found`.

会发生：

- 正好命中一篇 → 删除、重新生成 README、推送、关闭 issue。
- 标题**匹配到多篇** → **故意拒绝删除**。机器人会列出候选，让你改用 arXiv 编号重新提交，保证一次只会删一篇。
- 没匹配到 → 回复 `not_found`。

> The remove form only works on `data/papers.yaml`. Benchmarks and surveys are
> removed by hand (see Section 6).
>
> 删除表单只作用于 `data/papers.yaml`。benchmarks 和 surveys 需要手动删除（见第 6 节）。

---

## 4. Collect many papers at once / 一次性采集很多论文

There are three ways to gather candidates, all under the **Actions** tab. In the
Actions tab: pick a workflow on the left → click **Run workflow** on the right →
fill in the inputs → click the green **Run workflow**.

有三种批量收集候选的方式，都在 **Actions** 标签页里。操作方法：在 Actions 页左侧选一个 workflow → 右侧点
**Run workflow** → 填参数 → 点绿色的 **Run workflow**。

**Important:** these three only *find* papers — they do **not** add anything
automatically. When a run finishes it opens a new issue containing the list of
candidates. You look through it, and for the ones you want, you add them using
the Add form in Section 2. When you are done, close that issue yourself.

**重要：** 这三个只*查找*论文，**不会**自动添加任何东西。跑完后会新开一个 issue，里面是候选列表。你看一遍，
想要的就用第 2 节的添加表单加进去。处理完后，自己把那个 issue 关掉。

| Goal / 目的 | Workflow | Inputs / 参数 |
|---|---|---|
| Scan one conference / 扫一个会议 | **Scan a venue** | `venue` (e.g. ICDE), `year` (e.g. 2026), `source` = auto / dblp / openreview |
| Scan a date range / 扫一个时间段 | **Weekly arXiv digest** | `from` / `to` (YYYY-MM-DD); leave blank for the last 8 days |
| Backfill from another list / 从别的列表回补 | **Harvest a sibling list** | `sources` = space-separated `owner/name` or README URLs |

Notes / 说明:

- **Weekly arXiv digest** also runs on a schedule (every Monday, 01:00 UTC), so
  each week you automatically get a new issue with candidates to review.
  **Weekly arXiv digest** 还会定时运行（每周一 01:00 UTC），所以你每周会自动收到一个候选 issue。
- Each workflow has a `no_triage` switch: turn it on to skip the AI relevance
  filter and list everything that matches the topic. (If `DEEPSEEK_API_KEY` is
  not set, the filter is skipped anyway.)
  每个 workflow 都有 `no_triage` 开关：打开它就跳过 AI 相关性过滤，把所有 topic 命中的都列出来。（如果没设
  `DEEPSEEK_API_KEY`，本来也会自动跳过过滤。）
- The default harvest sources (NL2SQL_Handbook, Awesome-Text2SQL) contain many
  older, pre-LLM papers that are out of scope here — the filter drops most of
  them, so do not blindly import everything.
  Harvest 默认的来源（NL2SQL_Handbook、Awesome-Text2SQL）含大量较早、LLM 之前的论文，超出本列表范围——
  过滤会去掉大部分，所以别无脑全导入。

---

## 5. Change the README layout, sections, or intro / 修改 README 版式、分区或介绍

Everything about how the README looks is decided by
`scripts/generate_readme.py`. To edit it on the website: open the file → click
the **pencil ✏️** at the top right → make your change → **Commit changes** (commit
straight to `main`). The build will rebuild the README for you.

README 的所有外观都由 `scripts/generate_readme.py` 决定。在网页上编辑：打开该文件 → 点右上角的**铅笔 ✏️** →
改动 → **Commit changes**（直接提交到 `main`）。构建会自动重新生成 README。

Where to change what / 改哪里:

- **Section order / titles / blurbs** → the `CATEGORIES` list. / **分区顺序、标题、说明** → `CATEGORIES` 列表。
- **Top banner / badges** → `HEADER`; footer → `FOOTER`. / **顶部横幅、徽章** → `HEADER`；页脚 → `FOOTER`。
- **Table columns / sort order** → `render_papers()` (column defs and `sort_key`). / **表格列、排序** → `render_papers()`（列定义和 `sort_key`）。
- **Intro paragraph and taxonomy** → the `INTRO` value. / **开头介绍和分类说明** → `INTRO`。
- **⭐ Must-Read highlights** → set `featured: true` on a paper entry in
  `papers.yaml` to promote it into the highlights section. / **⭐ 精选（Must-Read）** →
  在 `papers.yaml` 里给某篇加一行 `featured: true`，就把它提升进精选区。
- **Community & Resources (people, workshops, related repos)** →
  `data/resources.yaml`. / **社区与资源（学者、研讨会、相关仓库）** → `data/resources.yaml`。

> Adding a brand-new category is more involved: you must update `CATEGORIES` in
> `generate_readme.py` **and** the category lists in `add_paper.py`,
> `llm_triage.py`, `curate.py`, `add_from_issue.py`, `scan_venue.py`, **and** the
> dropdown in the `add-paper.yml` issue form — otherwise papers in the new
> category have nowhere to go. This is easy to get wrong; ask Claude to do it.
>
> 新增一个分区要改多处：`generate_readme.py` 的 `CATEGORIES`，**加上** `add_paper.py`、`llm_triage.py`、
> `curate.py`、`add_from_issue.py`、`scan_venue.py` 里的分类表，**再加上** `add-paper.yml` 表单的下拉选项——
> 否则新分区的论文无处安放。容易漏，建议交给 Claude 做。

---

## 6. Editing the YAML by hand / 手动编辑 YAML

Benchmarks and surveys have no issue form — edit their files directly.

benchmarks 和 surveys 没有 issue 表单——直接编辑它们的文件。

- On the website: open `data/benchmarks.yaml` → ✏️ edit → add or remove an entry
  following the same indentation and fields as the existing ones → commit to
  `main`.
- Locally, a benchmark can also be added with
  `python scripts/add_benchmark.py <arxiv-id>` (it drafts a short summary and
  inserts it into the right group).

- 网页上：打开 `data/benchmarks.yaml` → ✏️ 编辑 → 照已有条目的缩进和字段格式增删 → 提交到 `main`。
- 本地也可以用 `python scripts/add_benchmark.py <arxiv-id>` 添加 benchmark（会起草一段简短摘要并插进对应分组）。

After committing, the build rebuilds the README automatically.

提交后，构建会自动重新生成 README。

---

## 7. Local maintenance (optional) / 本地维护（可选）

Faster when your network is good. From the repo folder:

网络好的时候更快。在仓库目录下：

```bash
cd ~/desktop/Data-Agent-Reading-List

# Add / 添加:
python scripts/add_paper.py <arxiv-id> --category nl2sql     # have the id / 有编号
python scripts/add_by_title.py "DIN-SQL: ..."                # title only / 只有标题

# Remove / 删除:
python scripts/remove_paper.py <arxiv-id-or-title> --dry-run # preview / 先看命中什么
python scripts/remove_paper.py <arxiv-id>                    # actually delete / 真删

# Collect / 采集:
python scripts/scan_venue.py --venue ICDE --year 2026 --source auto
python scripts/curate.py --from 2026-06-01 --to 2026-07-01
python scripts/harvest_list.py HKUSTDial/NL2SQL_Handbook

# Check the README matches the data (CI runs this on pull requests):
# 检查 README 与数据是否一致（CI 会在 pull request 上跑这个）:
python scripts/generate_readme.py --check
```

Then `git add data/ && git commit && git push`.

然后 `git add data/ && git commit && git push`。

> Commit rule: do **not** add a `Co-Authored-By: Claude` trailer to commits in
> this repo.
>
> 提交规则：本仓库的提交**不要**加 `Co-Authored-By: Claude` 尾注。

---

## 8. When something fails / 出错时怎么排查

1. Go to the **Actions** tab → find the failed run (red ✗) → open it → expand
   the red step to read the log.
2. A failed issue task shows up as: the issue **stays open** and the bot's
   comment explains why.
3. Common causes:
   - Venue could not be found → the entry is saved as
     `arXiv'YY.MM  # TODO verify venue`; you can re-fill it later with
     `backfill_venues.py` (run it in CI with the keys set — this China-based
     machine often cannot reach the lookup services directly).
   - Classification / filtering did nothing → usually `DEEPSEEK_API_KEY` is not
     set (Section 1).
   - A pull request is blocked because the README is out of date → run
     `generate_readme.py` locally and commit the result.
4. After fixing: edit the failed issue and submit again (an edit re-triggers the
   workflow), or just open a fresh one.

1. 打开 **Actions** 标签页 → 找到失败的 run（红 ✗）→ 点进去 → 展开报红的步骤看日志。
2. Issue 任务失败的标志：issue **保持打开**，机器人的回复说明了原因。
3. 常见原因：
   - 查不到会议名 → 条目会存成 `arXiv'YY.MM  # TODO verify venue`；之后可用 `backfill_venues.py` 重新补
     （在配好密钥的 CI 里跑——这台国内机器常常直连不上查询服务）。
   - 分类/过滤没生效 → 通常是没设 `DEEPSEEK_API_KEY`（第 1 节）。
   - pull request 因 README 过期被卡 → 本地跑 `generate_readme.py` 并提交结果。
4. 修好后：编辑失败的那个 issue 再提交一次（编辑会重新触发 workflow），或者干脆新开一个。

---

## Quick reference / 速查

| I want to… / 我想… | Where / 去哪 |
|---|---|
| Add one paper / 加一篇 | Issues → 📄 Add a paper |
| Remove one paper / 删一篇 | Issues → 🗑️ Remove a paper |
| Scan a conference / 扫一个会议 | Actions → Scan a venue |
| Scan a date range / 扫一个时间段 | Actions → Weekly arXiv digest (fill in from/to) |
| Backfill from another list / 从别的列表回补 | Actions → Harvest a sibling list |
| Change layout / intro / 改版式、介绍 | Edit `scripts/generate_readme.py` |
| Change highlights / 改精选 | Add `featured: true` in `papers.yaml` |
| Change people / resources / 改学者、资源 | `data/resources.yaml` |
| Add/remove a benchmark or survey / 增删 benchmark、survey | Edit `data/*.yaml` by hand |
