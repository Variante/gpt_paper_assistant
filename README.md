# GPT Paper Assistant: A Daily ArXiv Scanner

A daily ArXiv scanner that uses an LLM to find papers matching your research interests. It can run in GitHub Actions or fully locally, then publish results to Slack, Google Chat, or a static GitHub Pages website. Supports OpenAI models, GPT-style local OpenAI-compatible endpoints, and self-hosted models via [vLLM](https://github.com/vllm-project/vllm).

A live demo running on `cs.CV,cs.LG,cs.RO` is available [here](https://variante.github.io/gpt_paper_assistant/).

---

## How It Works

1. **Scrape** — fetch new papers from ArXiv RSS feeds for the configured categories (e.g. `cs.CV,cs.LG,cs.RO`). Only new submissions are announced; updated papers are skipped.
2. **Pre-filter** — a fast LLM call with just titles removes obviously irrelevant papers to reduce cost.
3. **Score** — remaining papers are batched and sent to the LLM with your topic criteria. The model returns a relevance score (1–10) and a comment for each matching paper. Output is enforced as a JSON object via `response_format={"type": "json_object"}`.
4. **Select** — papers with relevance ≥ `relevance_cutoff` are kept and sorted by score.
5. **Output** — results are written to JSON/Markdown and optionally pushed to Slack or Google Chat.

---

## Quickstart

### 1. Configure Paper Selection

1. Copy `configs/paper_topics.template.txt` to `configs/paper_topics.txt` and describe the topics you want to follow (see [Writing paper_topics.txt](#writing-paper_topicstxt)).
2. Set your ArXiv categories in `configs/config.ini` under `arxiv_category`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Choose A Deployment Mode

This project supports two publishing paths:

| Mode | Where the LLM runs | How the website is published | Best for |
|------|--------------------|------------------------------|----------|
| GitHub Actions deploy | GitHub-hosted runner | GitHub Actions Pages deploy | OpenAI API or cloud-accessible LLMs |
| Local deploy | Your machine | Push generated site to `gh-pages` | Local-only LLMs on `127.0.0.1` or a private LAN |

Use only one mode for scheduled production runs. If local deploy is active, the Actions deploy workflows can stay disabled.

---

## Deployment: GitHub Actions

The original cloud deployment runs the scanner in GitHub Actions and deploys Pages from an uploaded artifact.

1. Fork this repo and enable scheduled workflows.
2. Put the workflow files under `.github/workflows/`. This repo keeps the old workflows under `.github/disabled-workflows/`; move them back if you want the Actions pipeline:

```bash
mkdir -p .github/workflows
mv .github/disabled-workflows/cron_runs.yaml .github/workflows/
mv .github/disabled-workflows/publish_md_test.yml .github/workflows/
# optional keep-alive workflow
mv .github/disabled-workflows/keep_alive.yml .github/workflows/
```

3. Set `OAI_KEY` as a GitHub secret. If Slack or Google Chat is enabled, also set `SLACK_KEY`, `SLACK_CHANNEL_ID`, or `WEBHOOK_URL` as needed.
4. In GitHub repo settings, set Pages to:

```text
Settings -> Pages -> Build and deployment
Source: GitHub Actions
```

5. Trigger `Run daily arxiv` manually from the Actions tab to test.

Pipeline:

```text
cron_runs.yaml
  -> installs dependencies
  -> runs python main.py
  -> uploads out/ as arxiv-scanner-outputs

publish_md_test.yml
  -> downloads arxiv-scanner-outputs
  -> converts output.md to a Pages artifact
  -> deploys with actions/deploy-pages
```

**Optional — Slack:**
- [Set up a Slack bot](https://api.slack.com/start/quickstart) and set `SLACK_KEY` and `SLACK_CHANNEL_ID` as GitHub secrets.
- Set `push_to_slack = true` in `configs/config.ini`.

**Optional — Google Chat:**
- Create a Chat space and get the webhook URL from *Space settings -> Apps & integrations -> Webhooks*.
- Set `WEBHOOK_URL` as a GitHub secret and `push_to_google = true` in `configs/config.ini`.

---

## Deployment: Local LLM To GitHub Pages

Use this mode when the LLM endpoint is only available from your machine, for example `http://127.0.0.1:10531/v1`.

1. Configure GitHub Pages to deploy from the generated branch:

```text
Settings -> Pages -> Build and deployment
Source: Deploy from a branch
Branch: gh-pages
Folder: / (root)
```

2. Keep the old Actions workflows disabled by leaving them outside `.github/workflows/`.

3. Put local secrets in an untracked file such as `var.sh`:

```bash
export GPTPA_LOCAL_LLM_URL="http://127.0.0.1:10531/v1"
export GPTPA_LOCAL_LLM_API_KEY="your-local-api-key"
```

`var.sh` is ignored by git because `*.sh` is ignored except for `scripts/*.sh`.

4. Test without publishing:

```bash
set -a && source var.sh && set +a
PYTHON=/home/xiangli/miniconda3/envs/paper/bin/python scripts/local_publish.sh --dry-run
```

The dry run scrapes ArXiv, calls the local LLM, writes `out/output_local.md`, and stages `.local_site/index.md`. It does not commit or push.

5. Publish the generated website:

```bash
set -a && source var.sh && set +a
PYTHON=/home/xiangli/miniconda3/envs/paper/bin/python scripts/local_publish.sh --push
```

Pipeline:

```text
local cron or shell
  -> source var.sh
  -> scripts/local_publish.sh --push
  -> main.py scrapes ArXiv
  -> local LLM filters and scores papers
  -> writes out/output_local.md and out/output_local.json
  -> stages .local_site/index.md
  -> commits and pushes gh-pages
  -> GitHub Pages serves gh-pages
```

Useful overrides:

```bash
# Concurrent LLM requests. Default for local_publish.sh is 4.
GPTPA_OPENAI_WORKERS=4 scripts/local_publish.sh --dry-run

# Use a specific model instead of auto-detecting from /v1/models.
GPTPA_LOCAL_LLM_MODEL=gpt-5.5 scripts/local_publish.sh --dry-run

# Use xhigh reasoning for full abstract scoring; title filtering stays fast by default.
GPTPA_LOCAL_LLM_REASONING_EFFORT=xhigh scripts/local_publish.sh --dry-run

# Optional: add reasoning effort to title filtering too, usually slower.
GPTPA_TITLE_FILTER_REASONING_EFFORT=medium scripts/local_publish.sh --dry-run
```

For a scheduled local run, add a cron entry on the machine that can reach the local LLM:

```cron
0 10 * * * cd /path/to/gpt_paper_assistant && set -a && . ./var.sh && set +a && PYTHON=/home/xiangli/miniconda3/envs/paper/bin/python scripts/local_publish.sh --push >> local_publish.log 2>&1
```

---

## Configuration (`configs/config.ini`)

```ini
[SELECTION]
run_llm = true                     # set to false to skip LLM calls entirely (debug/dry-run)
use_local_llm = false              # set to true for a local OpenAI-compatible endpoint
local_llm_url = http://127.0.0.1:10531/v1
local_llm_model = gpt-5.5          # model name reported by GET /v1/models
local_llm_request_style = openai   # openai/gpt/codex for GPT-style proxies; vllm for vLLM sampling params
local_llm_reasoning_effort = xhigh # optional; used for full abstract scoring in GPT-style local mode
title_filter_reasoning_effort =    # optional; leave empty for faster title-only filtering
model = gpt-5-mini                 # OpenAI model used when use_local_llm = false
batch_size = 5                     # papers per full-scoring LLM batch
title_filter_batch_size = 20       # titles per title-filter request
openai_workers = 4                 # parallel LLM requests for title-filter + scoring stages

[FILTERING]
arxiv_category = cs.CV,cs.LG,cs.RO # comma-separated ArXiv categories
force_primary = true               # ignore papers only cross-listed into these categories
relevance_cutoff = 5               # minimum relevance score (1-10) to include a paper

[OUTPUT]
debug_messages = true
dump_debug_file = true             # writes papers.debug.json and gpt_paper_batches.debug.json
output_path = out/
dump_json = true                   # writes output.json or output_local.json
dump_md = true                     # writes output.md or output_local.md
push_to_slack = false
push_to_google = false
```

### Output file naming

| Mode | JSON | Markdown |
|------|------|----------|
| OpenAI (`use_local_llm = false`) | `out/output.json` | `out/output.md` |
| Local LLM (`use_local_llm = true`) | `out/output_local.json` | `out/output_local.md` |

---

## Local LLM Modes

Set `use_local_llm = true` to use an OpenAI-compatible local endpoint. `main.py` writes local-mode results to `out/output_local.json` and `out/output_local.md`.

### GPT-Style Local Proxy

Use this for endpoints that behave like OpenAI chat completions, including local proxy servers exposing models such as `gpt-5.5`:

```ini
use_local_llm = true
local_llm_url = http://127.0.0.1:10531/v1
local_llm_model = gpt-5.5
local_llm_request_style = openai
local_llm_reasoning_effort = xhigh
title_filter_reasoning_effort =
```

Set the API key through the environment, not in `configs/config.ini`:

```bash
export GPTPA_LOCAL_LLM_API_KEY="your-local-api-key"
```

### vLLM

Use this for vLLM servers that accept vLLM-specific sampling parameters:

```ini
use_local_llm = true
local_llm_url = http://192.168.191.34:8000/v1
local_llm_model = Qwen/Qwen3.5-9B
local_llm_request_style = vllm
```

The vLLM path sends these sampling parameters:

| Parameter | Value |
|-----------|-------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 1.5 |
| repetition_penalty | 1.0 |

`<think>...</think>` blocks produced by reasoning models are stripped before JSON parsing.

---

## Comparing OpenAI vs Local LLM

Run both modes to produce their respective output files, then diff them with no API calls:

```bash
# 1. Run with OpenAI  →  out/output.json
use_local_llm = false   # in config.ini
OAI_KEY=<key> python main.py

# 2. Run with local LLM  →  out/output_local.json
use_local_llm = true    # in config.ini
python main.py

# 3. Compare (no API calls)
python compare_llms.py

# Custom paths
python compare_llms.py --openai out/output.json --local out/output_local.json --save out/comparison.json
```

The comparison prints a summary (counts, overlap) and a per-paper breakdown with each model's relevance score and comment, and saves a structured `out/comparison.json`.

---

## Writing `paper_topics.txt`

Number each criterion and include explicit relevant/not-relevant examples — specificity reduces false positives.

```text
1. New methodological improvements to self-supervised learning (SSL) for image or video representation.
   - Relevant: papers proposing or rigorously analyzing specific SSL methods such as
     contrastive learning or masked autoencoders, with a clear focus on representation quality.
   - Not relevant: papers that merely apply a pretrained SSL backbone to a downstream task
     without modifying the SSL method itself.
2. Significant advances in text-to-image or text-to-video diffusion models.
   - Relevant: papers improving generation quality, reducing inference cost, or analyzing
     output fidelity of diffusion models.
   - Not relevant: discrete/language diffusion models or 3D generation work.

In suggesting papers, remember that your friend is primarily interested in self-supervised
learning, computer vision, and robotics. He values methodological novelty over applications.
```

The summary line at the end helps the LLM prioritise when a paper only loosely matches.

---

## LLM Prompt Structure

Three files in `configs/` compose the full prompt:

| File | Role |
|------|------|
| `base_prompt.txt` | Sets the assistant persona: selective, conservative, clear matches only |
| `paper_topics.txt` | Numbered criteria with relevant/not-relevant examples |
| `postfix_prompt.txt` | Enforces output format: `{"papers": [{"ARXIVID", "COMMENT", "RELEVANCE"}]}` |

`response_format={"type": "json_object"}` is passed to the API to hard-enforce valid JSON.

---

## Debugging and Testing

**Test the LLM filter in isolation:**
```bash
# reads in/debug_papers.json → writes out/filter_paper_test.debug.json
python filter_papers.py
```

If the filter makes mistakes, find the relevant batch in `out/gpt_paper_batches.debug.json`, copy it into `in/debug_papers.json`, adjust the prompts, and re-run.

**Debug files written when `dump_debug_file = true`:**
- `out/papers.debug.json` — all scraped papers before filtering
- `out/gpt_paper_batches.debug.json` — LLM-scored batches

---

*Originally built by [Tatsunori Hashimoto](https://github.com/tatsu-lab), licensed under Apache 2.0.*
*Extended with local LLM support, structured JSON output, and comparison tooling.*

