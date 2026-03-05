# GPT Paper Assistant: A Daily ArXiv Scanner

A daily ArXiv scanner that uses an LLM to find papers matching your research interests. Runs via GitHub Actions and can post results to Slack, Google Chat, or a static GitHub Pages website. Supports both OpenAI models and self-hosted local models via [vLLM](https://github.com/vllm-project/vllm).

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

### Running on GitHub Actions

1. Fork this repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).
2. Copy `configs/paper_topics.template.txt` → `configs/paper_topics.txt` and describe the topics you want to follow (see [Writing paper_topics.txt](#writing-paper_topicstxt)).
3. Set your ArXiv categories in `configs/config.ini` under `arxiv_category`.
4. Set `OAI_KEY` as a [GitHub secret](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions).
5. In repo Settings → Pages, set the build source to [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).

The bot runs daily at 1 pm UTC and publishes a static website. Trigger it manually via the Actions tab to test.

**Optional — Slack:**
- [Set up a Slack bot](https://api.slack.com/start/quickstart) and set `SLACK_KEY` and `SLACK_CHANNEL_ID` as GitHub secrets.
- Set `push_to_slack = true` in `configs/config.ini`.

**Optional — Google Chat:**
- Create a Chat space and get the webhook URL from *Space settings → Apps & integrations → Webhooks*.
- Set `WEBHOOK_URL` as a GitHub secret and `push_to_google = true` in `configs/config.ini`.

**Optional — keep alive:**
Set the repo to private to prevent GitHub from [disabling scheduled workflows after 60 days of inactivity](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).

### Running Locally

```bash
pip install -r requirements.txt

# OpenAI mode
export OAI_KEY=<your-openai-key>
python main.py

# Local LLM mode (set use_local_llm = true in configs/config.ini)
python main.py
```

**Self-hosted cron:**
```
0 13 * * * cd ~/gpt_paper_assistant && python main.py
```

---

## Configuration (`configs/config.ini`)

```ini
[SELECTION]
run_llm = true                  # set to false to skip LLM calls entirely (debug/dry-run)
use_local_llm = false           # set to true to use a local vLLM server instead of OpenAI
local_llm_url = http://192.168.191.34:8000/v1
local_llm_model = Qwen/Qwen3.5-9B   # model name reported by GET /v1/models
model = gpt-5-mini              # OpenAI model (used when use_local_llm = false)
batch_size = 5                  # papers per LLM batch (larger = cheaper, less accurate)
openai_workers = 4              # parallel LLM requests for title-filter + scoring stages

[FILTERING]
arxiv_category = cs.CV,cs.LG,cs.RO  # comma-separated ArXiv categories
force_primary = true            # ignore papers only cross-listed into these categories
relevance_cutoff = 5            # minimum relevance score (1–10) to include a paper

[OUTPUT]
debug_messages = true
dump_debug_file = true          # writes papers.debug.json and gpt_paper_batches.debug.json
output_path = out/
dump_json = true                # writes output.json (or output_local.json in local mode)
dump_md = true                  # writes output.md (or output_local.md in local mode)
push_to_slack = false
push_to_google = false
```

### Output file naming

| Mode | JSON | Markdown |
|------|------|----------|
| OpenAI (`use_local_llm = false`) | `out/output.json` | `out/output.md` |
| Local LLM (`use_local_llm = true`) | `out/output_local.json` | `out/output_local.md` |

---

## Local LLM Mode (vLLM)

Set `use_local_llm = true` and point `local_llm_url` at your vLLM server. No `OAI_KEY` is needed.

The local LLM call uses these sampling parameters:

| Parameter | Value |
|-----------|-------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 1.5 |
| repetition_penalty | 1.0 |

`<think>...</think>` blocks (produced by reasoning models like Qwen3) are automatically stripped before JSON parsing.

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

Thu Mar  5 04:11:01 AM EST 2026
