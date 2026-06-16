#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/local_publish.sh [--dry-run|--push] [--skip-main]

Runs the paper scanner against a local OpenAI-compatible LLM endpoint, stages
the generated Markdown as a GitHub Pages site, and optionally pushes it to the
gh-pages branch.

Environment:
  GPTPA_LOCAL_LLM_URL       Local /v1 endpoint. Default: http://127.0.0.1:10531/v1
  GPTPA_LOCAL_LLM_MODEL     Model id to send to chat.completions. Prefers gpt-5.5 when available.
  GPTPA_LOCAL_LLM_REASONING_EFFORT  Reasoning effort for GPT-style endpoints. Default: xhigh
  LOCAL_LLM_API_KEY         API key for local endpoint if it requires one.
  GPTPA_LOCAL_LLM_API_KEY   Same as LOCAL_LLM_API_KEY, takes precedence.
  GPTPA_OPENAI_WORKERS     Concurrent local LLM requests. Default: 4
  GPTPA_TITLE_FILTER_BATCH_SIZE  Optional titles per title-filter call. Default: config value
  GPTPA_BATCH_SIZE          Optional papers per full abstract call. Default: config value
  GPTPA_TITLE_FILTER_REASONING_EFFORT  Optional reasoning effort for title filtering. Default: empty
  GPTPA_SITE_DIR            Staging directory. Default: .local_site
  GPTPA_PAGES_BRANCH        Publish branch. Default: gh-pages
  GPTPA_PAGES_REMOTE        Git remote. Default: origin
  PYTHON                    Python executable. Default: python3

Examples:
  scripts/local_publish.sh --dry-run
  scripts/local_publish.sh --push
USAGE
}

publish=0
run_main=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      publish=0
      ;;
    --push)
      publish=1
      ;;
    --skip-main)
      run_main=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
remote="${GPTPA_PAGES_REMOTE:-origin}"
pages_branch="${GPTPA_PAGES_BRANCH:-gh-pages}"
site_dir="${GPTPA_SITE_DIR:-$repo_root/.local_site}"
worktree_dir="${GPTPA_PAGES_WORKTREE:-$repo_root/.local_pages_worktree}"

export GPTPA_USE_LOCAL_LLM="${GPTPA_USE_LOCAL_LLM:-true}"
export GPTPA_LOCAL_LLM_URL="${GPTPA_LOCAL_LLM_URL:-http://127.0.0.1:10531/v1}"
export GPTPA_PUSH_TO_GOOGLE="${GPTPA_PUSH_TO_GOOGLE:-false}"
export GPTPA_PUSH_TO_SLACK="${GPTPA_PUSH_TO_SLACK:-false}"
export GPTPA_DUMP_DEBUG_FILE="${GPTPA_DUMP_DEBUG_FILE:-false}"
export GPTPA_OPENAI_WORKERS="${GPTPA_OPENAI_WORKERS:-4}"
export GPTPA_LOCAL_LLM_REQUEST_STYLE="${GPTPA_LOCAL_LLM_REQUEST_STYLE:-openai}"
export GPTPA_LOCAL_LLM_REASONING_EFFORT="${GPTPA_LOCAL_LLM_REASONING_EFFORT:-xhigh}"
export GPTPA_TITLE_FILTER_REASONING_EFFORT="${GPTPA_TITLE_FILTER_REASONING_EFFORT:-}"

api_key="${GPTPA_LOCAL_LLM_API_KEY:-${LOCAL_LLM_API_KEY:-${OPENAI_API_KEY:-${OAI_KEY:-EMPTY}}}}"

if [ -z "${GPTPA_LOCAL_LLM_MODEL:-}" ]; then
  models_json="$(curl -fsS -H "Authorization: Bearer $api_key" "$GPTPA_LOCAL_LLM_URL/models" 2>/dev/null || true)"
  detected_model="$(
    printf '%s' "$models_json" |
      "$python_bin" -c 'import json, sys; data=json.load(sys.stdin); ids=[item.get("id", "") for item in data.get("data", [])]; print("gpt-5.5" if "gpt-5.5" in ids else (ids[0] if ids else ""))' 2>/dev/null || true
  )"
  if [ -n "$detected_model" ]; then
    export GPTPA_LOCAL_LLM_MODEL="$detected_model"
    echo "Detected local LLM model: $GPTPA_LOCAL_LLM_MODEL"
  else
    echo "Could not auto-detect model id; using configs/config.ini local_llm_model." >&2
  fi
fi

if [ "$run_main" -eq 1 ]; then
  "$python_bin" main.py
fi

source_md="$repo_root/out/output_local.md"
if [ ! -s "$source_md" ]; then
  echo "Expected generated Markdown at $source_md, but it is missing or empty." >&2
  exit 1
fi

mkdir -p "$site_dir"
{
  printf -- '---\n'
  printf 'title: Arxiv Daily\n'
  printf -- '---\n\n'
  cat "$source_md"
} > "$site_dir/index.md"

cat > "$site_dir/_config.yml" <<'CONFIG'
title: Arxiv Daily
markdown: kramdown
CONFIG

echo "Staged GitHub Pages source in $site_dir"

if [ "$publish" -eq 0 ]; then
  echo "Dry run complete. Re-run with --push to commit and push $pages_branch."
  exit 0
fi

if git ls-remote --exit-code --heads "$remote" "$pages_branch" >/dev/null 2>&1; then
  git fetch "$remote" "$pages_branch:$pages_branch"
  if [ ! -e "$worktree_dir/.git" ]; then
    git worktree add "$worktree_dir" "$pages_branch"
  fi
else
  if git show-ref --verify --quiet "refs/heads/$pages_branch"; then
    if [ ! -e "$worktree_dir/.git" ]; then
      git worktree add "$worktree_dir" "$pages_branch"
    fi
  else
    git worktree add --orphan -b "$pages_branch" "$worktree_dir"
  fi
fi

find "$worktree_dir" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$site_dir"/. "$worktree_dir"/

git -C "$worktree_dir" add -A
if git -C "$worktree_dir" diff --cached --quiet; then
  echo "No website changes to publish."
  exit 0
fi

commit_date="$(date +%Y-%m-%d)"
git -C "$worktree_dir" commit -m "Update daily paper site $commit_date"
git -C "$worktree_dir" push "$remote" "$pages_branch:$pages_branch"

echo "Published $pages_branch to $remote."
