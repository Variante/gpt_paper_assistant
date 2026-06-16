import configparser
import dataclasses
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import EnhancedJSONEncoder, Paper, get_papers_from_arxiv_rss_api
from filter_papers import filter_by_gpt
from parse_json_to_md import render_md_string


ENV_CONFIG_OVERRIDES = {
    ("SELECTION", "run_llm"): "GPTPA_RUN_LLM",
    ("SELECTION", "use_local_llm"): "GPTPA_USE_LOCAL_LLM",
    ("SELECTION", "local_llm_url"): "GPTPA_LOCAL_LLM_URL",
    ("SELECTION", "local_llm_model"): "GPTPA_LOCAL_LLM_MODEL",
    ("SELECTION", "local_llm_request_style"): "GPTPA_LOCAL_LLM_REQUEST_STYLE",
    ("SELECTION", "local_llm_reasoning_effort"): "GPTPA_LOCAL_LLM_REASONING_EFFORT",
    ("SELECTION", "title_filter_reasoning_effort"): "GPTPA_TITLE_FILTER_REASONING_EFFORT",
    ("SELECTION", "batch_size"): "GPTPA_BATCH_SIZE",
    ("SELECTION", "title_filter_batch_size"): "GPTPA_TITLE_FILTER_BATCH_SIZE",
    ("SELECTION", "openai_workers"): "GPTPA_OPENAI_WORKERS",
    ("FILTERING", "arxiv_category"): "GPTPA_ARXIV_CATEGORY",
    ("FILTERING", "force_primary"): "GPTPA_FORCE_PRIMARY",
    ("FILTERING", "relevance_cutoff"): "GPTPA_RELEVANCE_CUTOFF",
    ("OUTPUT", "debug_messages"): "GPTPA_DEBUG_MESSAGES",
    ("OUTPUT", "dump_debug_file"): "GPTPA_DUMP_DEBUG_FILE",
    ("OUTPUT", "debug_input_file"): "GPTPA_DEBUG_INPUT_FILE",
    ("OUTPUT", "output_path"): "GPTPA_OUTPUT_PATH",
    ("OUTPUT", "dump_json"): "GPTPA_DUMP_JSON",
    ("OUTPUT", "dump_md"): "GPTPA_DUMP_MD",
    ("OUTPUT", "push_to_slack"): "GPTPA_PUSH_TO_SLACK",
    ("OUTPUT", "push_to_google"): "GPTPA_PUSH_TO_GOOGLE",
}


def apply_env_overrides(config: configparser.ConfigParser) -> None:
    for (section, option), env_name in ENV_CONFIG_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is not None:
            config[section][option] = value


def load_config(path: str | None = None) -> configparser.ConfigParser:
    path = path or os.environ.get("GPTPA_CONFIG", "configs/config.ini")
    config = configparser.ConfigParser()
    config.read(path)
    apply_env_overrides(config)
    return config


def build_openai_client(config: configparser.ConfigParser) -> tuple[OpenAI, str]:
    if config["SELECTION"].getboolean("use_local_llm"):
        print(f"Using local LLM at {config['SELECTION']['local_llm_url']}")
        config["SELECTION"]["model"] = config["SELECTION"]["local_llm_model"]
        api_key = (
            os.environ.get("GPTPA_LOCAL_LLM_API_KEY")
            or os.environ.get("LOCAL_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OAI_KEY")
            or "EMPTY"
        )
        return (
            OpenAI(base_url=config["SELECTION"]["local_llm_url"], api_key=api_key),
            "output_local",
        )

    api_key = os.environ.get("OAI_KEY")
    if not api_key:
        raise ValueError("OpenAI key is not set - please set OAI_KEY to your OpenAI key")
    return OpenAI(api_key=api_key), "output"


def load_debug_papers(path: str) -> list[Paper]:
    with open(path, "r") as handle:
        raw = json.load(handle)

    if raw and isinstance(raw[0], list):
        # Support legacy list-of-lists format.
        raw = [item for sublist in raw for item in sublist]

    paper_fields = {field.name for field in dataclasses.fields(Paper)}
    return [Paper(**{k: v for k, v in item.items() if k in paper_fields}) for item in raw]


def scrape_papers(config: configparser.ConfigParser) -> list[Paper]:
    areas = [area.strip() for area in config["FILTERING"]["arxiv_category"].split(",") if area.strip()]
    papers_by_id: dict[str, Paper] = {}

    for area in tqdm(areas, desc="Scraping arxiv categories"):
        try:
            for paper in get_papers_from_arxiv_rss_api(area, config):
                papers_by_id[paper.arxiv_id] = paper
        except Exception as exc:
            tqdm.write(f"Error scraping {area}: {exc}")

    tqdm.write(f"Total papers scraped: {len(papers_by_id)}")
    return list(papers_by_id.values())


def sort_selected_papers(selected_papers: dict[str, dict], sort_dict: dict[str, int]) -> dict[str, dict]:
    sorted_ids = sorted(sort_dict, key=lambda arxiv_id: sort_dict[arxiv_id], reverse=True)
    return {arxiv_id: selected_papers[arxiv_id] for arxiv_id in sorted_ids if arxiv_id in selected_papers}


def write_outputs(
    selected_papers: dict[str, dict],
    config: configparser.ConfigParser,
    output_stem: str,
) -> None:
    output_dir = Path(config["OUTPUT"]["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["OUTPUT"].getboolean("dump_json"):
        with open(output_dir / f"{output_stem}.json", "w") as handle:
            json.dump(selected_papers, handle, indent=4)

    if config["OUTPUT"].getboolean("dump_md"):
        with open(output_dir / f"{output_stem}.md", "w") as handle:
            handle.write(render_md_string(selected_papers))

    if config["OUTPUT"].getboolean("push_to_slack"):
        if os.environ.get("SLACK_KEY"):
            try:
                from push_to_slack import push_to_slack

                push_to_slack(selected_papers)
            except ModuleNotFoundError as exc:
                tqdm.write(f"Warning: slack dependencies missing ({exc}) - not pushing to slack")
        else:
            tqdm.write("Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack")

    if config["OUTPUT"].getboolean("push_to_google"):
        if os.environ.get("WEBHOOK_URL"):
            from push_to_google_chat import push_to_google_chat

            push_to_google_chat(selected_papers)
        else:
            tqdm.write("Warning: push_to_google is true, but WEBHOOK_URL is not set - not pushing to google chat")


def main() -> None:
    config = load_config()
    openai_client, output_stem = build_openai_client(config)

    steps = ["scrape", "filter", "output"]
    with tqdm(total=len(steps), desc="Pipeline", unit="step") as progress:
        debug_input = config["OUTPUT"].get("debug_input_file", "").strip()
        if debug_input:
            tqdm.write(f"Loading papers from {debug_input}")
            papers = load_debug_papers(debug_input)
        else:
            progress.set_postfix(step="scraping arxiv")
            papers = scrape_papers(config)
            if config["OUTPUT"].getboolean("dump_debug_file"):
                output_dir = Path(config["OUTPUT"]["output_path"])
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "papers.debug.json", "w") as handle:
                    json.dump(papers, handle, cls=EnhancedJSONEncoder, indent=4)
        progress.update(1)

        progress.set_postfix(step="LLM filtering")
        selected_papers, sort_dict = filter_by_gpt(papers, config, openai_client)
        sorted_papers = sort_selected_papers(selected_papers, sort_dict)
        tqdm.write(f"Selected {len(sorted_papers)} papers out of {len(papers)}")
        progress.update(1)

        progress.set_postfix(step="writing output")
        if papers:
            write_outputs(sorted_papers, config, output_stem)
        progress.update(1)
    print(sort_dict)
    print(sorted_papers)


if __name__ == "__main__":
    main()
