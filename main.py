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


def load_config(path: str = "configs/config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path)
    return config


def build_openai_client(config: configparser.ConfigParser) -> tuple[OpenAI, str]:
    if config["SELECTION"].getboolean("use_local_llm"):
        print(f"Using local LLM at {config['SELECTION']['local_llm_url']}")
        config["SELECTION"]["model"] = config["SELECTION"]["local_llm_model"]
        return (
            OpenAI(base_url=config["SELECTION"]["local_llm_url"], api_key="EMPTY"),
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


if __name__ == "__main__":
    main()
