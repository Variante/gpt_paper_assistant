import dataclasses
import json
import configparser
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import get_papers_from_arxiv_rss_api, EnhancedJSONEncoder, Paper
from filter_papers import filter_by_gpt
from parse_json_to_md import render_md_string
from push_to_slack import push_to_slack
from push_to_google_chat import push_to_google_chat


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_papers_from_arxiv(config):
    area_list = [a.strip() for a in config["FILTERING"]["arxiv_category"].split(",")]
    paper_set = set()
    max_workers = min(len(area_list), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_papers_from_arxiv_rss_api, area, config): area for area in area_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping arxiv categories"):
            area = futures[future]
            try:
                papers = future.result()
                paper_set.update(set(papers))
            except Exception as e:
                tqdm.write(f"Error scraping {area}: {e}")
    tqdm.write(f"Total papers scraped: {len(paper_set)}")
    return paper_set


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")

    if config["SELECTION"].getboolean("use_local_llm"):
        openai_client = OpenAI(
            base_url=config["SELECTION"]["local_llm_url"],
            api_key="EMPTY",
        )
        config["SELECTION"]["model"] = config["SELECTION"]["local_llm_model"]
        output_stem = "output_local"
        print("Using local LLM at " + config["SELECTION"]["local_llm_url"])
    else:
        OAI_KEY = os.environ.get("OAI_KEY")
        if OAI_KEY is None:
            raise ValueError(
                "OpenAI key is not set - please set OAI_KEY to your OpenAI key"
            )
        openai_client = OpenAI(api_key=OAI_KEY)
        output_stem = "output"

    steps = ["scrape", "filter", "output"]
    with tqdm(total=len(steps), desc="Pipeline", unit="step") as pbar:
        debug_input = config["OUTPUT"].get("debug_input_file", "").strip()
        if debug_input:
            tqdm.write(f"Loading papers from {debug_input}")
            with open(debug_input, "r") as f:
                raw = json.load(f)
            # handle both flat list and list-of-lists (legacy format)
            if raw and isinstance(raw[0], list):
                raw = [item for sublist in raw for item in sublist]
            paper_fields = {f.name for f in dataclasses.fields(Paper)}
            papers = [Paper(**{k: v for k, v in d.items() if k in paper_fields}) for d in raw]
        else:
            pbar.set_postfix(step="scraping arxiv")
            papers = list(get_papers_from_arxiv(config))
            if config["OUTPUT"].getboolean("dump_debug_file"):
                with open(
                    config["OUTPUT"]["output_path"] + "papers.debug.json", "w"
                ) as outfile:
                    json.dump(papers, outfile, cls=EnhancedJSONEncoder, indent=4)
        pbar.update(1)

        pbar.set_postfix(step="LLM filtering")
        selected_papers, sort_dict = filter_by_gpt(papers, config, openai_client)
        keys = list(sort_dict.keys())
        values = list(sort_dict.values())
        sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
        selected_papers = {key: selected_papers[key] for key in sorted_keys}
        tqdm.write(f"Selected {len(selected_papers)} papers out of {len(papers)}")
        pbar.update(1)

        pbar.set_postfix(step="writing output")
        if len(papers) > 0:
            if config["OUTPUT"].getboolean("dump_json"):
                with open(config["OUTPUT"]["output_path"] + output_stem + ".json", "w") as outfile:
                    json.dump(selected_papers, outfile, indent=4)
            if config["OUTPUT"].getboolean("dump_md"):
                with open(config["OUTPUT"]["output_path"] + output_stem + ".md", "w") as f:
                    f.write(render_md_string(selected_papers))
            if config["OUTPUT"].getboolean("push_to_slack"):
                SLACK_KEY = os.environ.get("SLACK_KEY")
                if SLACK_KEY is None:
                    tqdm.write("Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack")
                else:
                    push_to_slack(selected_papers)
            if config["OUTPUT"].getboolean("push_to_google"):
                WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
                if WEBHOOK_URL is None:
                    tqdm.write("Warning: push_to_google is true, but WEBHOOK_URL is not set - not pushing to google chat")
                else:
                    push_to_google_chat(selected_papers)
        pbar.update(1)
