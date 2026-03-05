import configparser
import dataclasses
import json
import os
import re
from pathlib import Path

import retry
from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import EnhancedJSONEncoder, Paper


TITLE_FILTER_POSTFIX = (
    "Identify papers that are absolutely and completely irrelevant to the criteria, "
    "formatted as a JSON list of arxiv ids like [\"ID1\", \"ID2\"]. "
    "Be extremely cautious; if unsure, do not add a paper. "
    "Even if every paper seems irrelevant, keep at least TWO papers. "
    "Respond with the JSON list only."
)


def calc_price(usage: object | None) -> float:
    if usage is None:
        return 0.0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return (0.25 * prompt_tokens + 2 * completion_tokens) / 1e6


@retry.retry(tries=3, delay=2)
def call_chatgpt(
    full_prompt: str,
    openai_client: OpenAI,
    model: str,
    json_mode: bool = True,
    **kwargs,
):
    create_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        **kwargs,
    }
    if json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}
    return openai_client.chat.completions.create(**create_kwargs)


def _local_llm_kwargs() -> dict:
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
    }


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _batched(items: list[Paper], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _paper_to_string(paper: Paper) -> str:
    return (
        f"ArXiv ID: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {' and '.join(paper.authors)}\n"
        f"Abstract: {paper.abstract[:4000]}"
    )


def _paper_to_title(paper: Paper) -> str:
    return f"ArXiv ID: {paper.arxiv_id} Title: {paper.title}\n"


def run_and_parse_chatgpt(
    full_prompt: str,
    openai_client: OpenAI,
    config: configparser.ConfigParser,
) -> tuple[list[dict], float]:
    use_local = config["SELECTION"].getboolean("use_local_llm")
    kwargs = _local_llm_kwargs() if use_local else {}
    completion = call_chatgpt(full_prompt, openai_client, config["SELECTION"]["model"], **kwargs)

    out_text = completion.choices[0].message.content or ""
    if use_local:
        out_text = _strip_thinking(out_text)

    try:
        json_dicts = json.loads(out_text).get("papers", [])
        if not isinstance(json_dicts, list):
            json_dicts = []
    except Exception as exc:
        if config["OUTPUT"].getboolean("debug_messages"):
            print(f"Failed to parse LM output as JSON: {exc}")
            print(out_text)
        json_dicts = []

    return json_dicts, calc_price(completion.usage)


def filter_papers_by_title(
    papers: list[Paper],
    config: configparser.ConfigParser,
    openai_client: OpenAI,
    base_prompt: str,
    criterion: str,
) -> tuple[list[Paper], float]:
    final_list: list[Paper] = []
    total_cost = 0.0
    model = config["SELECTION"]["model"]
    use_local = config["SELECTION"].getboolean("use_local_llm")
    kwargs = _local_llm_kwargs() if use_local else {}

    for batch in _batched(papers, 20):
        papers_string = "".join(_paper_to_title(paper) for paper in batch)
        full_prompt = f"{base_prompt}\n{criterion}\n{papers_string}{TITLE_FILTER_POSTFIX}"
        completion = call_chatgpt(full_prompt, openai_client, model, json_mode=False, **kwargs)
        total_cost += calc_price(completion.usage)

        out_text = completion.choices[0].message.content or "[]"
        try:
            filtered_ids = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id in filtered_ids:
                    if config["OUTPUT"].getboolean("debug_messages"):
                        print(f"Filtered out paper {paper.arxiv_id}")
                    continue
                final_list.append(paper)
        except Exception as exc:
            if config["OUTPUT"].getboolean("debug_messages"):
                print(f"Failed to parse title-filter output: {exc}")
                print(out_text)

    return final_list, total_cost


def run_on_batch(
    paper_batch: list[Paper],
    base_prompt: str,
    criterion: str,
    postfix_prompt: str,
    openai_client: OpenAI,
    config: configparser.ConfigParser,
) -> tuple[list[dict], float]:
    papers_block = "\n\n".join(_paper_to_string(paper) for paper in paper_batch)
    full_prompt = "\n".join([base_prompt, criterion + "\n", papers_block + "\n", postfix_prompt])
    return run_and_parse_chatgpt(full_prompt, openai_client, config)


def _load_prompt_files() -> tuple[str, str, str]:
    with open("configs/base_prompt.txt", "r") as handle:
        base_prompt = handle.read()
    with open("configs/paper_topics.txt", "r") as handle:
        criterion = handle.read()
    with open("configs/postfix_prompt.txt", "r") as handle:
        postfix_prompt = handle.read()
    return base_prompt, criterion, postfix_prompt


def _pick_selected_papers(
    json_dicts: list[dict],
    all_papers: dict[str, Paper],
    relevance_cutoff: int,
) -> tuple[dict[str, dict], dict[str, int], list[dict]]:
    selected: dict[str, dict] = {}
    sort_dict: dict[str, int] = {}
    scored_in_batch: list[dict] = []

    for entry in json_dicts:
        arxiv_id = entry.get("ARXIVID")
        if arxiv_id not in all_papers:
            continue

        try:
            relevance = int(entry.get("RELEVANCE", 0))
        except Exception:
            relevance = 0

        if relevance < relevance_cutoff:
            continue

        merged = {
            **dataclasses.asdict(all_papers[arxiv_id]),
            **{k: v for k, v in entry.items() if k != "ARXIVID"},
        }
        selected[arxiv_id] = merged
        sort_dict[arxiv_id] = relevance
        scored_in_batch.append(merged)

    return selected, sort_dict, scored_in_batch


def filter_by_gpt(
    papers: list[Paper],
    config: configparser.ConfigParser,
    openai_client: OpenAI,
) -> tuple[dict[str, dict], dict[str, int]]:
    if not config["SELECTION"].getboolean("run_llm"):
        return {}, {}

    base_prompt, criterion, postfix_prompt = _load_prompt_files()
    all_papers = {paper.arxiv_id: paper for paper in papers}
    relevance_cutoff = int(config["FILTERING"]["relevance_cutoff"])

    paper_list, title_filter_cost = filter_papers_by_title(
        papers,
        config,
        openai_client,
        base_prompt,
        criterion,
    )

    if config["OUTPUT"].getboolean("debug_messages"):
        print(f"{len(paper_list)} papers after title filtering with cost ${title_filter_cost:.6f}")

    selected_papers: dict[str, dict] = {}
    sort_dict: dict[str, int] = {}
    scored_batches: list[list[dict]] = []
    total_cost = title_filter_cost
    batch_size = max(1, int(config["SELECTION"]["batch_size"]))

    total_batches = (len(paper_list) + batch_size - 1) // batch_size
    for batch in tqdm(_batched(paper_list, batch_size), total=total_batches):
        json_dicts, cost = run_on_batch(
            batch,
            base_prompt,
            criterion,
            postfix_prompt,
            openai_client,
            config,
        )
        total_cost += cost

        selected_in_batch, sort_in_batch, scored_in_batch = _pick_selected_papers(
            json_dicts,
            all_papers,
            relevance_cutoff,
        )
        selected_papers.update(selected_in_batch)
        sort_dict.update(sort_in_batch)
        scored_batches.append(scored_in_batch)

    if config["OUTPUT"].getboolean("dump_debug_file"):
        output_dir = Path(config["OUTPUT"]["output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "gpt_paper_batches.debug.json", "w") as handle:
            json.dump(scored_batches, handle, cls=EnhancedJSONEncoder, indent=4)

    if config["OUTPUT"].getboolean("debug_messages"):
        print(f"Total cost: ${total_cost:.6f}")

    return selected_papers, sort_dict


def _load_debug_papers(path: str) -> list[Paper]:
    with open(path, "r") as handle:
        raw = json.load(handle)

    if raw and isinstance(raw[0], list):
        raw = [item for batch in raw for item in batch]

    return [
        Paper(
            arxiv_id=paper["arxiv_id"],
            authors=paper["authors"],
            title=paper["title"],
            abstract=paper["abstract"],
        )
        for paper in raw
    ]


def _sort_selected(selected: dict[str, dict], scores: dict[str, int]) -> dict[str, dict]:
    sorted_ids = sorted(scores, key=lambda arxiv_id: scores[arxiv_id], reverse=True)
    return {arxiv_id: selected[arxiv_id] for arxiv_id in sorted_ids if arxiv_id in selected}


def main() -> None:
    config = configparser.ConfigParser()
    config.read("configs/config.ini")

    if config["SELECTION"].getboolean("use_local_llm"):
        client = OpenAI(
            base_url=config["SELECTION"]["local_llm_url"],
            api_key="EMPTY",
        )
        config["SELECTION"]["model"] = config["SELECTION"]["local_llm_model"]
    else:
        api_key = os.environ.get("OAI_KEY")
        if not api_key:
            raise ValueError("OpenAI key is not set - please set OAI_KEY")
        client = OpenAI(api_key=api_key)

    papers = _load_debug_papers("in/debug_papers.json")
    selected_papers, sort_dict = filter_by_gpt(papers, config, client)
    sorted_selected = _sort_selected(selected_papers, sort_dict)

    output_dir = Path(config["OUTPUT"]["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "filter_paper_test.debug.json", "w") as handle:
        json.dump(sorted_selected, handle, cls=EnhancedJSONEncoder, indent=4)


if __name__ == "__main__":
    main()
