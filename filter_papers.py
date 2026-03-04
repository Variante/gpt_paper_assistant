import configparser
import dataclasses
import json
import os
import re
from typing import List

import retry
from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import Paper
from arxiv_scraper import EnhancedJSONEncoder


def calc_price(model, usage):
    return (0.25 * usage.prompt_tokens + 2 * usage.completion_tokens) / 1e6


@retry.retry(tries=3, delay=2)
def call_chatgpt(full_prompt, openai_client, model, **kwargs):
    return openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        **kwargs,
    )


def _local_llm_kwargs() -> dict:
    return dict(
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
    )


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_and_parse_chatgpt(full_prompt, openai_client, config):
    use_local = config["SELECTION"].getboolean("use_local_llm")
    kwargs = _local_llm_kwargs() if use_local else {}
    completion = call_chatgpt(full_prompt, openai_client, config["SELECTION"]["model"], **kwargs)
    out_text = completion.choices[0].message.content
    if use_local:
        out_text = _strip_thinking(out_text)
    try:
        json_dicts = json.loads(out_text).get("papers", [])
    except Exception as ex:
        if config["OUTPUT"].getboolean("debug_messages"):
            print("Failed to parse LM output as JSON: " + str(ex))
            print(out_text)
        json_dicts = []
    return json_dicts, calc_price(config["SELECTION"]["model"], completion.usage)


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    return (
        "ArXiv ID: " + paper_entry.arxiv_id + "\n"
        + "Title: " + paper_entry.title + "\n"
        + "Authors: " + " and ".join(paper_entry.authors) + "\n"
        + "Abstract: " + paper_entry.abstract[:4000]
    )


def paper_to_titles(paper_entry: Paper) -> str:
    return "ArXiv ID: " + paper_entry.arxiv_id + " Title: " + paper_entry.title + "\n"


def batched(items, batch_size):
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def filter_papers_by_title(
    papers, config, openai_client, base_prompt, criterion
) -> List[Paper]:
    filter_postfix = 'Identify any papers that are absolutely and completely irrelavent to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of arxiv ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. Even if every paper seems irrelevant, please keep at least TWO papers'
    batches_of_papers = batched(papers, 20)
    final_list = []
    cost = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)
        cost += calc_price(model, completion.usage)
        out_text = completion.choices[0].message.content
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id not in filtered_set:
                    final_list.append(paper)
                else:
                    print("Filtered out paper " + paper.arxiv_id)
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(completion)
            continue
    return final_list, cost


def run_on_batch(
    paper_batch, base_prompt, criterion, postfix_prompt, openai_client, config
):
    batch_str = [paper_to_string(paper) for paper in paper_batch]
    full_prompt = "\n".join(
        [
            base_prompt,
            criterion + "\n",
            "\n\n".join(batch_str) + "\n",
            postfix_prompt,
        ]
    )
    json_dicts, cost = run_and_parse_chatgpt(full_prompt, openai_client, config)
    if cost is None:
        cost = 0
    return json_dicts, cost


def filter_by_gpt(papers, config, openai_client):
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()

    all_papers = {paper.arxiv_id: paper for paper in papers}
    selected_papers = {}
    sort_dict = {}
    all_cost = 0

    if config["SELECTION"].getboolean("run_llm"):
        paper_list, cost = filter_papers_by_title(
            papers, config, openai_client, base_prompt, criterion
        )
        if config["OUTPUT"].getboolean("debug_messages"):
            print(
                str(len(paper_list))
                + " papers after title filtering with cost of $"
                + str(cost)
            )
        all_cost += cost

        batch_of_papers = batched(paper_list, int(config["SELECTION"]["batch_size"]))
        scored_batches = []
        for batch in tqdm(batch_of_papers):
            scored_in_batch = []
            json_dicts, cost = run_on_batch(
                batch, base_prompt, criterion, postfix_prompt, openai_client, config
            )
            all_cost += cost
            for jdict in json_dicts:
                if (
                    jdict["ARXIVID"] in all_papers
                    and int(jdict.get("RELEVANCE", 0)) >= int(config["FILTERING"]["relevance_cutoff"])
                ):
                    paper_dict = {
                        **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                        **{k: v for k, v in jdict.items() if k != "ARXIVID"},
                    }
                    selected_papers[jdict["ARXIVID"]] = paper_dict
                    sort_dict[jdict["ARXIVID"]] = int(jdict["RELEVANCE"])
                    scored_in_batch.append(paper_dict)
            scored_batches.append(scored_in_batch)
        if config["OUTPUT"].getboolean("dump_debug_file"):
            with open(
                config["OUTPUT"]["output_path"] + "gpt_paper_batches.debug.json", "w"
            ) as outfile:
                json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)
        if config["OUTPUT"].getboolean("debug_messages"):
            print("Total cost: $" + str(all_cost))

    return selected_papers, sort_dict


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    OAI_KEY = os.environ.get("OAI_KEY")
    openai_client = OpenAI(api_key=OAI_KEY)
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    # loads papers from 'in/debug_papers.json' and filters them
    with open("in/debug_papers.json", "r") as f:
        paper_list_in_dict = json.load(f)
    papers = [
        [
            Paper(
                arxiv_id=paper["arxiv_id"],
                authors=paper["authors"],
                title=paper["title"],
                abstract=paper["abstract"],
            )
            for paper in batch
        ]
        for batch in paper_list_in_dict
    ]
    all_papers = {}
    paper_outputs = {}
    sort_dict = {}
    total_cost = 0
    for batch in tqdm(papers):
        json_dicts, cost = run_on_batch(
            batch, base_prompt, criterion, postfix_prompt, openai_client, config
        )
        total_cost += cost
        for paper in batch:
            all_papers[paper.arxiv_id] = paper
        for jdict in json_dicts:
            if (
                jdict["ARXIVID"] in all_papers
                and int(jdict.get("RELEVANCE", 0)) >= int(config["FILTERING"]["relevance_cutoff"])
            ):
                paper_outputs[jdict["ARXIVID"]] = {
                    **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                    **{k: v for k, v in jdict.items() if k != "ARXIVID"},
                }
                sort_dict[jdict["ARXIVID"]] = int(jdict["RELEVANCE"])

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    print("total cost:" + str(total_cost))
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())
    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: paper_outputs[key] for key in sorted_keys}

    with open(
        config["OUTPUT"]["output_path"] + "filter_paper_test.debug.json", "w"
    ) as outfile:
        json.dump(selected_papers, outfile, cls=EnhancedJSONEncoder, indent=4)
