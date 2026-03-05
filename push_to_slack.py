"""Render selected papers as Slack blocks and post them to a channel."""

import json
import os
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from parse_json_to_md import extract_criterion_from_paper, topic_shift


def batched(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def send_main_message(block_list: list, channel_id: str, client: WebClient) -> str | None:
    try:
        result = client.chat_postMessage(
            channel=channel_id,
            blocks=block_list,
            text="Arxiv update",
            unfurl_links=False,
        )
        return result["ts"]
    except SlackApiError as exc:
        print(f"Error posting main Slack message: {exc}")
        return None


def send_thread(batches: list[list], channel_id: str, thread_id: str, client: WebClient) -> None:
    for batch in batches:
        if len(batch) < 3:
            continue
        for smaller_batch in batched(batch[1:], 49):
            try:
                client.chat_postMessage(
                    thread_ts=thread_id,
                    text=batch[0]["text"]["text"],
                    channel=channel_id,
                    blocks=batch[:1] + smaller_batch,
                    unfurl_links=False,
                )
            except SlackApiError as exc:
                print(f"Error posting Slack thread batch: {exc}")


def render_paper(paper_entry: dict, counter: int) -> str:
    arxiv_id = paper_entry["arxiv_id"]
    title = paper_entry["title"].replace("&", "&amp;")
    authors = ", ".join(paper_entry["authors"])
    abstract = paper_entry["abstract"]
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

    parts = [
        f"<{arxiv_url}|*{counter}. {title}*>",
        f"*Authors*: {authors}",
        "",
        f"*Abstract*: {abstract}",
        "",
    ]

    if "RELEVANCE" in paper_entry:
        parts.append(f"*Relevance*: {paper_entry['RELEVANCE']}\t")
    if "COMMENT" in paper_entry:
        parts.append(f"*Comment*: {paper_entry['COMMENT']}")

    return "\n".join(parts)


def render_title(paper_entry: dict, counter: int) -> str:
    arxiv_id = paper_entry["arxiv_id"]
    title = paper_entry["title"].replace("&", "&amp;")
    authors = ", ".join(paper_entry["authors"])
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    return f"<{arxiv_url}|*{counter}. {title}*>\n*Authors*: {authors}\n"


def render_topic_block(title: str) -> dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*Topic: {title.strip()}*"},
    }


def _load_topics() -> list[str]:
    with open("configs/paper_topics.txt", "r") as handle:
        lines = [line.strip() for line in handle.readlines()]
    return [line for line in lines if line and line[0].isdigit()]


def build_block_list(
    title_strings: list[str],
    paper_strings: list[str],
    topic_ids: list[int],
) -> tuple[list[dict], list[list[dict]]]:
    slack_block_list: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Paper alert bot update on {datetime.today().strftime('%m/%d/%Y')}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Web: https://variante.github.io/gpt_paper_assistant/",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Total relevant papers (grouped by topic in thread): "
                    f"{len(title_strings)}\nFirst title in each topic shown below"
                ),
            },
        },
        {"type": "divider"},
    ]

    covered_topics: set[int] = set()
    for topic_id, title in zip(topic_ids, title_strings):
        if topic_id in covered_topics:
            continue
        slack_block_list.append({"type": "section", "text": {"type": "mrkdwn", "text": title}})
        covered_topics.add(topic_id)

    criteria = _load_topics()
    thread_blocks = [[render_topic_block("GPT thinks you may like:")]]
    thread_blocks.extend([[render_topic_block(topic)] for topic in criteria])

    for topic_id, paper in zip(topic_ids, paper_strings):
        normalized_topic = topic_id if 0 <= topic_id < len(thread_blocks) else 0
        thread_blocks[normalized_topic].append(
            {"type": "section", "text": {"type": "mrkdwn", "text": paper}}
        )
        thread_blocks[normalized_topic].append({"type": "divider"})

    return slack_block_list, thread_blocks


def push_to_slack(papers_dict: dict[str, dict]) -> None:
    if not papers_dict:
        return

    client = WebClient(token=os.environ["SLACK_KEY"])
    channel_id = os.environ["SLACK_CHANNEL_ID"]

    topic_ids = [extract_criterion_from_paper(paper) for paper in papers_dict.values()]
    title_strings = [
        render_title(paper, idx + topic_shift * topic_ids[idx])
        for idx, paper in enumerate(papers_dict.values())
    ]
    paper_strings = [
        render_paper(paper, idx + topic_shift * topic_ids[idx])
        for idx, paper in enumerate(papers_dict.values())
    ]

    blocks, thread_blocks = build_block_list(title_strings, paper_strings, topic_ids)
    ts = send_main_message(blocks, channel_id, client)
    if ts is not None:
        send_thread(thread_blocks, channel_id, ts, client)


def main() -> None:
    with open("out/output.json", "r") as handle:
        output = json.load(handle)
    push_to_slack(output)


if __name__ == "__main__":
    main()
