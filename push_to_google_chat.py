"""Send selected papers to Google Chat via webhook."""

import json
import os
from datetime import datetime

import requests
from requests import Response

from parse_json_to_md import extract_criterion_from_paper


def send_text_card(title: str, paragraphs: list[str], webhook_url: str) -> Response:
    card = {
        "cards": [
            {
                "header": {"title": title},
                "sections": [{"widgets": [{"textParagraph": {"text": paragraph}}]} for paragraph in paragraphs],
            }
        ]
    }
    return requests.post(webhook_url, json=card)


def render_paper(paper_entry: dict) -> str:
    arxiv_id = paper_entry["arxiv_id"]
    title = paper_entry["title"].replace("&", "&amp;")
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    authors = ", ".join(paper_entry["authors"])
    return f"<a href='{arxiv_url}'>{title}</a>\n{authors}"


def group_by_topics(topic_ids: list[int], paper_strings: list[str]) -> list[str]:
    grouped: dict[int, list[str]] = {}
    for topic_id, paper in zip(topic_ids, paper_strings):
        grouped.setdefault(topic_id, []).append(paper)

    sections: list[str] = []
    for topic_id in sorted(topic for topic in grouped if topic != 0):
        sections.append(f"<b>Topic {topic_id}: </b>\n" + "\n\n".join(grouped[topic_id]))

    if grouped.get(0):
        sections.append("<b>GPT thinks you might like: </b>\n" + "\n\n".join(grouped[0]))

    return sections


def push_to_google_chat(papers_dict: dict[str, dict]) -> None:
    if not papers_dict:
        return

    webhook_url = os.environ["WEBHOOK_URL"]
    topic_ids = [extract_criterion_from_paper(paper) for paper in papers_dict.values()]
    paper_strings = [render_paper(paper) for paper in papers_dict.values()]

    paragraphs = group_by_topics(topic_ids, paper_strings)
    paragraphs.append("Check <a href='https://variante.github.io/gpt_paper_assistant/'>the web version</a>.")

    title = f"Arxiv update on {datetime.today().strftime('%m/%d/%Y')}"
    response = send_text_card(title, paragraphs, webhook_url)
    if response.status_code >= 400:
        print(f"Google Chat webhook failed with status {response.status_code}: {response.text}")


def main() -> None:
    with open("out/output.json", "r") as handle:
        output = json.load(handle)
    push_to_google_chat(output)


if __name__ == "__main__":
    main()
