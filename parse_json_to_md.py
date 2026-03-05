import json
import re
from datetime import datetime


link_prefix = "user-content-"
topic_shift = 1000
_CRITERION_RE = re.compile(r"criteri(?:on|a)?\D*(\d+)", re.IGNORECASE)
_NON_ANCHOR_RE = re.compile(r"[^a-zA-Z0-9 -]")


def _arxiv_links(arxiv_id: str) -> tuple[str, str]:
    return f"https://arxiv.org/abs/{arxiv_id}", f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _anchor_from_title(idx: int, title: str) -> str:
    cleaned = _NON_ANCHOR_RE.sub("", f"{idx} {title}").replace(" ", "-").lower()
    return cleaned


def render_paper(paper_entry: dict, idx: int) -> str:
    arxiv_id = paper_entry["arxiv_id"]
    title = paper_entry["title"]
    abstract = paper_entry["abstract"]
    authors = paper_entry["authors"]
    arxiv_url, pdf_url = _arxiv_links(arxiv_id)

    lines = [
        f"### {idx}\\. [{title}]({arxiv_url})",
        f"**ArXiv:** {arxiv_id} [[page]({arxiv_url})] [[pdf]({pdf_url})]",
        "",
        f"**Authors:** {', '.join(authors)}",
        "",
        f"**Abstract:** {abstract}",
        "",
    ]

    if "COMMENT" in paper_entry:
        lines.extend([f"**Comment:** {paper_entry['COMMENT']}", ""])
    if "RELEVANCE" in paper_entry:
        lines.append(f"**Relevance:** {paper_entry['RELEVANCE']}")

    topic_id = idx // topic_shift
    topic_anchor = f"topic-{topic_id}" if topic_id else "go-beyond"
    lines.append(f"Back to [[topic](#{link_prefix}{topic_anchor})] [[top](#{link_prefix}topics)]")
    return "\n".join(lines)


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    arxiv_id = paper_entry["arxiv_id"]
    title = paper_entry["title"]
    authors = paper_entry["authors"]
    arxiv_url, _ = _arxiv_links(arxiv_id)
    anchor = _anchor_from_title(idx, title)

    return (
        f"{idx}\\. [{title}]({arxiv_url}) [[more](#{link_prefix}{anchor})] \\\\n"
        f"**Authors:** {', '.join(authors)}"
    )


def render_criteria(criteria: list[str]) -> str:
    items = []
    for criterion in criteria:
        topic_idx = int(criterion.split(".", 1)[0])
        items.append(f"[{criterion}](#{link_prefix}topic-{topic_idx})")
    items.append(f"[Go beyond](#{link_prefix}go-beyond)")
    return "\n\n".join(items) + "\n\n"


def extract_criterion_from_paper(paper_entry: dict) -> int:
    comment = paper_entry.get("COMMENT", "")
    match = _CRITERION_RE.search(comment)
    if not match:
        return 0
    return int(match.group(1))


def render_md_paper_title_by_topic(topic: str, papers: list[str]) -> str:
    if not papers:
        return ""
    return (
        f"### {topic}\n"
        + "\n".join(papers)
        + f"\n\nBack to [[top](#{link_prefix}topics)]\n\n---\n"
    )


def _load_criteria() -> list[str]:
    with open("configs/paper_topics.txt", "r") as handle:
        lines = [line.strip() for line in handle.readlines()]
    return [line for line in lines if line and line[0].isdigit()]


def render_md_string(papers_dict: dict[str, dict]) -> str:
    criteria = _load_criteria()
    criteria_string = render_criteria(criteria)

    output = [
        f"# Personalized Daily Arxiv Papers {datetime.today().strftime('%m/%d/%Y')}",
        "",
        "This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant).",
        "The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant).",
        "",
        "## Topics",
        "",
        "Paper selection prompt and criteria (jump to the section by clicking the link):",
        "",
        criteria_string,
        "---",
    ]

    title_groups = [[] for _ in range(len(criteria) + 1)]
    full_groups = [[] for _ in range(len(criteria) + 1)]

    for idx, paper in enumerate(papers_dict.values()):
        topic_idx = extract_criterion_from_paper(paper)
        if topic_idx > len(criteria):
            topic_idx = 0
        item_idx = idx + topic_idx * topic_shift
        title_groups[topic_idx].append(render_title_and_author(paper, item_idx))
        full_groups[topic_idx].append(render_paper(paper, item_idx))

    for topic_idx in range(1, len(title_groups)):
        output.append(render_md_paper_title_by_topic(f"Topic {topic_idx}", title_groups[topic_idx]))
    output.append(render_md_paper_title_by_topic("Go beyond", title_groups[0]))

    full_sections = ["\n".join(group) for group in full_groups[1:] + full_groups[:1] if group]
    full_list = "\n---\n".join(full_sections)
    output.append(f"## Full paper list\n{full_list}")
    return "\n".join(part for part in output if part)


def main() -> None:
    with open("out/output.json", "r") as handle:
        output = json.load(handle)
    with open("out/output.md", "w") as handle:
        handle.write(render_md_string(output))


if __name__ == "__main__":
    main()
