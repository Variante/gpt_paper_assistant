import dataclasses
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import unescape
from typing import Any, Optional

import feedparser


_HTML_TAG_RE = re.compile(r"<[^<]+?>")
_TITLE_SUFFIX_RE = re.compile(r"\(arXiv:[0-9]+\.[0-9]+v[0-9]+ \[.*\]\)$")


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


@dataclass
class Paper:
    authors: list[str]
    title: str
    abstract: str
    arxiv_id: str

    def __hash__(self) -> int:
        return hash(self.arxiv_id)


def _clean_text(value: str) -> str:
    no_html = _HTML_TAG_RE.sub("", value or "")
    return unescape(no_html.replace("\n", " ")).strip()


def _debug_enabled(config: Optional[dict[str, dict[str, str]]]) -> bool:
    if not config:
        return False
    try:
        return config["OUTPUT"].getboolean("debug_messages")
    except Exception:
        return False


def get_papers_from_arxiv_rss(area: str, config: Optional[dict[str, dict[str, str]]]) -> list[Paper]:
    # Use If-Modified-Since so feed endpoint can skip unchanged responses.
    modified = (datetime.utcnow() - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    feed = feedparser.parse(f"http://rss.arxiv.org/rss/{area}", modified=modified)

    if getattr(feed, "status", None) == 304:
        if _debug_enabled(config):
            print(f"No new papers since {modified} for {area}")
        return []

    entries = feed.entries
    if not entries:
        if _debug_enabled(config):
            print(f"No entries found for {area}")
        return []

    force_primary = False
    if config:
        try:
            force_primary = config["FILTERING"].getboolean("force_primary")
        except Exception:
            force_primary = False
    papers: list[Paper] = []
    for entry in entries:
        if entry.get("arxiv_announce_type") != "new":
            continue

        primary_area = entry.tags[0]["term"] if getattr(entry, "tags", None) else ""
        if force_primary and primary_area and primary_area != area:
            if _debug_enabled(config):
                print(f"Ignoring cross-listed paper: {entry.title}")
            continue

        authors = [_clean_text(author) for author in entry.get("author", "").split("\n") if author.strip()]
        title = _TITLE_SUFFIX_RE.sub("", entry.get("title", "")).strip()
        paper = Paper(
            authors=authors,
            title=title,
            abstract=_clean_text(entry.get("summary", "")),
            arxiv_id=entry.get("link", "").split("/")[-1],
        )
        papers.append(paper)

    return papers


def get_papers_from_arxiv_rss_api(area: str, config: Optional[dict[str, dict[str, str]]]) -> list[Paper]:
    # Kept for backward compatibility with older callers.
    return get_papers_from_arxiv_rss(area, config)
