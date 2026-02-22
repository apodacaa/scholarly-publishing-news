"""RSS 2.0 feed generator for curated news articles."""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import List

from feeds import Article


def generate_rss_feed(
    new_articles: List[Article],
    existing_items: List[dict],
    output_path: str = "docs/feed.xml",
    max_items: int = 50,
) -> int:
    """Generate an RSS 2.0 feed from new articles prepended to existing items.

    Args:
        new_articles: Article objects rated interesting this run
        existing_items: Dicts parsed from current feed.xml (title, url, description, pub_date, source)
        output_path: Path to write the RSS XML file
        max_items: Maximum number of items to include

    Returns:
        Number of items written to the feed
    """
    # Build RSS structure
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = "Scholarly Publishing News - Curated Feed"
    ET.SubElement(channel, "description").text = (
        "AI-curated news about tools, technology, and partnerships in scholarly publishing"
    )
    ET.SubElement(channel, "link").text = "https://scholarlykitchen.sspnet.org"
    ET.SubElement(channel, "language").text = "en-us"
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(
        datetime.now(timezone.utc)
    )

    count = 0

    # Prepend new articles first
    for article in new_articles:
        if count >= max_items:
            break
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = article.title
        ET.SubElement(item, "link").text = article.url
        ET.SubElement(item, "description").text = article.description or ""

        if article.pub_date:
            try:
                dt = datetime.fromisoformat(article.pub_date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ET.SubElement(item, "pubDate").text = format_datetime(dt)
            except (ValueError, TypeError):
                pass

        if article.source:
            source_el = ET.SubElement(
                item, "source", url=f"https://{article.source}"
            )
            source_el.text = article.source

        guid = ET.SubElement(item, "guid", isPermaLink="true")
        guid.text = article.url
        count += 1

    # Append existing items
    for existing in existing_items:
        if count >= max_items:
            break
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = existing.get("title", "")
        ET.SubElement(item, "link").text = existing.get("url", "")
        ET.SubElement(item, "description").text = existing.get("description", "")

        if existing.get("pub_date"):
            ET.SubElement(item, "pubDate").text = existing["pub_date"]

        if existing.get("source"):
            source_el = ET.SubElement(
                item, "source", url=f"https://{existing['source']}"
            )
            source_el.text = existing["source"]

        guid = ET.SubElement(item, "guid", isPermaLink="true")
        guid.text = existing.get("url", "")
        count += 1

    # Pretty-print and serialize
    ET.indent(rss)
    tree_str = ET.tostring(rss, encoding="unicode")

    # Wrap <description> content in CDATA sections
    tree_str = re.sub(
        r"<description>(.*?)</description>",
        lambda m: f"<description><![CDATA[{m.group(1)}]]></description>",
        tree_str,
        flags=re.DOTALL,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(tree_str)

    return count
