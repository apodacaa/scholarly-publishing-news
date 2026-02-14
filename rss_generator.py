"""RSS 2.0 feed generator for curated news articles."""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime


def generate_rss_feed(db, output_path="feed.xml", max_items=50) -> int:
    """Generate an RSS 2.0 feed from interesting article summaries.

    Args:
        db: Database instance with get_interesting_summaries method
        output_path: Path to write the RSS XML file
        max_items: Maximum number of items to include

    Returns:
        Number of items written to the feed
    """
    summaries = db.get_interesting_summaries(limit=max_items)

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

    for s in summaries:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = s["title"]
        ET.SubElement(item, "link").text = s["url"]
        ET.SubElement(item, "description").text = s["summary"]

        # Parse pub_date and convert to RFC-822
        if s.get("pub_date"):
            try:
                dt = datetime.fromisoformat(s["pub_date"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ET.SubElement(item, "pubDate").text = format_datetime(dt)
            except (ValueError, TypeError):
                pass

        if s.get("source"):
            source_el = ET.SubElement(
                item, "source", url=f"https://{s['source']}"
            )
            source_el.text = s["source"]

        guid = ET.SubElement(item, "guid", isPermaLink="true")
        guid.text = s["url"]

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

    return len(summaries)


if __name__ == "__main__":
    from database import get_db

    db = get_db()
    try:
        count = generate_rss_feed(db)
        print(f"Generated RSS feed with {count} items")
    finally:
        db.close()
