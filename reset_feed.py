#!/usr/bin/env python3
"""Reset feed.xml to an empty RSS skeleton, clearing all deduplication history."""
from pathlib import Path
from config import Config

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Scholarly Publishing News</title>
    <link>https://apodacaa.github.io/scholarly-publishing-news/</link>
    <description>Curated news about scholarly publishing, open access, and research infrastructure.</description>
  </channel>
</rss>"""

path = Path(Config.FEED_PATH)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(EMPTY_FEED, encoding="utf-8")
print(f"Reset {Config.FEED_PATH} — next run will reprocess all articles.")
