"""Configuration and guardrails for the news agent."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Agent configuration with built-in guardrails."""
    
    # === RESOURCE LIMITS ===
    MAX_ARTICLES_PER_RUN: int = 50
    MAX_ARTICLE_LENGTH: int = 10000  # characters
    MAX_SUMMARY_LENGTH: int = 500  # characters
    MAX_RUNTIME_SECONDS: int = 300  # 5 minutes
    
    # === FEED SAFETY ===
    ALLOWED_FEEDS: List[str] = [
        "https://scholarlykitchen.sspnet.org/feed/",
        "https://www.the-geyser.com/rss/",
        "https://www.platformer.news/rss/",
        "https://www.404media.co/rss/",
        "https://www.wheresyoured.at/rss/",
        "https://www.publishersweekly.com/pw/feeds/section/industry-news/index.xml",
    ]
    
    # === NETWORK SAFETY ===
    REQUEST_TIMEOUT: int = 10  # seconds
    MAX_RETRIES: int = 2
    
    # === QUALITY CONTROLS ===
    MIN_ARTICLE_LENGTH: int = 200  # Skip very short articles
    SKIP_IF_PROCESSED: bool = True  # Don't reprocess articles
    MAX_ARTICLE_AGE_DAYS: int = 0  # 0 = no limit; dedup against feed.xml prevents reprocessing
    
    # === CLAUDE SETTINGS ===
    CLAUDE_API_KEY: str = os.environ.get("CLAUDE_API_KEY", "")
    CLAUDE_MODEL: str = "claude-haiku-4-5-20251001"
    CLAUDE_TIMEOUT: int = 30
    
    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    SAVE_PROMPTS: bool = True
    PROMPT_VERSION: str = "1.0"
    
    # === USER INTERESTS ===
    # Your hardcoded interests for filtering
    INTERESTS: List[str] = [
        "new tools and technology in scholarly publishing, including what they are and what they do",
        "partnerships and collaborations between publishing organizations",
        "publishing platforms and infrastructure",
        "scholarly publishing industry trends and developments",
    ]
    
    # === PATHS ===
    FEED_PATH: str = "docs/feed.xml"
    LOG_PATH: str = "logs/agent.log"
    PROMPTS_DIR: str = "prompts"


# Validate configuration on import
def validate_config():
    """Validate configuration settings."""
    if len(Config.ALLOWED_FEEDS) == 0:
        raise ValueError("Must have at least one feed")
    if Config.MAX_ARTICLES_PER_RUN <= 0:
        raise ValueError("MAX_ARTICLES_PER_RUN must be positive")
    if len(Config.INTERESTS) == 0:
        raise ValueError("Must define at least one interest")
    if not Config.CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY must be set in .env")
    print("✓ Configuration validated")


if __name__ == "__main__":
    validate_config()
    print(f"Model: {Config.CLAUDE_MODEL}")
    print(f"Max articles: {Config.MAX_ARTICLES_PER_RUN}")
    print(f"Feeds: {len(Config.ALLOWED_FEEDS)}")
