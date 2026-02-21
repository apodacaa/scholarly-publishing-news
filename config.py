"""Configuration and guardrails for the news agent."""

from typing import List


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
    MAX_ARTICLE_AGE_DAYS: int = 14  # Only process articles from last N days (0 = no limit)
    
    # === OLLAMA SETTINGS ===
    OLLAMA_MODEL: str = "llama3.2"  # or "qwen2.5", "mistral"
    OLLAMA_TIMEOUT: int = 30  # seconds per LLM call
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
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
    DB_PATH: str = "data/articles.db"
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
    print("✓ Configuration validated")


if __name__ == "__main__":
    validate_config()
    print(f"Model: {Config.OLLAMA_MODEL}")
    print(f"Max articles: {Config.MAX_ARTICLES_PER_RUN}")
    print(f"Feeds: {len(Config.ALLOWED_FEEDS)}")
