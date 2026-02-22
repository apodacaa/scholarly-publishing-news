"""Main news agent orchestrator - brings all components together."""

import argparse
import logging
import signal
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from config import Config, validate_config
from feeds import FeedFetcher, Article
from content import ContentExtractor
from llm import ClaudeAgent
from rss_generator import generate_rss_feed

# Setup logging
def setup_logging():
    """Configure logging to file and console."""
    # Create logs directory
    Path(Config.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from httpx (Ollama library uses it)
    logging.getLogger('httpx').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def _load_existing_feed(feed_path: str) -> tuple[set[str], list[dict]]:
    """Return (existing_urls, existing_items) from feed.xml, or (set(), []) if missing."""
    existing_urls: set[str] = set()
    existing_items: list[dict] = []
    path = Path(feed_path)
    if not path.exists():
        return existing_urls, existing_items
    tree = ET.parse(path)
    channel = tree.getroot().find('channel')
    if channel is None:
        return existing_urls, existing_items
    for item in channel.findall('item'):
        guid_el = item.find('guid')
        link_el = item.find('link')
        url = (guid_el.text if guid_el is not None else None) or \
              (link_el.text if link_el is not None else None)
        if not url:
            continue
        existing_urls.add(url)
        existing_items.append({
            'url': url,
            'title': (item.findtext('title') or '').strip(),
            'description': (item.findtext('description') or '').strip(),
            'pub_date': (item.findtext('pubDate') or '').strip(),
            'source': (item.findtext('source') or '').strip(),
        })
    return existing_urls, existing_items


class NewsAgent:
    """Main news agent that orchestrates all components."""

    def __init__(self, test_mode: bool = False):
        """Initialize the news agent."""
        self.test_mode = test_mode
        if test_mode:
            logger.info("*** TEST MODE: 1 article, LLM skipped ***")
        logger.info("Initializing News Agent...")

        self.existing_urls, self.existing_items = _load_existing_feed(Config.FEED_PATH)
        self.feed_fetcher = FeedFetcher()
        self.content_extractor = ContentExtractor()
        self.llm_agent = ClaudeAgent()
        self.interesting_articles: List[Article] = []

        # Statistics for this run
        self.stats = {
            'articles_fetched': 0,
            'articles_processed': 0,
            'articles_interesting': 0,
            'errors': 0
        }

        logger.info(f"Loaded {len(self.existing_items)} existing items from feed")
        logger.info("News Agent initialized")

    def _setup_timeout(self):
        """Setup timeout handler to prevent infinite runs."""
        def timeout_handler(signum, frame):
            logger.error(f"Agent timeout after {Config.MAX_RUNTIME_SECONDS} seconds")
            raise TimeoutError("Agent execution exceeded maximum runtime")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(Config.MAX_RUNTIME_SECONDS)

    def _cancel_timeout(self):
        """Cancel the timeout alarm."""
        signal.alarm(0)

    def fetch_articles(self) -> List[Article]:
        """Fetch new articles from RSS feeds.

        Returns:
            List of new Article objects
        """
        logger.info("Fetching RSS feeds...")

        # Fetch all articles from feeds
        all_articles = self.feed_fetcher.fetch_all_feeds()
        self.stats['articles_fetched'] = len(all_articles)

        # Filter out duplicates using existing feed URLs
        dedup_urls = set() if self.test_mode else self.existing_urls
        new_articles = self.feed_fetcher.deduplicate_articles(
            all_articles,
            dedup_urls
        )

        # Filter by date if configured
        if Config.MAX_ARTICLE_AGE_DAYS > 0:
            cutoff_date = datetime.now() - timedelta(days=Config.MAX_ARTICLE_AGE_DAYS)

            filtered_articles = []
            for article in new_articles:
                try:
                    article_date = datetime.fromisoformat(article.pub_date)
                    if article_date >= cutoff_date:
                        filtered_articles.append(article)
                    else:
                        logger.debug(f"Skipping old article: {article.title} ({article.pub_date})")
                except (ValueError, TypeError):
                    # If date parsing fails, include the article
                    filtered_articles.append(article)

            old_count = len(new_articles) - len(filtered_articles)
            if old_count > 0:
                logger.info(f"Filtered {old_count} articles older than {Config.MAX_ARTICLE_AGE_DAYS} days")

            new_articles = filtered_articles

        # Apply limit
        if len(new_articles) > Config.MAX_ARTICLES_PER_RUN:
            logger.info(
                f"Limiting to {Config.MAX_ARTICLES_PER_RUN} articles "
                f"(found {len(new_articles)} new)"
            )
            new_articles = new_articles[:Config.MAX_ARTICLES_PER_RUN]

        if self.test_mode and new_articles:
            new_articles = new_articles[:1]
            logger.info("Test mode: limiting to 1 article")

        logger.info(f"Found {len(new_articles)} new articles to process")
        return new_articles

    def process_article(self, article: Article) -> Optional[Article]:
        """Process a single article through the agent pipeline.

        Args:
            article: Article to process

        Returns:
            Article if interesting, None otherwise
        """
        logger.info(f"Processing: {article.title[:60]}...")

        if self.test_mode:
            logger.info("Test mode: skipping LLM, assuming interested=True")
            self.stats['articles_processed'] += 1
            self.stats['articles_interesting'] += 1
            return article

        try:
            # 1. Extract full content
            logger.debug(f"Extracting content from {article.url}")
            content = self.content_extractor.extract_content(article.url)

            if not content:
                logger.warning(f"Failed to extract content, using description")
                content = article.description

            if not content or len(content) < Config.MIN_ARTICLE_LENGTH:
                logger.warning(f"Content too short, skipping")
                self.stats['articles_processed'] += 1
                return None

            # 2. Check if interesting
            logger.debug("Checking if article matches interests...")
            interested, reasoning = self.llm_agent.check_interest(
                article.title,
                content,
                article.url
            )

            logger.info(f"Interested: {interested} - {reasoning[:100]}...")

            self.stats['articles_processed'] += 1

            if interested:
                self.stats['articles_interesting'] += 1
                return article

            return None

        except Exception as e:
            logger.error(f"Error processing article: {e}", exc_info=True)
            self.stats['errors'] += 1
            return None

    def run(self):
        """Run the news agent pipeline."""
        logger.info("="*60)
        logger.info("Starting News Agent Run")
        logger.info("="*60)

        try:
            # Setup timeout protection
            self._setup_timeout()

            # Fetch new articles
            articles = self.fetch_articles()

            if not articles:
                logger.info("No new articles to process")
            else:
                # Process each article
                logger.info(f"Processing {len(articles)} articles...")

                for i, article in enumerate(articles, 1):
                    logger.info(f"\n[{i}/{len(articles)}] " + "="*50)
                    result = self.process_article(article)
                    if result is not None:
                        self.interesting_articles.append(result)

            # Cancel timeout
            self._cancel_timeout()

            # Print summary
            self._print_summary()

        except TimeoutError as e:
            logger.error(str(e))
            raise

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            raise

        finally:
            # Generate RSS feed from new interesting articles + existing items
            try:
                count = generate_rss_feed(
                    self.interesting_articles,
                    self.existing_items,
                    output_path=Config.FEED_PATH,
                )
                logger.info(f"Generated RSS feed with {count} items")
            except Exception as e:
                logger.error(f"RSS feed generation failed: {e}")

            logger.info("News Agent run complete")

    def _print_summary(self):
        """Print run summary statistics."""
        logger.info("\n" + "="*60)
        logger.info("RUN SUMMARY")
        logger.info("="*60)
        logger.info(f"Articles fetched:     {self.stats['articles_fetched']}")
        logger.info(f"Articles processed:   {self.stats['articles_processed']}")
        logger.info(f"Articles interesting: {self.stats['articles_interesting']}")
        logger.info(f"Errors:               {self.stats['errors']}")

        if self.interesting_articles:
            logger.info("\n" + "="*60)
            logger.info("INTERESTING ARTICLES")
            logger.info("="*60)
            for i, article in enumerate(self.interesting_articles, 1):
                logger.info(f"\n{i}. {article.title}")
                logger.info(f"   Source: {article.source}")
                logger.info(f"   URL: {article.url}")

        logger.info("="*60)


def main():
    """Main entry point for the news agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test mode: 1 article, no LLM')
    args = parser.parse_args()

    # Setup
    setup_logging()

    logger.info("News Agent Starting...")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {Config.CLAUDE_MODEL}")
    logger.info(f"  Max articles: {Config.MAX_ARTICLES_PER_RUN}")
    logger.info(f"  Feeds: {len(Config.ALLOWED_FEEDS)}")
    logger.info(f"  Interests: {', '.join(Config.INTERESTS)}")

    # Validate config
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Run agent
    try:
        agent = NewsAgent(test_mode=args.test)
        agent.run()

    except KeyboardInterrupt:
        logger.info("\nAgent interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
