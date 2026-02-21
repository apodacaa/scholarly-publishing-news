"""Main news agent orchestrator - brings all components together."""

import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from config import Config, validate_config
from database import Database, get_db
from feeds import FeedFetcher, Article
from content import ContentExtractor
from llm import OllamaAgent
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


class NewsAgent:
    """Main news agent that orchestrates all components."""
    
    def __init__(self):
        """Initialize the news agent."""
        logger.info("Initializing News Agent...")
        
        self.db = get_db()
        self.feed_fetcher = FeedFetcher()
        self.content_extractor = ContentExtractor()
        self.llm_agent = OllamaAgent()
        
        # Statistics for this run
        self.stats = {
            'articles_fetched': 0,
            'articles_processed': 0,
            'articles_interesting': 0,
            'errors': 0
        }
        
        self.run_id = None
        
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
        
        # Get existing URLs from database
        existing_urls = self.db.get_existing_urls()
        
        # Filter out duplicates
        new_articles = self.feed_fetcher.deduplicate_articles(
            all_articles, 
            existing_urls
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
        
        logger.info(f"Found {len(new_articles)} new articles to process")
        return new_articles
    
    def process_article(self, article: Article) -> bool:
        """Process a single article through the agent pipeline.
        
        Args:
            article: Article to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing: {article.title[:60]}...")
        
        try:
            # 1. Insert article into database
            article_id = self.db.insert_article(
                url=article.url,
                title=article.title,
                source=article.source,
                pub_date=article.pub_date,
                description=article.description
            )
            
            # 2. Extract full content
            logger.debug(f"Extracting content from {article.url}")
            content = self.content_extractor.extract_content(article.url)
            
            if not content:
                logger.warning(f"Failed to extract content, using description")
                content = article.description
            
            if not content or len(content) < Config.MIN_ARTICLE_LENGTH:
                logger.warning(f"Content too short, skipping")
                self.db.mark_article_processed(article_id)
                return False
            
            # 3. Check if interesting
            logger.debug("Checking if article matches interests...")
            interested, reasoning = self.llm_agent.check_interest(
                article.title,
                content,
                article.url
            )
            
            logger.info(f"Interested: {interested} - {reasoning[:100]}...")
            
            # 4. Track interesting articles
            if interested:
                self.stats['articles_interesting'] += 1

            # 5. Save results
            self.db.insert_summary(
                article_id=article_id,
                interested=interested,
                reasoning=reasoning,
                summary=None
            )
            
            # 6. Mark as processed
            self.db.mark_article_processed(article_id)
            
            self.stats['articles_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing article: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    def run(self):
        """Run the news agent pipeline."""
        logger.info("="*60)
        logger.info("Starting News Agent Run")
        logger.info("="*60)
        
        try:
            # Setup timeout protection
            self._setup_timeout()
            
            # Start tracking this run
            self.run_id = self.db.start_run()
            
            # Fetch new articles
            articles = self.fetch_articles()
            
            if not articles:
                logger.info("No new articles to process")
                self.db.complete_run(
                    self.run_id,
                    articles_fetched=self.stats['articles_fetched'],
                    articles_processed=0,
                    articles_interesting=0,
                    errors=0
                )
                return
            
            # Process each article
            logger.info(f"Processing {len(articles)} articles...")
            
            for i, article in enumerate(articles, 1):
                logger.info(f"\n[{i}/{len(articles)}] " + "="*50)
                self.process_article(article)
            
            # Complete run tracking
            self.db.complete_run(
                self.run_id,
                articles_fetched=self.stats['articles_fetched'],
                articles_processed=self.stats['articles_processed'],
                articles_interesting=self.stats['articles_interesting'],
                errors=self.stats['errors']
            )
            
            # Cancel timeout
            self._cancel_timeout()
            
            # Print summary
            self._print_summary()

        except TimeoutError as e:
            logger.error(str(e))
            self.db.complete_run(
                self.run_id,
                articles_fetched=self.stats['articles_fetched'],
                articles_processed=self.stats['articles_processed'],
                articles_interesting=self.stats['articles_interesting'],
                errors=self.stats['errors'],
                error_message="Timeout"
            )
            raise
            
        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            if self.run_id:
                self.db.complete_run(
                    self.run_id,
                    articles_fetched=self.stats['articles_fetched'],
                    articles_processed=self.stats['articles_processed'],
                    articles_interesting=self.stats['articles_interesting'],
                    errors=self.stats['errors'],
                    error_message=str(e)
                )
            raise
        
        finally:
            # Generate RSS feed from all interesting articles in DB
            try:
                count = generate_rss_feed(self.db, output_path="docs/feed.xml")
                logger.info(f"Generated RSS feed with {count} items")
            except Exception as e:
                logger.error(f"RSS feed generation failed: {e}")

            # Cleanup
            self.db.close()
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
        
        # Show interesting articles
        if self.stats['articles_interesting'] > 0:
            logger.info("\n" + "="*60)
            logger.info("INTERESTING ARTICLES")
            logger.info("="*60)
            
            summaries = self.db.get_interesting_summaries(limit=10)
            for i, s in enumerate(summaries, 1):
                logger.info(f"\n{i}. {s['title']}")
                logger.info(f"   Source: {s['source']}")
                logger.info(f"   URL: {s['url']}")
                logger.info(f"   Reasoning: {s['reasoning']}")
        
        # Database stats
        db_stats = self.db.get_stats()
        logger.info("\n" + "="*60)
        logger.info("DATABASE STATS")
        logger.info("="*60)
        logger.info(f"Total articles:       {db_stats['total_articles']}")
        logger.info(f"Processed articles:   {db_stats['processed_articles']}")
        logger.info(f"Interesting articles: {db_stats['interesting_articles']}")
        logger.info(f"Completed runs:       {db_stats['completed_runs']}")
        logger.info("="*60)


def main():
    """Main entry point for the news agent."""
    # Setup
    setup_logging()
    
    logger.info("News Agent Starting...")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {Config.OLLAMA_MODEL}")
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
        agent = NewsAgent()
        agent.run()
        
    except KeyboardInterrupt:
        logger.info("\nAgent interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
