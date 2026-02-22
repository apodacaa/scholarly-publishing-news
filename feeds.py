"""RSS feed fetching and parsing for the news agent."""

import feedparser
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)


class Article:
    """Represents a parsed article from an RSS feed."""
    
    def __init__(self, url: str, title: str, source: str, 
                 pub_date: str, description: str = ""):
        self.url = url
        self.title = title
        self.source = source
        self.pub_date = pub_date
        self.description = description
    
    def __repr__(self):
        return f"Article(title='{self.title[:50]}...', source='{self.source}')"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert article to dictionary."""
        return {
            'url': self.url,
            'title': self.title,
            'source': self.source,
            'pub_date': self.pub_date,
            'description': self.description
        }


class FeedFetcher:
    """Fetches and parses RSS feeds with safety guardrails."""
    
    def __init__(self):
        self.allowed_feeds = Config.ALLOWED_FEEDS
        self.timeout = Config.REQUEST_TIMEOUT
    
    def _is_feed_allowed(self, feed_url: str) -> bool:
        """Check if feed URL is in whitelist.
        
        Args:
            feed_url: URL to check
            
        Returns:
            True if allowed, False otherwise
        """
        return feed_url in self.allowed_feeds
    
    def _get_source_name(self, feed_url: str) -> str:
        """Extract source name from feed URL.
        
        Args:
            feed_url: Feed URL
            
        Returns:
            Domain name as source identifier
        """
        parsed = urlparse(feed_url)
        domain = parsed.netloc.replace('www.', '')
        return domain
    
    def _parse_date(self, entry) -> str:
        """Parse publication date from feed entry.
        
        Args:
            entry: feedparser entry object
            
        Returns:
            ISO format date string
        """
        # Try multiple date fields
        for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
            if hasattr(entry, date_field):
                parsed_date = getattr(entry, date_field)
                if parsed_date:
                    try:
                        return datetime(*parsed_date[:6]).isoformat()
                    except (TypeError, ValueError):
                        continue
        
        # Fallback to current time if no date found
        return datetime.now().isoformat()
    
    def fetch_feed(self, feed_url: str) -> List[Article]:
        """Fetch and parse a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of Article objects
            
        Raises:
            ValueError: If feed URL is not in whitelist
            requests.RequestException: If feed fetch fails
        """
        # Security: Check whitelist
        if not self._is_feed_allowed(feed_url):
            raise ValueError(f"Feed not in whitelist: {feed_url}")
        
        logger.info(f"Fetching feed: {feed_url}")
        
        try:
            # Fetch feed with timeout
            response = requests.get(
                feed_url,
                timeout=self.timeout,
                headers={'User-Agent': 'NewsAgent/1.0'}
            )
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                # Feed has parsing issues but might still be usable
                logger.warning(f"Feed has parsing issues: {feed_url}")
            
            # Extract articles
            articles = []
            source_name = self._get_source_name(feed_url)
            
            for entry in feed.entries:
                # Extract article data
                url = entry.get('link', '')
                title = entry.get('title', 'Untitled')
                raw_description = entry.get('summary', entry.get('description', ''))
                description = BeautifulSoup(raw_description, 'html.parser').get_text(separator=' ', strip=True)
                pub_date = self._parse_date(entry)
                
                # Skip entries without URLs
                if not url:
                    logger.debug(f"Skipping entry without URL: {title}")
                    continue
                
                article = Article(
                    url=url,
                    title=title,
                    source=source_name,
                    pub_date=pub_date,
                    description=description
                )
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles
            
        except requests.Timeout:
            logger.error(f"Timeout fetching feed: {feed_url}")
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing feed {feed_url}: {e}")
            return []
    
    def fetch_all_feeds(self) -> List[Article]:
        """Fetch all configured RSS feeds.
        
        Returns:
            List of all articles from all feeds
        """
        all_articles = []
        
        for feed_url in self.allowed_feeds:
            try:
                articles = self.fetch_feed(feed_url)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Failed to fetch feed {feed_url}: {e}")
                # Continue with other feeds
                continue
        
        logger.info(f"Fetched {len(all_articles)} total articles from {len(self.allowed_feeds)} feeds")
        return all_articles
    
    def deduplicate_articles(self, articles: List[Article], 
                            existing_urls: set) -> List[Article]:
        """Remove duplicate articles based on URL.
        
        Args:
            articles: List of articles to deduplicate
            existing_urls: Set of URLs already in database
            
        Returns:
            List of new articles not in existing_urls
        """
        new_articles = [
            article for article in articles 
            if article.url not in existing_urls
        ]
        
        duplicates = len(articles) - len(new_articles)
        if duplicates > 0:
            logger.info(f"Filtered {duplicates} duplicate articles")
        
        return new_articles


if __name__ == "__main__":
    # Test feed fetching
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing feed fetcher...")
    print(f"Configured feeds: {len(Config.ALLOWED_FEEDS)}")
    for feed in Config.ALLOWED_FEEDS:
        print(f"  - {feed}")
    print()

    fetcher = FeedFetcher()

    print("Fetching feeds...")
    articles = fetcher.fetch_all_feeds()

    print(f"\n✓ Fetched {len(articles)} total articles")

    if articles:
        print("\nFirst 3 articles:")
        for i, article in enumerate(articles[:3], 1):
            print(f"\n{i}. {article.title}")
            print(f"   Source: {article.source}")
            print(f"   URL: {article.url[:60]}...")
            print(f"   Date: {article.pub_date}")

    print("\n✓ Feed fetcher test complete!")
