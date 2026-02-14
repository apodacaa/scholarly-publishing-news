"""Database operations for the news agent."""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database interface for tracking articles and summaries."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. Defaults to Config.DB_PATH
        """
        self.db_path = db_path or Config.DB_PATH
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.connect()
        self.initialize_schema()
    
    def connect(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Return rows as dictionaries instead of tuples
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def initialize_schema(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Articles table - tracks all fetched articles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    source TEXT,
                    pub_date TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT 0
                )
            """)
            
            # Summaries table - tracks LLM analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL,
                    interested BOOLEAN NOT NULL,
                    reasoning TEXT,
                    summary TEXT,
                    model_used TEXT,
                    prompt_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                )
            """)
            
            # Runs table - tracks agent execution for debugging
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    articles_fetched INTEGER DEFAULT 0,
                    articles_processed INTEGER DEFAULT 0,
                    articles_interesting INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    status TEXT,
                    error_message TEXT
                )
            """)
            
            self.conn.commit()
            logger.info("Database schema initialized")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    def article_exists(self, url: str) -> bool:
        """Check if article URL already exists in database.
        
        Args:
            url: Article URL to check
            
        Returns:
            True if article exists, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
        return cursor.fetchone() is not None
    
    def get_existing_urls(self) -> set:
        """Get all article URLs currently in the database.

        Returns:
            Set of URL strings
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT url FROM articles")
        return {row['url'] for row in cursor.fetchall()}

    def insert_article(self, url: str, title: str, source: str, pub_date: str) -> int:
        """Insert new article into database.
        
        Args:
            url: Article URL
            title: Article title
            source: Feed source name
            pub_date: Publication date
            
        Returns:
            Article ID
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO articles (url, title, source, pub_date)
                VALUES (?, ?, ?, ?)
            """, (url, title, source, pub_date))
            
            self.conn.commit()
            article_id = cursor.lastrowid
            logger.debug(f"Inserted article: {title} (ID: {article_id})")
            return article_id
            
        except sqlite3.IntegrityError:
            # URL already exists (UNIQUE constraint)
            logger.debug(f"Article already exists: {url}")
            cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
            return cursor.fetchone()["id"]
    
    def mark_article_processed(self, article_id: int):
        """Mark article as processed.
        
        Args:
            article_id: Article ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE articles 
            SET processed = 1 
            WHERE id = ?
        """, (article_id,))
        self.conn.commit()
    
    def insert_summary(self, article_id: int, interested: bool, reasoning: str,
                      summary: Optional[str] = None, model_used: str = None,
                      prompt_version: str = None) -> int:
        """Insert article analysis summary.
        
        Args:
            article_id: Article ID
            interested: Whether article matched interests
            reasoning: LLM's reasoning for decision
            summary: Article summary (if interesting)
            model_used: LLM model used
            prompt_version: Prompt version used
            
        Returns:
            Summary ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO summaries 
            (article_id, interested, reasoning, summary, model_used, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (article_id, interested, reasoning, summary, 
              model_used or Config.OLLAMA_MODEL,
              prompt_version or Config.PROMPT_VERSION))
        
        self.conn.commit()
        summary_id = cursor.lastrowid
        logger.debug(f"Inserted summary (ID: {summary_id}) for article {article_id}")
        return summary_id
    
    def start_run(self) -> int:
        """Start tracking a new agent run.
        
        Returns:
            Run ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO runs (started_at, status)
            VALUES (?, 'running')
        """, (datetime.now().isoformat(),))
        
        self.conn.commit()
        run_id = cursor.lastrowid
        logger.info(f"Started run {run_id}")
        return run_id
    
    def complete_run(self, run_id: int, articles_fetched: int = 0,
                    articles_processed: int = 0, articles_interesting: int = 0,
                    errors: int = 0, error_message: str = None):
        """Complete an agent run with statistics.
        
        Args:
            run_id: Run ID
            articles_fetched: Number of articles fetched
            articles_processed: Number of articles processed
            articles_interesting: Number of interesting articles
            errors: Number of errors encountered
            error_message: Error message if failed
        """
        cursor = self.conn.cursor()
        
        status = 'completed' if error_message is None else 'failed'
        
        cursor.execute("""
            UPDATE runs 
            SET completed_at = ?,
                articles_fetched = ?,
                articles_processed = ?,
                articles_interesting = ?,
                errors = ?,
                status = ?,
                error_message = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), articles_fetched, articles_processed,
              articles_interesting, errors, status, error_message, run_id))
        
        self.conn.commit()
        logger.info(f"Completed run {run_id}: {status}")
    
    def get_unprocessed_articles(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get articles that haven't been processed yet.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of article dictionaries
        """
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute(
                "SELECT * FROM articles WHERE processed = 0 ORDER BY fetched_at DESC LIMIT ?",
                (limit,)
            )
        else:
            cursor.execute(
                "SELECT * FROM articles WHERE processed = 0 ORDER BY fetched_at DESC"
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_interesting_summaries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interesting article summaries.
        
        Args:
            limit: Maximum number of summaries to return
            
        Returns:
            List of summary dictionaries with article info
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                a.title,
                a.url,
                a.source,
                a.pub_date,
                s.summary,
                s.reasoning,
                s.created_at
            FROM summaries s
            JOIN articles a ON s.article_id = a.id
            WHERE s.interested = 1 AND s.summary IS NOT NULL
            ORDER BY s.created_at DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics.
        
        Returns:
            Dictionary with counts of articles, summaries, etc.
        """
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM articles")
        stats['total_articles'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM articles WHERE processed = 1")
        stats['processed_articles'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM summaries WHERE interested = 1")
        stats['interesting_articles'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM runs WHERE status = 'completed'")
        stats['completed_runs'] = cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for one-off operations
def get_db() -> Database:
    """Get a database instance.
    
    Returns:
        Database instance
    """
    return Database()


if __name__ == "__main__":
    # Test database setup
    import sys
    
    # Setup basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test database operations
    print("Testing database setup...")
    
    with get_db() as db:
        # Test article insertion
        article_id = db.insert_article(
            url="https://example.com/test",
            title="Test Article",
            source="Test Feed",
            pub_date="2025-02-14"
        )
        print(f"✓ Inserted test article (ID: {article_id})")
        
        # Test summary insertion
        summary_id = db.insert_summary(
            article_id=article_id,
            interested=True,
            reasoning="This is interesting because...",
            summary="Test summary of the article"
        )
        print(f"✓ Inserted test summary (ID: {summary_id})")
        
        # Test stats
        stats = db.get_stats()
        print(f"✓ Database stats: {stats}")
        
    print("\n✓ Database setup complete!")
