"""Article content extraction for the news agent."""

import requests
import logging
import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re

from config import Config

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extracts clean article text from web pages."""

    MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self):
        self.timeout = Config.REQUEST_TIMEOUT
        self.max_length = Config.MAX_ARTICLE_LENGTH
        self.min_length = Config.MIN_ARTICLE_LENGTH

    def _validate_url(self, url: str) -> bool:
        """Validate URL to prevent SSRF attacks.

        Rejects private IPs, localhost, and non-HTTP(S) schemes.

        Args:
            url: URL to validate

        Returns:
            True if URL is safe, False otherwise
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return False

        if parsed.scheme not in ('http', 'https'):
            logger.warning(f"Rejected non-HTTP(S) URL scheme: {parsed.scheme}")
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Reject localhost
        if hostname in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
            logger.warning(f"Rejected localhost URL: {url}")
            return False

        # Resolve hostname and check for private/reserved IPs
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in addr_info:
                ip = ipaddress.ip_address(sockaddr[0])
                if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                    logger.warning(f"Rejected private/reserved IP for URL: {url}")
                    return False
        except (socket.gaierror, ValueError):
            logger.warning(f"Could not resolve hostname for URL: {url}")
            return False

        return True

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL.

        Args:
            url: Article URL

        Returns:
            HTML content as string, or None if fetch fails
        """
        if not self._validate_url(url):
            return None

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; NewsAgent/1.0)'
                },
                stream=True
            )
            response.raise_for_status()

            # Check Content-Length if provided
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > self.MAX_RESPONSE_BYTES:
                logger.warning(f"Response too large ({content_length} bytes): {url}")
                response.close()
                return None

            # Read with size limit
            chunks = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                bytes_read += len(chunk.encode('utf-8')) if isinstance(chunk, str) else len(chunk)
                if bytes_read > self.MAX_RESPONSE_BYTES:
                    logger.warning(f"Response exceeded size limit ({self.MAX_RESPONSE_BYTES} bytes): {url}")
                    response.close()
                    return None
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode('utf-8', errors='replace'))

            return ''.join(chunks)

        except requests.Timeout:
            logger.warning(f"Timeout fetching URL: {url}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Error fetching URL {url}: {e}")
            return None
    
    def _extract_text(self, html: str) -> str:
        """Extract clean text from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            script.decompose()
        
        # Try to find main content area
        # Common article containers
        article_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.article-content',
            '.entry-content',
            'main'
        ]
        
        text = ""
        for selector in article_selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator=' ', strip=True)
                break
        
        # Fallback to body if no article content found
        if not text:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent prompt injection.
        
        Args:
            text: Raw text
            
        Returns:
            Sanitized text
        """
        # Remove potential prompt injection attempts
        # (This is basic - real prompt injection defense is complex)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove control characters except newline
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        if len(text) > self.max_length:
            logger.debug(f"Truncating text from {len(text)} to {self.max_length} chars")
            # Truncate at word boundary near limit
            truncated = text[:self.max_length]
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            return truncated + "..."
        
        return text
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract full article content from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Cleaned article text, or None if extraction fails
        """
        logger.debug(f"Extracting content from: {url}")
        
        # Fetch HTML
        html = self._fetch_html(url)
        if not html:
            return None
        
        # Extract text
        text = self._extract_text(html)
        
        # Check minimum length
        if len(text) < self.min_length:
            logger.debug(f"Article too short ({len(text)} chars): {url}")
            return None
        
        # Sanitize
        text = self._sanitize_text(text)
        
        # Truncate to limit
        text = self._truncate_text(text)
        
        logger.debug(f"Extracted {len(text)} characters from {url}")
        return text
    
    def extract_batch(self, urls: list) -> dict:
        """Extract content from multiple URLs.
        
        Args:
            urls: List of article URLs
            
        Returns:
            Dictionary mapping URL to extracted content (or None if failed)
        """
        results = {}
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Extracting content {i}/{len(urls)}: {url[:60]}...")
            content = self.extract_content(url)
            results[url] = content
            
            if content is None:
                logger.warning(f"Failed to extract content from: {url}")
        
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Successfully extracted {success_count}/{len(urls)} articles")
        
        return results


# Convenience function
def extract_article_content(url: str) -> Optional[str]:
    """Extract article content from a URL.
    
    Args:
        url: Article URL
        
    Returns:
        Cleaned article text, or None if extraction fails
    """
    extractor = ContentExtractor()
    return extractor.extract_content(url)


if __name__ == "__main__":
    # Test content extraction
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing content extractor...")
    print()
    
    # Test URLs
    test_urls = [
        "https://techcrunch.com/",  # Should work
        "https://news.ycombinator.com/",  # Should work
        "https://invalid-url-that-does-not-exist-12345.com/",  # Should fail
    ]
    
    extractor = ContentExtractor()
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n{i}. Testing: {url}")
        print("-" * 60)
        
        content = extractor.extract_content(url)
        
        if content:
            print(f"✓ Success! Extracted {len(content)} characters")
            print(f"\nFirst 200 chars:")
            print(f"{content[:200]}...")
        else:
            print("✗ Failed to extract content")
    
    print("\n" + "="*60)
    print("Testing batch extraction...")
    
    results = extractor.extract_batch(test_urls[:2])
    print(f"\n✓ Batch extraction complete")
    print(f"  Success: {sum(1 for v in results.values() if v)}/{len(results)}")
    
    print("\n✓ Content extractor test complete!")
