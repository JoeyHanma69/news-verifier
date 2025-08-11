#!/usr/bin/env python3
"""
Create a list of unreliable news URLs for training
This script helps identify and collect URLs from known unreliable sources
"""

import requests
import feedparser
import json
from datetime import datetime
import argparse
import logging
from typing import List, Dict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnreliableSourceCollector:
    def __init__(self):
        # Known unreliable sources with their RSS feeds (if available)
        # Note: These are examples - you should verify and update this list
        self.unreliable_sources = {
            'infowars': {
                'domain': 'infowars.com',
                'rss_feeds': [
                    # RSS feeds if available
                ],
                'reason': 'conspiracy_theories',
                'manual_urls': [
                    # Add specific article URLs manually
                ]
            },
            'naturalnews': {
                'domain': 'naturalnews.com',
                'rss_feeds': [],
                'reason': 'pseudoscience',
                'manual_urls': []
            },
            'satirical_news': {
                'domain': 'theonion.com',  # Satirical, not unreliable but good for training
                'rss_feeds': [
                    'https://www.theonion.com/rss'
                ],
                'reason': 'satirical',
                'manual_urls': []
            }
        }
        
        # Fact-checking sites that identify false claims
        self.fact_check_sources = {
            'snopes': 'https://www.snopes.com/fact-check/',
            'politifact': 'https://www.politifact.com/',
            'factcheck_org': 'https://www.factcheck.org/'
        }
    
    def collect_from_rss(self, source_name: str, max_articles: int = 20) -> List[str]:
        """Collect URLs from RSS feeds of unreliable sources"""
        urls = []
        
        if source_name not in self.unreliable_sources:
            logger.error(f"Unknown source: {source_name}")
            return urls
        
        source_info = self.unreliable_sources[source_name]
        
        for feed_url in source_info['rss_feeds']:
            try:
                logger.info(f"Fetching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                count = 0
                for entry in feed.entries:
                    if count >= max_articles:
                        break
                    
                    urls.append(entry.link)
                    count += 1
                
                logger.info(f"Collected {count} URLs from {feed_url}")
                
            except Exception as e:
                logger.error(f"Error processing RSS feed {feed_url}: {e}")
        
        return urls
    
    def collect_manual_urls(self, source_name: str) -> List[str]:
        """Get manually curated URLs from unreliable sources"""
        if source_name not in self.unreliable_sources:
            return []
        
        return self.unreliable_sources[source_name]['manual_urls']
    
    def collect_from_fact_checkers(self, max_articles: int = 50) -> List[Dict]:
        """Collect URLs of debunked articles from fact-checking sites"""
        # This would require scraping fact-checking sites to find debunked articles
        # For now, return empty list - implement based on specific fact-checker APIs
        logger.info("Fact-checker collection not implemented yet")
        return []
    
    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate that URLs are accessible"""
        valid_urls = []
        
        for url in urls:
            try:
                response = requests.head(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    valid_urls.append(url)
                else:
                    logger.warning(f"URL not accessible: {url} (status: {response.status_code})")
                    
            except Exception as e:
                logger.warning(f"Error validating URL {url}: {e}")
        
        logger.info(f"Validated {len(valid_urls)} out of {len(urls)} URLs")
        return valid_urls
    
    def save_urls_to_file(self, urls: List[str], filename: str):
        """Save URLs to a text file"""
        try:
            with open(filename, 'w') as f:
                f.write(f"# Unreliable news URLs for training\n")
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                f.write(f"# Total URLs: {len(urls)}\n\n")
                
                for url in urls:
                    f.write(f"{url}\n")
            
            logger.info(f"Saved {len(urls)} URLs to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving URLs to file: {e}")
    
    def create_comprehensive_list(self, output_file: str = 'unreliable_urls.txt'):
        """Create a comprehensive list of unreliable news URLs"""
        all_urls = []
        
        # Collect from all known unreliable sources
        for source_name in self.unreliable_sources.keys():
            logger.info(f"Collecting URLs from {source_name}...")
            
            # RSS feeds
            rss_urls = self.collect_from_rss(source_name, max_articles=15)
            all_urls.extend(rss_urls)
            
            # Manual URLs
            manual_urls = self.collect_manual_urls(source_name)
            all_urls.extend(manual_urls)
        
        # Remove duplicates
        all_urls = list(set(all_urls))
        
        # Validate URLs
        valid_urls = self.validate_urls(all_urls)
        
        # Save to file
        self.save_urls_to_file(valid_urls, output_file)
        
        return valid_urls

def main():
    parser = argparse.ArgumentParser(description='Create list of unreliable news URLs for training')
    parser.add_argument('--output', default='unreliable_urls.txt',
                       help='Output file for unreliable URLs')
    parser.add_argument('--source', 
                       help='Collect from specific source only')
    parser.add_argument('--validate', action='store_true',
                       help='Validate URLs before saving')
    parser.add_argument('--max_per_source', type=int, default=15,
                       help='Maximum URLs per source')
    
    args = parser.parse_args()
    
    collector = UnreliableSourceCollector()
    
    if args.source:
        # Collect from specific source
        urls = collector.collect_from_rss(args.source, args.max_per_source)
        urls.extend(collector.collect_manual_urls(args.source))
    else:
        # Collect from all sources
        urls = collector.create_comprehensive_list(args.output)
        return
    
    # Remove duplicates
    urls = list(set(urls))
    
    # Validate if requested
    if args.validate:
        urls = collector.validate_urls(urls)
    
    # Save to file
    collector.save_urls_to_file(urls, args.output)
    
    print(f"Collected {len(urls)} URLs and saved to {args.output}")

if __name__ == "__main__":
    main()
