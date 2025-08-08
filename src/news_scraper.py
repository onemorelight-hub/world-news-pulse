from GoogleNews import GoogleNews
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz
import random
import re
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of user agents to rotate and avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def clean_url(url):
    """Remove Google-specific query parameters from URL."""
    try:
        parsed_url = urlparse(url)
        cleaned_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
        return cleaned_url if cleaned_url else url
    except Exception as e:
        logger.warning(f"Error cleaning URL {url}: {str(e)}")
        return url

def parse_article(article_data, max_retries=4):
    """Helper function to parse article content using requests and BeautifulSoup."""
    try:
        url = clean_url(article_data.get('link', ''))
        if not url or any(domain in url for domain in ['youtube.com', 'twitter.com', 'x.com', 'facebook.com']):
            logger.warning(f"Skipping invalid URL: {url}")
            return None
        for attempt in range(max_retries):
            try:
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                full_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                if not full_text or len(full_text.strip()) < 50:
                    logger.warning(f"Empty or short content for {url}, using description")
                    return {
                        'title': article_data.get('title', ''),
                        'desc': article_data.get('desc', ''),
                        'link': url,
                        'media': article_data.get('media', ''),
                        'date': pd.to_datetime(article_data.get('date', ''), errors='coerce', utc=True),
                        'full_text': article_data.get('desc', '')
                    }
                return {
                    'title': article_data.get('title', ''),
                    'desc': article_data.get('desc', ''),
                    'link': url,
                    'media': article_data.get('media', ''),
                    'date': pd.to_datetime(article_data.get('date', ''), errors='coerce', utc=True),
                    'full_text': full_text
                }
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 404]:
                    logger.warning(f"HTTP {e.response.status_code} for {url}, skipping")
                    return {
                        'title': article_data.get('title', ''),
                        'desc': article_data.get('desc', ''),
                        'link': url,
                        'media': article_data.get('media', ''),
                        'date': pd.to_datetime(article_data.get('date', ''), errors='coerce', utc=True),
                        'full_text': article_data.get('desc', '')
                    }
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 2))
                continue
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 2))
                continue
        logger.error(f"Failed to fetch content for {url} after {max_retries} attempts")
        return {
            'title': article_data.get('title', ''),
            'desc': article_data.get('desc', ''),
            'link': url,
            'media': article_data.get('media', ''),
            'date': pd.to_datetime(article_data.get('date', ''), errors='coerce', utc=True),
            'full_text': article_data.get('desc', '')
        }
    except Exception as e:
        logger.error(f"Error processing article {url}: {str(e)}")
        return None

def fetch_news(query="", period="1d", min_articles=30):
    try:
        googlenews = GoogleNews(lang='en', region='IN', period=period)
        base_query = "India Top news" if not query else query + "in India"
        # Default search terms when no query is provided
        search_terms = [
            base_query,
            "India Sensex",  # Stock market news
            "RBI MPC meeting",  # Economic policy
            "Indian stock market",  # Stock market
            "India economy"  # Economic news
        ] if not query else [base_query ]  # Use only user query if provided
        all_articles = []
        seen_urls = set()
        seen_titles = set()

        for search_query in search_terms:
            try:
                logger.debug(f"Processing query: {search_query}")
                googlenews.clear()
                googlenews.search(search_query)
                results = googlenews.results(sort=True)
                if not results:
                    logger.warning(f"No results found for query '{search_query}'")
                    continue
                for article in results:
                    url = clean_url(article.get('link', ''))
                    title = article.get('title', '')
                    if url not in seen_urls and title not in seen_titles:
                        all_articles.append(article)
                        seen_urls.add(url)
                        seen_titles.add(title)
                logger.info(f"Fetched {len(results)} articles for query '{search_query}'")
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                logger.error(f"Error processing query '{search_query}': {str(e)}")
                continue

        unique_articles = all_articles[:min_articles]
        logger.info(f"Collected {len(unique_articles)} unique articles")

        if not unique_articles:
            logger.warning("No valid articles after deduplication")
            return pd.DataFrame()

        news_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_article = {executor.submit(parse_article, article): article for article in unique_articles}
            for future in as_completed(future_to_article):
                result = future.result()
                if result:
                    news_data.append(result)

        if not news_data:
            logger.warning("No articles processed successfully")
            return pd.DataFrame()

        df = pd.DataFrame(news_data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['date'] = df['date'].fillna(datetime.now(pytz.UTC))
        logger.info(f"Fetched and processed {len(df)} articles")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_news: {str(e)}")
        return pd.DataFrame()