import pandas as pd
from sentiment_analyzer import get_sentiment, label_sentiment
from ner_analyzer import extract_entities
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_news(news_df):
    try:
        if news_df.empty:
            logger.warning("Empty news DataFrame received")
            return news_df, []
        
        news_df['desc'] = news_df['desc'].fillna('')
        news_df['full_text'] = news_df['full_text'].fillna(news_df['desc'])
        
        # Apply sentiment analysis on full_text for better accuracy
        news_df['sentiment_score'] = news_df['full_text'].apply(get_sentiment)
        news_df['sentiment'] = news_df['sentiment_score'].apply(label_sentiment)
        
        # Extract entities from full_text for richer context
        all_entities = extract_entities(news_df['full_text'].tolist())
        logger.info(f"Processed {len(news_df)} articles with {len(all_entities)} entities")
        return news_df, all_entities
    except Exception as e:
        logger.error(f"Error in process_news: {str(e)}")
        return news_df, []