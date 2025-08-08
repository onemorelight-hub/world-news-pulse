from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize VADER analyzer: {str(e)}")
    raise

def get_sentiment(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        score = analyzer.polarity_scores(text)
        logger.debug(f"Sentiment score for text: {score['compound']}")
        return score["compound"]
    except Exception as e:
        logger.error(f"Error in get_sentiment: {str(e)}")
        return 0.0

def label_sentiment(score):
    try:
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        return "Neutral"
    except Exception as e:
        logger.error(f"Error in label_sentiment: {str(e)}")
        return "Neutral"