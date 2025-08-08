from bertopic import BERTopic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_topics(texts):
    try:
        if not texts or not any(isinstance(t, str) and t.strip() for t in texts):
            logger.warning("No valid texts for topic modeling")
            return []
        model = BERTopic(language="english", calculate_probabilities=True, verbose=False)
        topics, _ = model.fit_transform(texts)
        topic_info = model.get_topic_info()
        topic_data = [(row['Name'], row['Count']) for _, row in topic_info.iterrows() if row['Topic'] != -1]
        logger.info(f"Identified {len(topic_data)} topics")
        return topic_data[:10]
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        return []