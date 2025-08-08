import spacy
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise

def extract_entities(texts):
    try:
        entities = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            doc = nlp(text)
            entities.extend([(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()])
        logger.info(f"Extracted {len(entities)} entities from {len(texts)} texts")
        return entities
    except Exception as e:
        logger.error(f"Error in extract_entities: {str(e)}")
        return []

def get_top_entities(entities, top_n=20):
    try:
        counter = Counter([e[0] for e in entities])
        top_entities = counter.most_common(top_n)
        logger.info(f"Retrieved top {top_n} entities")
        return top_entities
    except Exception as e:
        logger.error(f"Error in get_top_entities: {str(e)}")
        return []