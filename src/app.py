import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from news_scraper import fetch_news
from ner_analyzer import extract_entities, get_top_entities
from sentiment_analyzer import get_sentiment, label_sentiment
from geo_visualizer import create_geo_map
from utils import process_news
import logging
from cachetools import TTLCache
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='newspulse.log'
)
logger = logging.getLogger(__name__)

# Initialize cache (TTL of 5 minutes)
cache = TTLCache(maxsize=100, ttl=300)

# Streamlit configuration
st.set_page_config(page_title="NewsPulse: Indian News Insights", layout="wide", initial_sidebar_state="expanded")

st.title("📰 NewsPulse: Indian News Insights")
st.markdown("""
**NewsPulse** delivers real-time insights into India's top news, stock market updates, and events, with a focus on named entities and sentiment analysis.
""")

# Sidebar for user input
st.sidebar.header("🔍 News Filters")
query = st.sidebar.text_input("Search Query (e.g., 'Indian stock market')", "")
period = st.sidebar.selectbox("Time Period", ["1h", "1d", "2d", "3d", "7d"], index=1)
entity_type = st.sidebar.multiselect("Entity Types", ["PERSON", "ORG", "GPE", "EVENT"], default=["PERSON", "ORG", "GPE", "EVENT"])

# Cache key
cache_key = f"india_news_{query}_{period}"

@st.cache_data(show_spinner=False)
def get_cached_news(query, period):
    try:
        start_time = time.time()
        news_df = fetch_news(query=query, period=period, min_articles=30)
        logger.info(f"Fetched {len(news_df)} Indian news articles in {time.time() - start_time:.2f} seconds")
        return news_df
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        st.error("Failed to fetch news. Please try again later or check your internet connection.")
        return pd.DataFrame()

# Fetch and process news
with st.spinner("Fetching and analyzing top Indian news..."):
    if cache_key not in cache:
        news_df = get_cached_news(query, period)
        if not news_df.empty:
            try:
                news_df, entities = process_news(news_df)
                cache[cache_key] = (news_df, entities)
            except Exception as e:
                logger.error(f"Error processing news: {str(e)}")
                st.error("Failed to process news data. Please check logs for details.")
                news_df, entities = pd.DataFrame(), []
    else:
        news_df, entities = cache[cache_key]

# Display results
if not news_df.empty:
    # Filter entities and news by user selection
    filtered_entities = [e for e in entities if e[1] in entity_type]
    filtered_news_df = news_df  # No sentiment filter to simplify

    # Top Entities Word Cloud (First Section)
    st.subheader("☁️ Top Entities Word Cloud")
    cleaned_entity_counts = {
        re.sub(r'[\n\r]+', ' ', k).strip(): v for k, v in 
        {e[0]: filtered_entities.count(e) for e in filtered_entities}.items()
    }
    if cleaned_entity_counts:
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                font_path=None,
                min_font_size=10,
                max_font_size=150
            ).generate_from_frequencies(cleaned_entity_counts)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            st.warning("Unable to generate word cloud due to text rendering issues.")
    else:
        st.warning("No entities found for the selected filters.")

    # News Sentiment Proportion (Second Section)
    st.subheader("🌍 News Sentiment Proportion")
    sentiment_counts = filtered_news_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    total_articles = sentiment_counts['Count'].sum()
    sentiment_counts['Proportion'] = sentiment_counts['Count'] / total_articles * 100
    sentiment_counts['Proportion'] = sentiment_counts['Proportion'].round(2)
    
    pie_chart = alt.Chart(sentiment_counts).mark_arc().encode(
        theta=alt.Theta("Proportion:Q", stack=True),
        color=alt.Color("Sentiment:N", scale=alt.Scale(scheme="set2")),
        tooltip=["Sentiment", "Proportion", "Count"]
    ).properties(
        width=300,
        height=300,
        title=f"Proportion of News Sentiment (Positive: {sentiment_counts[sentiment_counts['Sentiment'] == 'Positive']['Proportion'].iloc[0] if 'Positive' in sentiment_counts['Sentiment'].values else 0}%)"
    )
    st.altair_chart(pie_chart, use_container_width=True)

    # Unified Named Entity Explorer
    st.subheader("🧠 Named Entity Explorer")
    st.markdown("Explore all named entities (people, organizations, locations, events) in India's top news, stock market updates, and events.")
    
    entity_data = []
    for entity, entity_type in filtered_entities:
        entity_articles = filtered_news_df[filtered_news_df['full_text'].str.contains(re.escape(entity), case=False, na=False)]
        avg_sentiment = entity_articles['sentiment_score'].mean() if not entity_articles.empty else 0.0
        entity_data.append({
            'Entity': entity,
            'Type': entity_type,
            'Frequency': filtered_entities.count((entity, entity_type)),
            'Avg Sentiment': round(avg_sentiment, 3)
        })
    
    entity_df = pd.DataFrame(entity_data).drop_duplicates(subset=['Entity', 'Type'])
    entity_df = entity_df.groupby(['Entity', 'Type']).agg({
        'Frequency': 'sum',
        'Avg Sentiment': 'mean'
    }).reset_index().sort_values('Frequency', ascending=False)
    st.dataframe(entity_df.head(20), use_container_width=True)

    # Entity Frequency Chart
    st.subheader("📈 Top Named Entities")
    entity_freq = pd.DataFrame(get_top_entities(filtered_entities, top_n=20), columns=["Entity", "Frequency"])
    if not entity_freq.empty:
        bar_chart = alt.Chart(entity_freq).mark_bar().encode(
            x=alt.X("Frequency:Q", title="Frequency"),
            y=alt.Y("Entity:N", sort="-x", title="Entity"),
            tooltip=["Entity", "Frequency"],
            color=alt.Color("Frequency:Q", scale=alt.Scale(scheme="blues"))
        ).properties(height=400, title="Top 20 Named Entities")
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.warning("No entities match the selected filters.")

    # Optimized Geographic Visualization
    st.subheader("🗺️ Geographic Insights")
    try:
        map_html = create_geo_map([e for e in filtered_entities if e[1] == "GPE"])
        st.components.v1.html(map_html, height=300)
    except Exception as e:
        logger.error(f"Error in geo-visualization: {str(e)}")
        st.warning("Unable to generate geographic map at this time.")

    # News Articles Table with Clickable Links
    st.subheader("📰 Top Indian News Articles")
    display_df = filtered_news_df[["title", "desc", "sentiment", "link"]].copy()
    display_df['link'] = display_df['link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.warning("No news data available for the selected parameters. Try adjusting the query or time period.")