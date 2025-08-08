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

# Custom CSS for table styling
st.markdown("""
<style>
.stDataFrame {
    font-size: 14px;
}
.stDataFrame th {
    background-color: #f0f2f6;
    color: #333;
    padding: 8px;
}
.stDataFrame td {
    padding: 8px;
}
.stDataFrame tr:nth-child(even) {
    background-color: #f9f9f9;
}
.stDataFrame tr:hover {
    background-color: #e6f3ff;
}
.sentiment-positive {
    color: #28a745;
    font-weight: bold;
}
.sentiment-negative {
    color: #dc3545;
    font-weight: bold;
}
.sentiment-neutral {
    color: #6c757d;
    font-weight: bold;
}
.truncated {
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)

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
            logger.warning("No news data fetched; initializing empty DataFrame")
            news_df, entities = pd.DataFrame(columns=['title', 'desc', 'date', 'link', 'media', 'full_text', 'sentiment', 'sentiment_score']), []
    else:
        news_df, entities = cache[cache_key]
        logger.info(f"Retrieved {len(news_df)} articles from cache for key {cache_key}")

# Log DataFrame contents for debugging
logger.info(f"news_df shape: {news_df.shape}")
if not news_df.empty:
    logger.info(f"news_df columns: {list(news_df.columns)}")
    logger.info(f"news_df sample: {news_df[['title', 'desc', 'date', 'sentiment']].head().to_dict()}")

# Display results
# Top Entities Word Cloud (First Section)
st.subheader("☁️ Top Entities Word Cloud")
cleaned_entity_counts = {
    re.sub(r'[\n\r]+', ' ', k).strip(): v for k, v in 
    {e[0]: entities.count(e) for e in entities if e[1] in entity_type}.items()
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
if not news_df.empty:
    sentiment_counts = news_df['sentiment'].value_counts().reset_index()
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
else:
    st.warning("No sentiment data available.")

# Unified Named Entity Explorer
st.subheader("🧠 Named Entity Explorer")
st.markdown("Explore all named entities (people, organizations, locations, events) in India's top news, stock market updates, and events.")
entity_data = []
for entity, entity_type in [e for e in entities if e[1] in entity_type]:
    entity_articles = news_df[news_df['full_text'].str.contains(re.escape(entity), case=False, na=False)]
    avg_sentiment = entity_articles['sentiment_score'].mean() if not entity_articles.empty else 0.0
    entity_data.append({
        'Entity': entity,
        'Type': entity_type,
        'Frequency': entities.count((entity, entity_type)),
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
entity_freq = pd.DataFrame(get_top_entities([e for e in entities if e[1] in entity_type], top_n=20), columns=["Entity", "Frequency"])
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
    map_html = create_geo_map([e for e in entities if e[1] == "GPE"])
    st.components.v1.html(map_html, height=300)
except Exception as e:
    logger.error(f"Error in geo-visualization: {str(e)}")
    st.warning("Unable to generate geographic map at this time.")

# Improved News Articles Table
st.subheader("📰 Top Indian News Articles")
if not news_df.empty:
    display_df = news_df[["title", "desc", "sentiment", "link"]].copy()
    
    # Ensure required columns exist
    for col in ['title', 'desc', 'sentiment', 'link']:
        if col not in display_df.columns:
            display_df[col] = ''
    logger.info(f"display_df shape: {display_df.shape}")
    
    # Truncate title and desc for display
    display_df['title_display'] = display_df['title'].apply(lambda x: x[:50] + '...' if isinstance(x, str) and len(x) > 50 else x)
    display_df['desc_display'] = display_df['desc'].apply(lambda x: x[:100] + '...' if isinstance(x, str) and len(x) > 100 else x)

    
    # Map sentiment to CSS classes
    sentiment_styles = {
        'Positive': 'sentiment-positive',
        'Negative': 'sentiment-negative',
        'Neutral': 'sentiment-neutral'
    }
    display_df['sentiment_display'] = display_df['sentiment'].map(lambda x: f'<span class="{sentiment_styles.get(x, "")}">{x}</span>' if x in sentiment_styles else x)
    
    # Create HTML for clickable titles
    display_df['title_display'] = display_df.apply(
        lambda row: f'<a href="{row["link"]}" target="_blank" title="{row["title"]}">{row["title_display"]}</a>' if isinstance(row["title"], str) else row["title"],
        axis=1
    )
    
    # Create HTML for links
    display_df['link_display'] = display_df['link'].apply(
        lambda x: f'<a href="{x}" target="_blank" title="{x}">Link</a>' if isinstance(x, str) and x else 'No Link'
    )
    
    # Select columns for display
    display_df = display_df[['title_display', 'desc_display', 'sentiment_display', 'link_display']]
    display_df.columns = ['Title', 'Description', 'Sentiment', 'Link']
    
    # Render table
    st.dataframe(
        display_df,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Link": st.column_config.TextColumn("Link", width="small")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No news articles available for the selected parameters. Try adjusting the query or time period.")