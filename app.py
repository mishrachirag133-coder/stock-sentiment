import streamlit as st
from news import get_news
from sentiment import load_model,get_sentiment
# Page setup
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analyzer")
st.write("Stock ki news ka sentiment analyze karo!")

# API Key
api_key = st.secrets["NEWS_API_KEY"]


@st.cache_resource
def load():
    return load_model()

model = load()


stock = st.text_input("Stock ka naam likho (e.g. TCS, Reliance, RVNL)")

if st.button("Analyze"):
    if stock:
        with st.spinner("News fetch ho rahi hai..."):
            # News lao
            news = get_news(stock, api_key)
            
        with st.spinner("Sentiment analyze ho raha hai..."):
            # Sentiment nikalo
            results = get_sentiment(news, model)
        
        # Results dikhao
        st.subheader(f"{stock} ke baare mein Latest News:")
        
        for r in results:
            confidence = r["confidence"]
            
            if confidence >= 80:
                strength = " Bahut Strong"
            elif confidence >= 60:
                strength = "Strong"
            else:
                strength = " Weak Signal"
            
            if r["sentiment"] == "positive":
                st.success(f"✅ {r['title']}")
                st.caption(f"Signal: Positive | Strength: {strength}")
                
            elif r["sentiment"] == "negative":
                st.error(f"❌ {r['title']}")
                st.caption(f"Signal: Negative | Strength: {strength}")
                
            else:
                st.warning(f"⚠️ {r['title']}")
                st.caption(f"Signal: Neutral | Strength: {strength}")

    else:
        st.error("Stock ka naam likho pehle!")
