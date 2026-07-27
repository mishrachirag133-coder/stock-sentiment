import streamlit as st
from news import get_news
from sentiment import load_model, get_sentiment
from prediction import predict_price
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title=" Stock Sentiment & Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment & Price Predictor")
st.write("Real-time stock sentiment analysis and price prediction powered by AI.")

# API Key
try:
    api_key = st.secrets["NEWS_API_KEY"]
except:
    api_key = "3dd1b5a825664f0ca7f2c89f6cda0036"

@st.cache_resource
def load():
    return load_model()

model = load()



with st.form("analyze_form"):
    stock = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g. TCS, RELIANCE, INFY"
    )
    timeframe = st.selectbox(
        "Prediction Timeframe",
        ["2 Hours", "4 Hours", "1 Day", "4 Days"]
    )
    analyze_clicked = st.form_submit_button("Analyze")

if analyze_clicked:
    if stock:
        
        if timeframe == "2 Hours":
            steps, interval = 2, "1h"
        elif timeframe == "4 Hours":
            steps, interval = 4, "1h"
        elif timeframe == "1 Day":
            steps, interval = 1, "1d"
        elif timeframe == "4 Days":
            steps, interval = 4, "1d"
        elif timeframe == "2 Days":
            steps, interval = 2, "1d"

       
        st.subheader(f"📊 Price Prediction — {stock.upper()}")

        with st.spinner("Fetching market data and running LSTM model..."):
            current, predicted, direction = predict_price(
                stock, steps=steps, interval=interval
            )

        if current and predicted:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Market Price", f"₹{current:.2f}")

            with col2:
                st.metric(f"Predicted Price ({timeframe})", f"₹{predicted:.2f}")

            with col3:
                change = predicted - current
                change_pct = (change / current) * 100
                st.metric(
                    "Expected Movement",
                    direction,
                    delta=f"{change_pct:.2f}%"
                )
        else:
            st.error("Unable to fetch price data. Please use valid stock name.")

        st.divider()

        # News Sentiment
        st.subheader(f" News Sentiment Analysis — {stock.upper()}")

        with st.spinner("Fetching latest news articles..."):
            news = get_news(stock, api_key)

        with st.spinner():
            results = get_sentiment(news, model)

        if not results:
            st.warning("No relevant news found. Please try a different stock name.")
        else:
            for r in results:
                confidence = r["confidence"]

                if confidence >= 80:
                    strength = " Very Strong"
                elif confidence >= 60:
                    strength = " Strong"
                else:
                    strength = " Weak Signal"

                if r["sentiment"] == "positive":
                    st.success(f"✅ {r['title']}")
                    st.caption(f"Sentiment: Positive | Signal Strength: {strength} | Source: {r['source']}")

                elif r["sentiment"] == "negative":
                    st.error(f"❌ {r['title']}")
                    st.caption(f"Sentiment: Negative | Signal Strength: {strength} | Source: {r['source']}")

                else:
                    st.warning(f"⚠️ {r['title']}")
                    st.caption(f"Sentiment: Neutral | Signal Strength: {strength} | Source: {r['source']}")

    else:
        st.error("Please enter a valid stock name to proceed.")
