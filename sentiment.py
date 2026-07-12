from transformers import pipeline


def load_model():
    model = pipeline(
        "text-classification",
        model="ProsusAI/finbert"
    )
    return model


def get_sentiment(news_list, model):
    results = []
    
    for news in news_list:
        title = news["title"]
        
        # Sentiment analyze karo
        sentiment = model(title)[0]
        
        results.append({
            "title": title,
            "source": news["source"],
            "sentiment": sentiment["label"],
            "confidence": round(sentiment["score"] * 100, 2),
            "url": news["url"]
        })
    
    return results