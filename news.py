

import requests
def get_news(stock_name, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&apiKey={api_key}"

    
    response = requests.get(url)
    data = response.json()
    
    print(data)  # Terminal mein dekho kya aa raha hai
    
    if data.get("status") != "ok":
        print(f"Error: {data.get('message')}")
        return []
    
    articles = []
    for article in data["articles"][:10]:
        articles.append({
            "title": article["title"],
            "source": article["source"]["name"],
            "url": article["url"]
        })
    
    return articles
