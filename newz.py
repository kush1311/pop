import requests
import pandas as pd
from textblob import TextBlob
from tabulate import tabulate
import hashlib
from datetime import datetime, timedelta
 
# ---------------------------- YOUR API KEYS ----------------------------
NEWSDATA_API_KEY = "pub_79508cd750e96ccb61c080a4e004aca7439c3"
FINNHUB_API_KEY = "cvrokohr01qnpem8p570cvrokohr01qnpem8p57g"
GNEWS_API_KEY = "434440771cb4bd849b92821e037741ba"
 
# ---------------------------- SENTIMENT FUNCTION ----------------------------
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive", 1
    elif polarity < -0.1:
        return "Negative", -1
    else:
        return "Neutral", 0
 
# ---------------------------- NEWSDATA FETCHER ----------------------------
def fetch_newsdata():
    url = "https://newsdata.io/api/1/latest"
    params = {
        'apikey': NEWSDATA_API_KEY,
        'q': 'India stock market OR Indian economy OR SEBI OR NSE OR BSE OR Nifty OR Sensex OR RBI OR War Or '
        'India Pakistan conflict OR war OR China Taiwan OR oil prices OR inflation OR '
        'US Fed OR Trump tariffs OR global economy OR Middle East OR crash ',
        'country': 'in',
        'language': 'en',
        'category': 'business'
    }
    res = requests.get(url, params=params)
    news = []
    if res.status_code == 200:
        data = res.json()
        for item in data.get("results", []):
            sentiment_label, sentiment_score = classify_sentiment(item.get("title", ""))
            news.append({
                "title": item.get("title"),
                "published": item.get("pubDate"),
                "source": item.get("source_id", "newsdata.io"),
                "link": item.get("link"),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score
            })
    return news
 
# ---------------------------- FINNHUB FETCHER ----------------------------
def fetch_finnhub(symbols=['RELIANCE.NS', 'NSEI', 'TCS.NS', 'INFY.NS', 'SBIN.NS', "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "MARUTI.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBIN.NS", "NTPC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "BRITANNIA.NS", "M&M.NS", "SUNPHARMA.NS", "DIVISLAB.NS",
    "INDUSINDBK.NS", "TATAMOTORS.NS", "TITAN.NS", "DRREDDY.NS", "GRASIM.NS", "ADANIPORTS.NS",
    "ADANIENT.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "VEDL.NS", "SHREECEM.NS", "BAJAJAUTO.NS",
    "HEROMOTOCO.NS", "WIPRO.NS", "TECHM.NS", "COALINDIA.NS", "BPCL.NS", "GAIL.NS", "IOC.NS",
    "UPL.NS", "EICHERMOT.NS"]):
    url = "https://finnhub.io/api/v1/company-news"
    all_news = []
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    for symbol in symbols:
        params = {
            'symbol': symbol,
            'from': today,
            'to': today,
            'token': FINNHUB_API_KEY
        }
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                title = item.get("headline", "")
                sentiment_label, sentiment_score = classify_sentiment(title)
                all_news.append({
                    "title": title,
                    "published": pd.to_datetime(item.get("datetime"), unit='s'),
                    "source": item.get("source", "finnhub"),
                    "link": item.get("url"),
                    "sentiment": sentiment_label,
                    "sentiment_score": sentiment_score
                })
    return all_news
 
# ---------------------------- GNEWS FETCHER ----------------------------
def fetch_gnews():
    url = "https://gnews.io/api/v4/search"
    params = {
        'q': 'India stock OR Indian finance OR SEBI OR IPO OR RBI OR Indian shares OR Nifty OR Sensex'
        'India Pakistan war OR border conflict OR Trump tariffs OR US Fed OR oil prices OR inflation OR '
        'Middle East tensions OR China Taiwan OR global economy',
        'country': 'in',
        'lang': 'en',
        'token': GNEWS_API_KEY,
        'max': 50
    }
    res = requests.get(url, params=params)
    news = []
    if res.status_code == 200:
        data = res.json()
        for article in data.get("articles", []):
            sentiment_label, sentiment_score = classify_sentiment(article.get("title", ""))
            news.append({
                "title": article.get("title"),
                "published": article.get("publishedAt"),
                "source": article.get("source", {}).get("name", "gnews.io"),
                "link": article.get("url"),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score
            })
    return news
 
# ---------------------------- UNIFIED NEWS FETCHER ----------------------------
def get_news():
    print("Fetching news from Newsdata.io, Finnhub, and GNews with national Indian focus...")
 
    all_news_raw = fetch_newsdata() + fetch_finnhub() + fetch_gnews()
 
    # Deduplicate using hash of title
    seen = set()
    unique_news = []
    for article in all_news_raw:
        h = hashlib.sha256(article['title'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_news.append(article)
 
    # Convert to DataFrame
    df = pd.DataFrame(unique_news)
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
 
    # Filter for only today or yesterday
    now = pd.Timestamp.now()
    yesterday = now - timedelta(days=1)
    df = df[df['published'].between(yesterday.normalize(), now)]
 
    # Sort and format
    df = df.sort_values(by='published', ascending=False)
    display_df = df[['title', 'sentiment', 'sentiment_score', 'source', 'published', 'link']].copy()
    display_df['title'] = display_df['title'].str.slice(0, 90) + "..."
    print(tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=True))
 
    return df
 
# ---------------------------- RUN ----------------------------
if __name__ == "__main__":
    get_news()