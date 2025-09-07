# Real-Time Stock Prediction with Sentiment Analysis

This application predicts stock prices in real-time using sentiment analysis from news articles and Twitter.

## Features

- Real-time stock price visualization
- Sentiment analysis from news articles
- Sentiment analysis from Twitter
- Combined sentiment score
- Interactive dashboard

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock ticker (e.g., "AAPL" for Apple)
2. The dashboard will display:
   - Real-time stock price chart
   - Combined sentiment score
   - Individual sentiment scores from news and Twitter

## Requirements

- Python 3.8+
- Streamlit
- Yahoo Finance API
- NewsAPI
- Twitter API
- Various ML and visualization libraries
