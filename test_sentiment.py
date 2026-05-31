"""Tests for data/sentiment.py"""
from unittest.mock import patch, MagicMock
from data.sentiment import SentimentFetcher


def test_fear_greed_returns_valid_range():
    sf = SentimentFetcher()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"value": "25", "value_classification": "Extreme Fear"}]
    }
    with patch("httpx.get", return_value=mock_resp):
        result = sf.get_fear_greed_index()
    assert result["value"] == 25
    assert result["classification"] == "Extreme Fear"


def test_score_sentiment_range():
    sf = SentimentFetcher()
    score = sf.score_sentiment(
        news_items=[
            {"votes_positive": 10, "votes_negative": 2},
            {"votes_positive": 1, "votes_negative": 8},
        ],
        fear_greed={"value": 20}
    )
    assert -1.0 <= score <= 1.0


def test_score_neutral_no_data():
    sf = SentimentFetcher()
    score = sf.score_sentiment(news_items=[], fear_greed={"value": 50})
    assert -0.1 <= score <= 0.1


def test_full_report_structure():
    sf = SentimentFetcher()
    with patch.object(sf, "get_fear_greed_index", return_value={"value": 30, "classification": "Fear"}):
        with patch.object(sf, "get_cryptopanic_news", return_value=[]):
            report = sf.get_full_sentiment_report("BTC")
    assert "score" in report
    assert "bias" in report
    assert report["bias"] in ("bullish", "bearish", "neutral")


if __name__ == "__main__":
    test_fear_greed_returns_valid_range()
    test_score_sentiment_range()
    test_score_neutral_no_data()
    test_full_report_structure()
    print("All sentiment tests passed.")
