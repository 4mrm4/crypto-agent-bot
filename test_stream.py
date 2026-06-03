"""Tests for data/stream.py — MarketDataStream."""

from unittest.mock import AsyncMock, MagicMock, patch

from data.stream import MarketDataStream


def make_24hr_ticker(price="50000", high="51000", low="49000",
                      volume="1000", change="250"):
    return {
        "e": "24hrTicker", "E": 123456,
        "c": price, "h": high, "l": low, "v": volume, "p": change,
    }


def make_kline(open_p="50000", high="50500", low="49800", close="50200",
               volume="100", closed=True, interval="1h"):
    return {
        "e": "kline", "E": 123456,
        "k": {
            "o": open_p, "h": high, "l": low, "c": close,
            "v": volume, "i": interval, "x": closed, "T": 123456789,
        },
    }


def make_depth(bids=None, asks=None):
    if bids is None:
        bids = [["50000", "10"], ["49900", "20"]]
    if asks is None:
        asks = [["50100", "15"], ["50200", "5"]]
    return {"e": "depthUpdate", "b": bids, "a": asks}


def make_agg_trade(price="50100", volume="0.5"):
    return {"e": "aggTrade", "p": price, "q": volume}


class TestInit:
    def test_default_init(self):
        ms = MarketDataStream()
        assert ms._base_url == "wss://stream.binance.com:9443/ws"
        assert not ms._connected
        assert ms._latest_data == {}
        assert ms._subscribers == {}


class TestStreamConstants:
    def test_known_streams(self):
        ms = MarketDataStream()
        assert "ticker" in ms.STREAMS
        assert "kline_1h" in ms.STREAMS
        assert "depth" in ms.STREAMS
        assert "trades" in ms.STREAMS
        assert len(ms.STREAMS) == 6


class TestDispatch:
    def test_dispatch_ticker(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_24hr_ticker()))
        assert "ticker" in ms._latest_data

    def test_dispatch_kline(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_kline()))
        assert "kline_1h" in ms._latest_data

    def test_dispatch_kline_5m(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_kline(interval="5m")))
        assert "kline_5m" in ms._latest_data

    def test_dispatch_depth(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_depth()))
        assert "depth" in ms._latest_data

    def test_dispatch_trades(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_agg_trade()))
        assert "trades" in ms._latest_data

    def test_dispatch_non_dict(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch("not_a_dict"))
        assert ms._latest_data == {}

    def test_dispatch_to_subscribers(self):
        ms = MarketDataStream()
        import asyncio

        q = asyncio.Queue()
        ms._subscribers["ticker"] = [q]
        asyncio.run(ms._dispatch(make_24hr_ticker()))
        assert not q.empty()
        received = asyncio.run(q.get())
        assert received["e"] == "24hrTicker"

    def test_dispatch_multiple_subscribers(self):
        ms = MarketDataStream()
        import asyncio

        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        ms._subscribers["ticker"] = [q1, q2]
        asyncio.run(ms._dispatch(make_24hr_ticker()))
        assert not q1.empty()
        assert not q2.empty()

    def test_dispatch_unknown_event(self):
        ms = MarketDataStream()
        import asyncio
        unknown = {"e": "unknownEvent"}
        asyncio.run(ms._dispatch(unknown))
        assert "unknownEvent" in ms._latest_data


class TestGetLatestCandle:
    def test_no_data(self):
        ms = MarketDataStream()
        assert ms.get_latest_candle("BTCUSDT") is None

    def test_returns_candle(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_kline(close="50200", volume="100")))
        candle = ms.get_latest_candle("BTCUSDT")
        assert candle is not None
        assert candle["close"] == 50200.0
        assert candle["volume"] == 100.0
        assert candle["closed"] is True

    def test_specific_timeframe(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_kline(interval="5m", close="50500")))
        candle = ms.get_latest_candle("BTCUSDT", "5m")
        assert candle is not None
        assert candle["close"] == 50500.0

    def test_wrong_timeframe_returns_none(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_kline(interval="1h", close="50200")))
        candle = ms.get_latest_candle("BTCUSDT", "5m")
        assert candle is None


class TestGetOrderBookImbalance:
    def test_no_data_returns_neutral(self):
        ms = MarketDataStream()
        assert ms.get_order_book_imbalance("BTCUSDT") == 0.5

    def test_more_bids_imbalance(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_depth(
            bids=[["50000", "100"], ["49900", "50"]],
            asks=[["50100", "10"], ["50200", "5"]],
        )))
        imb = ms.get_order_book_imbalance("BTCUSDT")
        assert imb > 0.5  # more bid volume

    def test_more_asks_imbalance(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_depth(
            bids=[["50000", "10"], ["49900", "5"]],
            asks=[["50100", "100"], ["50200", "50"]],
        )))
        imb = ms.get_order_book_imbalance("BTCUSDT")
        assert imb < 0.5  # more ask volume

    def test_equal_volume(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_depth(
            bids=[["50000", "10"]],
            asks=[["50100", "10"]],
        )))
        imb = ms.get_order_book_imbalance("BTCUSDT")
        assert imb == 0.5

    def test_zero_volume(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_depth(bids=[], asks=[])))
        imb = ms.get_order_book_imbalance("BTCUSDT")
        assert imb == 0.5


class TestGetTicker:
    def test_no_data(self):
        ms = MarketDataStream()
        assert ms.get_ticker("BTCUSDT") is None

    def test_returns_ticker(self):
        ms = MarketDataStream()
        import asyncio
        asyncio.run(ms._dispatch(make_24hr_ticker(price="50200", high="51000",
                                                    low="49800", volume="1500",
                                                    change="200")))
        ticker = ms.get_ticker("BTCUSDT")
        assert ticker is not None
        assert ticker["price"] == 50200.0
        assert ticker["high"] == 51000.0
        assert ticker["volume"] == 1500.0


class TestSubscribe:
    def test_subscribe_creates_queue(self):
        ms = MarketDataStream()
        import asyncio
        q = asyncio.run(ms.subscribe("ticker"))
        assert isinstance(q, asyncio.Queue)
        assert "ticker" in ms._subscribers
        assert len(ms._subscribers["ticker"]) == 1

    def test_subscribe_multiple_same_channel(self):
        ms = MarketDataStream()
        import asyncio
        q1 = asyncio.run(ms.subscribe("ticker"))
        q2 = asyncio.run(ms.subscribe("ticker"))
        assert len(ms._subscribers["ticker"]) == 2


class TestIsConnected:
    def test_not_connected_by_default(self):
        ms = MarketDataStream()
        assert not ms.is_connected

    def test_connected_flag(self):
        ms = MarketDataStream()
        ms._connected = True
        assert ms.is_connected
