import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import requests
import os
from dotenv import load_dotenv
import openai

# -----------------------------------
# SETUP
# -----------------------------------
load_dotenv()
st.set_page_config(layout="wide", page_title="Stock Analyzer")

# -----------------------------------
# HELPERS
# -----------------------------------
@st.cache_data(ttl=600)
def fetch_ticker(ticker: str):
    """Fetch info safely, avoiding Yahoo rate limits."""
    info, fast_info = {}, {}
    try:
        t = yf.Ticker(ticker)

        # Lightweight data (fast_info never triggers rate limits)
        try:
            fast_info = t.fast_info or {}
        except Exception:
            fast_info = {}

        # Heavier data (info)
        try:
            raw_info = t.get_info()
            if isinstance(raw_info, dict):
                info.update(raw_info)
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg and "limit" in msg:
                st.warning("⚠️ Yahoo rate limit reached — showing limited company info.")
            else:
                st.info("⚠️ Limited company data available.")
    except Exception as e:
        st.error(f"Error fetching ticker data: {e}")

    return info, fast_info


@st.cache_data(ttl=600)
def get_history(ticker, period="1y", interval="1d"):
    """Fetch historical prices."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def compute_indicators(df):
    """Add technical indicators."""
    if df.empty:
        return df
    df = df.copy()
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['RSI_14'] = RSIIndicator(df['Close']).rsi()
    return df


def compute_signal(df):
    """Simple rule-based trading signal."""
    if df.empty or len(df) < 3:
        return "No data"
    latest, prev = df.iloc[-1], df.iloc[-2]
    macd_up = prev['MACD'] < prev['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']
    macd_down = prev['MACD'] > prev['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']
    price_above = latest['Close'] > latest.get('SMA_50', latest['Close'])
    price_below = latest['Close'] < latest.get('SMA_50', latest['Close'])
    rsi = latest.get('RSI_14', 50)

    if macd_up and rsi < 70 and price_above:
        return "BUY"
    if macd_down and rsi > 30 and price_below:
        return "SELL"
    return "HOLD"


def build_plot(df, chart_type="candlestick", show_ma=True):
    fig = go.Figure()
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open']()
