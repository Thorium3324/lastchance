import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import requests, os, json
import openai

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(layout="wide", page_title="Pro Stock Analyzer")
os.environ["OPENAI_API_KEY"] = "sk-abcdef1234567890abcdef1234567890abcdef12"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Stock & Settings")
ticker_input = st.sidebar.text_input("Ticker (AAPL, MSFT, TSLA)", "AAPL").upper()
period_choice = st.sidebar.selectbox("Historical Period", ["1mo","3mo","6mo","1y","2y","5y","max"], index=3)
interval_choice = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
chart_type = st.sidebar.selectbox("Chart Type", ["candlestick","line","bar"])
show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_indicators = st.sidebar.checkbox("Show Indicators (RSI, MACD)", True)

st.sidebar.header("AI Assistant")
use_ai = st.sidebar.checkbox("Enable AI Stock Assistant")

# -----------------------------
# FETCH DATA FUNCTION
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    # Historical prices from Yahoo
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        df.dropna(inplace=True)
    except:
        df = pd.DataFrame()
    # Company info and fundamentals from Finnhub
    FINNHUB_TOKEN = "cd7v7iad3i8s1h8j7o30"
    company = {}
    fundamentals = {}
    try:
        profile_r = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_TOKEN}")
        profile = profile_r.json()
        company = {
            "name": profile.get("name") or ticker,
            "sector": profile.get("finnhubIndustry"),
            "industry": profile.get("industry"),
            "website": profile.get("weburl"),
            "logo": profile.get("logo")
        }
    except:
        company = {"name": ticker}
    try:
        metrics_r = requests.get(f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_TOKEN}")
        metrics = metrics_r.json().get("metric", {})
        fundamentals = {
            "marketCap": metrics.get("marketCapitalization"),
            "peRatio": metrics.get("peBasicExclExtraTTM"),
            "eps": metrics.get("epsExclExtraItemsTTM"),
            "dividendYield": metrics.get("dividendsPerShareTTM") / metrics.get("price",1) if metrics.get("dividendsPerShareTTM") else None
        }
    except:
        fundamentals = {}
    return df, company, fundamentals

# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
def compute_indicators(df):
    if df.empty: return df
    df = df.copy()
    if 'Close' not in df: return df
    close = df['Close'].dropna()
    try:
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_SIGNAL'] = macd.macd_signal()
        df['RSI_14'] = RSIIndicator(close).rsi()
    except: pass
    df.dropna(inplace=True)
    return df

def compute_signal(df):
    """Safe rule-based signal computation."""
    if df.empty or len(df)<2:
        return "No data"

    required_cols = ['MACD', 'MACD_SIGNAL', 'SMA_50', 'RSI_14', 'Close']
    for col in required_cols:
        if col not in df.columns:
            return "No data"

    latest, prev = df.iloc[-1], df.iloc[-2]

    macd_up = prev['MACD'] < prev['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']
    macd_down = prev['MACD'] > prev['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']
    price_above = latest['Close'] > latest['SMA_50']
    price_below = latest['Close'] < latest['SMA_50']
    rsi = latest['RSI_14']

    if macd_up and rsi < 70 and price_above:
        return "BUY"
    if macd_down and rsi > 30 and price_below:
        return "SELL"
    return "HOLD"
# -----------------------------
# CHART
# -----------------------------
def build_chart(df, chart_type="candlestick"):
    if df.empty: return go.Figure()
    fig = go.Figure()
    if chart_type=="candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="Price"))
    elif chart_type=="line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines", name="Close", line=dict(color="#42a5f5", width=2)))
    elif chart_type=="bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Close", marker_color="#42a5f5"))
    # Moving averages
    if 'SMA_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode="lines", name="SMA 20", line=dict(color="orange")))
    if 'SMA_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode="lines", name="SMA 50", line=dict(color="purple")))
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    return fig

# -----------------------------
# AI NEWS SENTIMENT
# -----------------------------
def analyze_news_sentiment(ticker):
    query = f"Summarize latest news and sentiment for {ticker}."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":query}],
            max_tokens=300,
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except:
        return "AI News sentiment unavailable."

# -----------------------------
# FETCH DATA
# -----------------------------
df, company, fundamentals = fetch_stock_data(ticker_input)
df = compute_indicators(df)
signal = compute_signal(df)

# -----------------------------
# DASHBOARD UI
# -----------------------------
st.title(f"📈 Pro Stock Analyzer — {ticker_input}")

# Company Header
col1,col2 = st.columns([4,1])
with col1:
    st.subheader(f"{company.get('name', ticker_input)} ({ticker_input})")
    subinfo = [x for x in [company.get('sector'), company.get('industry')] if x]
    if subinfo: st.caption(" · ".join(subinfo))
with col2:
    if company.get('logo'): st.image(company['logo'], width=80)

st.markdown("---")
left,right = st.columns([3,1])

with left:
    if df.empty: st.error("No price data available")
    else: st.plotly_chart(build_chart(df, chart_type), use_container_width=True)
    if show_indicators and not df.empty:
        st.write("### RSI & MACD")
        rsi_fig = go.Figure(); rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI (14)")); st.plotly_chart(rsi_fig)
        macd_fig = go.Figure(); macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD")); macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name="Signal")); st.plotly_chart(macd_fig)

with right:
    st.subheader("Fundamentals")
    st.write(f"**Market Cap:** {fundamentals.get('marketCap','N/A')}")
    st.write(f"**P/E:** {fundamentals.get('peRatio','N/A')}")
    st.write(f"**EPS:** {fundamentals.get('eps','N/A')}")
    st.write(f"**Dividend Yield:** {fundamentals.get('dividendYield','N/A')}")
    if company.get("website"): st.markdown(f"[Website]({company['website']})")
    st.subheader("Signal")
    if signal=="BUY": st.success("BUY — bullish")
    elif signal=="SELL": st.error("SELL — bearish")
    elif signal=="HOLD": st.info("HOLD — neutral")
    else: st.write(signal)

# -----------------------------
# AI NEWS SENTIMENT
# -----------------------------
if use_ai:
    st.markdown("---")
    st.header("🤖 AI News Sentiment")
    sentiment = analyze_news_sentiment(ticker_input)
    st.write(sentiment)

st.markdown("---")
st.caption("Data via Yahoo Finance & Finnhub, logos via Clearbit, AI powered sentiment. Not financial advice.")
