import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import requests
import os
import json
import openai

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(layout="wide", page_title="Pro Stock Analyzer")

# Your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdef1234567890abcdef12345678"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# SIDEBAR
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
# FETCH STOCK DATA
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    df = pd.DataFrame()
    company = {}
    fundamentals = {}
    # --- Yahoo historical prices ---
    try:
        df = yf.download(ticker, period=period_choice, interval=interval_choice, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
    except:
        st.warning("Could not fetch Yahoo data. Showing limited data.")

    # --- Finnhub for fundamentals and profile ---
    FINNHUB_TOKEN = "cd7v7iad3i8s1h8j7o30"
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
        metrics = metrics_r.json().get("metric",{})
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
    if df.empty or 'Close' not in df: return df
    df = df.copy()
    close = df['Close'].dropna()
    try:
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_SIGNAL'] = macd.macd_signal()
        df['RSI_14'] = RSIIndicator(close).rsi()
    except:
        st.warning("Could not compute technical indicators.")
    df.dropna(inplace=True)
    return df

def compute_signal(df):
    if df.empty or len(df)<2: return "No data"
    latest, prev = df.iloc[-1], df.iloc[-2]
    macd_up = 'MACD' in df.columns and prev['MACD']<prev['MACD_SIGNAL'] and latest['MACD']>latest['MACD_SIGNAL']
    macd_down = 'MACD' in df.columns and prev['MACD']>prev['MACD_SIGNAL'] and latest['MACD']<latest['MACD_SIGNAL']
    price_above = latest['Close']>latest.get('SMA_50', latest['Close'])
    price_below = latest['Close']<latest.get('SMA_50', latest['Close'])
    rsi = latest.get('RSI_14',50)
    if macd_up and rsi<70 and price_above: return "BUY"
    if macd_down and rsi>30 and price_below: return "SELL"
    return "HOLD"

# -----------------------------
# BUILD CHART
# -----------------------------
def build_chart(df, chart_type="candlestick"):
    if df.empty: return go.Figure()
    fig = go.Figure()
    if chart_type=="candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="Price"))
    elif chart_type=="line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines", name="Close", line=dict(color="#42a5f5", width=2)))
    # Moving averages
    if 'SMA_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode="lines", name="SMA 20", line=dict(color="orange")))
    if 'SMA_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode="lines", name="SMA 50", line=dict(color="purple")))
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    return fig

# -----------------------------
# AI STOCK ANALYSIS
# -----------------------------
def ai_stock_analysis(ticker, company, fundamentals, df):
    prompt = f"""
You are a professional stock market AI assistant.
TICKER: {ticker}
COMPANY: {company.get('name', ticker)}
SECTOR: {company.get('sector')}
INDUSTRY: {company.get('industry')}
MARKET CAP: {fundamentals.get('marketCap','N/A')}
PRICE DATA AVAILABLE: {not df.empty}

Provide a summary of the stock, recent performance, technical indicators (if available), and suggest BUY/HOLD/SELL reasoning.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {e}"

# -----------------------------
# FETCH & COMPUTE
# -----------------------------
df, company, fundamentals = fetch_stock_data(ticker_input)
df = compute_indicators(df)
signal = compute_signal(df)

# -----------------------------
# DASHBOARD
# -----------------------------
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
        if 'RSI_14' in df: st.line_chart(df['RSI_14'], height=200, use_container_width=True)
        if 'MACD' in df: st.line_chart(df[['MACD','MACD_SIGNAL']], height=200, use_container_width=True)
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
# AI Section
# -----------------------------
if use_ai:
    st.markdown("---")
    st.header("🤖 AI Stock Analysis")
    st.write(ai_stock_analysis(ticker_input, company, fundamentals, df))

st.markdown("---")
st.caption("Data via Yahoo Finance & Finnhub, logos via Clearbit, AI included. Not financial advice.")
