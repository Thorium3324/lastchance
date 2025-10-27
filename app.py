import streamlit as st
import yfinance as yf
import pandas as pd
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
st.set_page_config(layout="wide", page_title="Stock Analyzer — Pro Dashboard")

# Use your API key directly (for this purpose only)
os.environ["OPENAI_API_KEY"] = "sk-abcdef1234567890abcdef1234567890abcdef12"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# FETCHERS
# -----------------------------
@st.cache_data(ttl=43200)
def fetch_ticker_info(ticker: str):
    """Fetch ticker info safely with caching."""
    info, fast_info = {}, {}
    try:
        t = yf.Ticker(ticker)
        try:
            fast_info = dict(t.fast_info)
        except Exception:
            pass
        try:
            raw_info = t.get_info()
            if isinstance(raw_info, dict):
                info = json.loads(json.dumps(raw_info))
        except Exception:
            st.warning("⚠️ Yahoo rate limit reached — showing limited company info.")
    except Exception as e:
        st.error(f"Error fetching ticker data: {e}")
    return info or {}, fast_info or {}

@st.cache_data(ttl=3600)
def get_history(ticker, period="1y", interval="1d"):
    """Fetch historical price data."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    """Compute SMA, EMA, MACD, RSI."""
    if df.empty: return df
    df = df.copy()
    if 'Close' not in df: return df
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
    close = close.dropna()
    try:
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_SIGNAL'] = macd.macd_signal()
        df['RSI_14'] = RSIIndicator(close).rsi()
    except Exception:
        pass
    df.dropna(inplace=True)
    return df

def compute_signal(df):
    if df.empty or len(df)<2: return "No data"
    latest, prev = df.iloc[-1], df.iloc[-2]
    macd_up = prev['MACD'] < prev['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']
    macd_down = prev['MACD'] > prev['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']
    price_above = latest['Close'] > latest.get('SMA_50', latest['Close'])
    price_below = latest['Close'] < latest.get('SMA_50', latest['Close'])
    rsi = latest.get('RSI_14',50)
    if macd_up and rsi < 70 and price_above: return "BUY"
    if macd_down and rsi > 30 and price_below: return "SELL"
    return "HOLD"

# -----------------------------
# CHART
# -----------------------------
def build_plot(df, chart_type="candlestick", show_ma=True, show_volume=True):
    if df.empty: return go.Figure()
    fig = go.Figure()
    if chart_type=="candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="Price"))
    elif chart_type=="line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines", name="Close", line=dict(color="#42a5f5", width=2)))
    elif chart_type=="bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Close", marker_color="#42a5f5"))
    # MA
    if show_ma:
        if 'SMA_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode="lines", name="SMA 20", line=dict(color="orange", width=1.5)))
        if 'SMA_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode="lines", name="SMA 50", line=dict(color="purple", width=1.5)))
    # Signals
    buy,sell=[],[]
    if all(c in df.columns for c in ['MACD','MACD_SIGNAL','RSI_14','SMA_50']):
        for i in range(1,len(df)):
            prev,curr=df.iloc[i-1],df.iloc[i]
            macd_up=prev['MACD']<prev['MACD_SIGNAL'] and curr['MACD']>curr['MACD_SIGNAL']
            macd_down=prev['MACD']>prev['MACD_SIGNAL'] and curr['MACD']<curr['MACD_SIGNAL']
            price_above=curr['Close']>curr['SMA_50']
            price_below=curr['Close']<curr['SMA_50']
            rsi=curr['RSI_14']
            if macd_up and rsi<70 and price_above: buy.append((df.index[i], curr['Low']*0.98))
            elif macd_down and rsi>30 and price_below: sell.append((df.index[i], curr['High']*1.02))
    if buy: fig.add_trace(go.Scatter(x=[x[0] for x in buy], y=[x[1] for x in buy], mode="markers", name="BUY", marker=dict(symbol="triangle-up", color="lime", size=12)))
    if sell: fig.add_trace(go.Scatter(x=[x[0] for x in sell], y=[x[1] for x in sell], mode="markers", name="SELL", marker=dict(symbol="triangle-down", color="red", size=12)))
    if show_volume and 'Volume' in df:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,149,237,0.4)", yaxis='y2'))
        fig.update_layout(yaxis=dict(domain=[0.25,1], title="Price"), yaxis2=dict(domain=[0,0.2], title="Volume", showgrid=False))
    else: fig.update_layout(yaxis=dict(title="Price"))
    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
    return fig

# -----------------------------
# FUNDAMENTALS FALLBACK
# -----------------------------
@st.cache_data(ttl=43200)
def fetch_fundamentals_fallback(ticker):
    data={}
    try:
        FINNHUB_TOKEN = "cd7v7iad3i8s1h8j7o30"  # free token fallback
        r = requests.get(f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_TOKEN}")
        if r.status_code==200:
            m=r.json().get("metric",{})
            data["marketCap"]=m.get("marketCapitalization")
            data["peRatio"]=m.get("peBasicExclExtraTTM")
            data["eps"]=m.get("epsExclExtraItemsTTM")
            data["dividendYield"]=m.get("dividendsPerShareTTM")/m.get("price",1) if m.get("dividendsPerShareTTM") else None
    except Exception:
        pass
    return data

# -----------------------------
# COMPANY INFO FALLBACK
# -----------------------------
@st.cache_data(ttl=43200)
def get_company_info_fallback(ticker, info_dict):
    name=info_dict.get("shortName") or info_dict.get("longName") or ticker
    sector=info_dict.get("sector")
    industry=info_dict.get("industry")
    website=info_dict.get("website")
    logo=info_dict.get("logo_url") or info_dict.get("logo")
    if not logo and website:
        try:
            domain=website.replace("http://","").replace("https://","").split("/")[0]
            logo=f"https://logo.clearbit.com/{domain}"
        except: pass
    return {"name":name,"sector":sector,"industry":industry,"website":website,"logo":logo}

# -----------------------------
# SIDEBAR & INPUT
# -----------------------------
st.sidebar.header("Search & Settings")
ticker_input = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TSLA)", "AAPL").upper()
period_choice = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
interval_choice = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
chart_type = st.sidebar.selectbox("Chart type", ["candlestick", "line", "bar"])
show_ma = st.sidebar.checkbox("Show moving averages", True)
show_indicators = st.sidebar.checkbox("Show indicators (RSI, MACD)", True)

# AI Section
st.sidebar.header("AI Assistant (optional)")
use_ai = st.sidebar.checkbox("Enable AI agent")
