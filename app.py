"""
Streamlit Stock Analyzer
- Search ticker using yfinance
- Interactive chart (candlestick / line / bars) with time range
- Common indicators: SMA, EMA, MACD, RSI, Volume
- Fundamentals snapshot (market cap, PE, EPS, sector, industry)
- Company name + ticker + logo (best-effort)
- Simple rule-based Buy/Sell/Hold signal
- Optional AI agent (requires OPENAI_API_KEY)
"""

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
from datetime import datetime, timedelta

load_dotenv()

st.set_page_config(layout="wide", page_title="Stock Analyzer")

# ----------------------
# Helper functions
@st.cache_data(ttl=600)
def fetch_ticker(ticker):
    """
    Fetch basic ticker info safely without triggering rate limit.
    Tries lightweight info first; falls back gracefully.
    """
    t = yf.Ticker(ticker)
    info = {}
    try:
        # Try fast_info (does not hit heavy endpoints)
        fi = t.fast_info
        info.update({
            "shortName": ticker.upper(),
            "longName": ticker.upper(),
            "marketCap": fi.get("market_cap"),
            "lastPrice": fi.get("last_price"),
            "currency": fi.get("currency"),
        })
        # Try normal info but with fail-safe
        try:
            raw_info = t.get_info()
            if raw_info:
                info.update(raw_info)
        except yf.YFRateLimitError:
            st.warning("⚠️ Yahoo Finance rate limit reached — using cached/basic data only.")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Error fetching ticker data: {e}")
    return t, info

def get_history(ticker, period="1y", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    return df

def compute_indicators(df):
    out = df.copy()
    if len(out) == 0:
        return out
    out['SMA_20'] = SMAIndicator(out['Close'], window=20).sma_indicator()
    out['SMA_50'] = SMAIndicator(out['Close'], window=50).sma_indicator()
    out['EMA_20'] = EMAIndicator(out['Close'], window=20).ema_indicator()
    macd = MACD(out['Close'])
    out['MACD'] = macd.macd()
    out['MACD_SIGNAL'] = macd.macd_signal()
    out['RSI_14'] = RSIIndicator(out['Close'], window=14).rsi()
    out['Volume'] = out['Volume']
    return out

def build_plot(df, chart_type="candlestick", add_ma=True):
    fig = go.Figure()
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Close'], name='Close'))

    # overlays
    if add_ma and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', opacity=0.8))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', opacity=0.8))

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=30, b=20))
    return fig

def compute_signal(df):
    """
    Simple rule-based signal:
    - Buy: MACD crosses above signal & RSI < 70 & Price > SMA50
    - Sell: MACD crosses below signal & RSI > 30 & Price < SMA50
    - Hold: otherwise
    This is a simplistic heuristic for demo only.
    """
    if df is None or df.empty or len(df) < 3:
        return "No data"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    macd_cross_up = (prev['MACD'] < prev['MACD_SIGNAL']) and (latest['MACD'] > latest['MACD_SIGNAL'])
    macd_cross_down = (prev['MACD'] > prev['MACD_SIGNAL']) and (latest['MACD'] < latest['MACD_SIGNAL'])

    price_above_sma = latest['Close'] > latest.get('SMA_50', latest['Close'])
    price_below_sma = latest['Close'] < latest.get('SMA_50', latest['Close'])

    rsi = latest.get('RSI_14', 50)

    if macd_cross_up and rsi < 70 and price_above_sma:
        return "BUY"
    if macd_cross_down and rsi > 30 and price_below_sma:
        return "SELL"
    return "HOLD"

def get_logo_url(info):
    # try yfinance info first
    logo = info.get('logo_url') or info.get('logo')
    if logo:
        return logo
    # fallback: try derive domain from website and use Clearbit (public endpoint)
    website = info.get('website') or info.get('shortName') or None
    if website:
        try:
            domain = website.replace('http://','').replace('https://','').split('/')[0]
            return f"https://logo.clearbit.com/{domain}"
        except Exception:
            pass
    # final fallback: blank (streamlit will show nothing)
    return None

# ----------------------
# App UI
# ----------------------
st.title("📈 Stock Analyzer — Streamlit + Yahoo Finance")
st.sidebar.header("Search & Settings")

# sidebar inputs
ticker_input = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL").upper()
period_map = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "Max": "max"
}
period_choice = st.sidebar.selectbox("Period", list(period_map.keys()), index=3)
interval_choice = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo", "1h"], index=0)
chart_type = st.sidebar.selectbox("Chart type", ["candlestick", "line", "bar"])
show_ma = st.sidebar.checkbox("Show moving averages (SMA20, SMA50)", value=True)
show_indicators = st.sidebar.checkbox("Show indicator panel (RSI / MACD)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("AI Agent (optional)")
use_ai = st.sidebar.checkbox("Enable AI agent (OpenAI)")
openai_key_input = st.sidebar.text_input("OpenAI API key (or set env OPENAI_API_KEY)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input

# Load data
with st.spinner(f"Fetching {ticker_input} ..."):
    ticker_obj, info = fetch_ticker(ticker_input)
    period = period_map[period_choice]
    df = get_history(ticker_input, period=period, interval=interval_choice)
    df = compute_indicators(df)

# Top header: company name, ticker, logo, sector/industry
col1, col2 = st.columns([4,1])
with col1:
    company_name = info.get('shortName') or info.get('longName') or ticker_input
    st.subheader(f"{company_name}  ({ticker_input})")
    sub_head = []
    if info.get('sector'):
        sub_head.append(info.get('sector'))
    if info.get('industry'):
        sub_head.append(info.get('industry'))
    if sub_head:
        st.write(" · ".join(sub_head))
with col2:
    logo_url = get_logo_url(info)
    if logo_url:
        try:
            st.image(logo_url, width=80)
        except:
            st.write("")  # no logo

st.markdown("---")

# Left: Chart and indicators; Right: fundamentals & AI
left, right = st.columns([3,1])

with left:
    if df is None or df.empty:
        st.error("No historical data found for this ticker / timeframe.")
    else:
        fig = build_plot(df, chart_type=chart_type, add_ma=show_ma)
        st.plotly_chart(fig, use_container_width=True)

        if show_indicators:
            # RSI plot
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI (14)'))
            rsi_fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)

            # MACD plot
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal'))
            macd_fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(macd_fig, use_container_width=True)

        # quick data table of latest candles
        st.subheader("Latest data")
        latest = df[['Open','High','Low','Close','Volume']].iloc[-10:].sort_index(ascending=False)
        st.dataframe(latest)

with right:
    st.subheader("Fundamentals & Snapshot")
    mcap = info.get('marketCap')
    pe = info.get('trailingPE') or info.get('forwardPE')
    eps = info.get('trailingEps') or info.get('epsTrailingTwelveMonths')
    div_yield = info.get('dividendYield')
    sector = info.get('sector')
    industry = info.get('industry')
    website = info.get('website')

    st.write(f"**Market cap:** {mcap if mcap else 'N/A'}")
    st.write(f"**P/E:** {pe if pe else 'N/A'}")
    st.write(f"**EPS:** {eps if eps else 'N/A'}")
    st.write(f"**Dividend yield:** {div_yield if div_yield else 'N/A'}")
    if sector:
        st.write(f"**Sector:** {sector}")
    if industry:
        st.write(f"**Industry:** {industry}")
    if website:
        st.write(f"[Website]({website})")

    st.markdown("---")
    st.subheader("Quick Signals")
    signal = compute_signal(df)
    if signal == "BUY":
        st.success("BUY — conditions favorable (heuristic)")
    elif signal == "SELL":
        st.error("SELL — conditions unfavorable (heuristic)")
    elif signal == "HOLD":
        st.info("HOLD — no clear signal (heuristic)")
    else:
        st.write(signal)

    st.markdown("---")
    st.subheader("Key metrics (recent)")
    # compute recent percent change, volatility
    if df is not None and not df.empty:
        pct_1d = df['Close'].pct_change().iloc[-1] * 100
        pct_7d = df['Close'].pct_change(7).iloc[-1] * 100 if len(df) > 7 else np.nan
        vol = df['Volume'].iloc[-20:].mean() if 'Volume' in df.columns else np.nan
        st.write(f"1d change: {pct_1d:.2f}%")
        if not np.isnan(pct_7d):
            st.write(f"7d change: {pct_7d:.2f}%")
        st.write(f"Avg volume (20): {int(vol) if not np.isnan(vol) else 'N/A'}")

# ----------------------
# AI Agent (optional)
# ----------------------
st.markdown("---")
st.header("AI Assistant (optional)")

if use_ai:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.warning("OpenAI API key not found. Set OPENAI_API_KEY in environment or enter it in sidebar.")
    else:
        openai.api_key = openai_key
        st.write("Ask the AI about this stock. The assistant can summarize fundamentals, create a short thesis, or generate questions to research.")
        user_q = st.text_area("Question or request for AI (e.g. 'Summarize recent drivers for AAPL' or 'What are risks?')", value=f"Summarize {ticker_input} fundamentals and recent technical signals.")
        if st.button("Run AI analysis"):
            with st.spinner("Contacting AI..."):
                # Build a short context for the assistant
                last_close = df['Close'].iloc[-1] if (df is not None and not df.empty) else "N/A"
                prompt = f"""
You are a helpful stock research assistant.

TICKER: {ticker_input}
COMPANY: {company_name}
LATEST CLOSE: {last_close}
SHORT INFO: {info.get('longBusinessSummary', '')[:800]}

User request:
{user_q}

Respond concisely (4-6 short paragraphs), mention technicals (RSI, MACD, SMA50) from the latest data if available, and add 3 short follow-up research questions the user should check.
"""
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini", # user may change model per their OpenAI usage
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=500,
                        temperature=0.2,
                    )
                    answer = resp['choices'][0]['message']['content']
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

else:
    st.info("Enable the AI Agent in the left sidebar to use OpenAI-powered analysis (requires API key).")

# Footer / notes
st.markdown("---")
st.caption("Data via Yahoo Finance (yfinance). This tool is for research/demo only — not financial advice. Improve the strategy and indicators before trading.")
