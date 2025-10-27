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
import json

# -----------------------------------
# SETUP
# -----------------------------------
load_dotenv()
st.set_page_config(layout="wide", page_title="Stock Analyzer")

# -----------------------------------
# SAFE FETCHERS
# -----------------------------------
@st.cache_data(ttl=600)
def fetch_ticker_safe(ticker: str):
    """Fetch info safely and return only serializable dicts."""
    info, fast_info = {}, {}
    try:
        t = yf.Ticker(ticker)

        # Try light, serializable parts
        try:
            fi = t.fast_info
            if fi:
                fast_info = dict(fi)
        except Exception:
            pass

        try:
            raw_info = t.get_info()
            if isinstance(raw_info, dict):
                info = json.loads(json.dumps(raw_info))  # force serialization
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg and "limit" in msg:
                st.warning("⚠️ Yahoo rate limit reached — showing limited company info.")
            else:
                st.info("⚠️ Limited company data available.")
    except Exception as e:
        st.error(f"Error fetching ticker data: {e}")

    # ensure always serializable
    return info or {}, fast_info or {}


@st.cache_data(ttl=600)
def get_history(ticker, period="1y", interval="1d"):
    """Fetch historical prices."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
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
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Close'], name='Close'))

    if show_ma:
        if 'SMA_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
        if 'SMA_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=30, b=20))
    return fig


@st.cache_data(ttl=3600)
def get_company_info_fallback(ticker, info_dict):
    """Fetch logo and industry using fallback APIs if Yahoo data missing."""
    name = info_dict.get("shortName") or info_dict.get("longName") or ticker
    sector = info_dict.get("sector")
    industry = info_dict.get("industry")
    website = info_dict.get("website")
    logo_url = info_dict.get("logo_url") or info_dict.get("logo")

    # Try Clearbit for logo
    if not logo_url and website:
        try:
            domain = website.replace("http://", "").replace("https://", "").split("/")[0]
            logo_url = f"https://logo.clearbit.com/{domain}"
        except Exception:
            pass

    # Try Finnhub fallback
    if (not sector or not industry) and ticker.isalpha():
        try:
            r = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token=cd7v7iad3i8s1h8j7o30")
            if r.status_code == 200:
                data = r.json()
                sector = sector or data.get("finnhubIndustry")
                logo_url = logo_url or data.get("logo")
                website = website or data.get("weburl")
        except Exception:
            pass

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "website": website,
        "logo": logo_url,
    }

# -----------------------------------
# UI
# -----------------------------------
st.title("📈 Stock Analyzer — Professional Stock Insights")
st.sidebar.header("Search & Settings")

ticker_input = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL").upper()
period_choice = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
interval_choice = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
chart_type = st.sidebar.selectbox("Chart type", ["candlestick", "line", "bar"])
show_ma = st.sidebar.checkbox("Show moving averages", True)
show_indicators = st.sidebar.checkbox("Show indicators (RSI, MACD)", True)

# AI Section
st.sidebar.header("AI Assistant (optional)")
use_ai = st.sidebar.checkbox("Enable AI agent")
openai_key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input

# -----------------------------------
# FETCH DATA
# -----------------------------------
info, fast_info = fetch_ticker_safe(ticker_input)
df = get_history(ticker_input, period_choice, interval_choice)
df = compute_indicators(df)
company = get_company_info_fallback(ticker_input, info)

# -----------------------------------
# HEADER DISPLAY
# -----------------------------------
col1, col2 = st.columns([4, 1])
with col1:
    st.subheader(f"{company['name']} ({ticker_input})")
    subinfo = [x for x in [company['sector'], company['industry']] if x]
    if subinfo:
        st.caption(" · ".join(subinfo))
with col2:
    if company['logo']:
        st.image(company['logo'], width=80)

st.markdown("---")

# -----------------------------------
# CHARTS & FUNDAMENTALS
# -----------------------------------
left, right = st.columns([3, 1])
with left:
    if df.empty:
        st.error("No price data found.")
    else:
        st.plotly_chart(build_plot(df, chart_type, show_ma), use_container_width=True)

        if show_indicators:
            # RSI chart
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI (14)"))
            rsi_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)

            # MACD chart
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name="Signal"))
            macd_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(macd_fig, use_container_width=True)

with right:
    st.subheader("Fundamentals")
    st.write(f"**Market cap:** {info.get('marketCap') or fast_info.get('market_cap', 'N/A')}")
    st.write(f"**P/E:** {info.get('trailingPE') or info.get('forwardPE', 'N/A')}")
    st.write(f"**EPS:** {info.get('trailingEps') or info.get('epsTrailingTwelveMonths', 'N/A')}")
    st.write(f"**Dividend yield:** {info.get('dividendYield', 'N/A')}")
    if company["website"]:
        st.markdown(f"[Website]({company['website']})")

    st.subheader("Signal")
    signal = compute_signal(df)
    if signal == "BUY":
        st.success("BUY — bullish conditions (heuristic)")
    elif signal == "SELL":
        st.error("SELL — bearish conditions (heuristic)")
    elif signal == "HOLD":
        st.info("HOLD — neutral conditions")
    else:
        st.write(signal)

# -----------------------------------
# AI ASSISTANT
# -----------------------------------
st.markdown("---")
st.header("🤖 AI Stock Assistant")

if use_ai:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        openai.api_key = openai_key
        user_q = st.text_area("Ask the AI about this stock:", f"Summarize fundamentals and outlook for {ticker_input}.")
        if st.button("Analyze with AI"):
            with st.spinner("Thinking..."):
                try:
                    last_close = df['Close'].iloc[-1] if not df.empty else "N/A"
                    prompt = f"""
You are a financial analyst AI.
TICKER: {ticker_input}
COMPANY: {company['name']}
LAST CLOSE: {last_close}
SECTOR: {company.get('sector')}
INDUSTRY: {company.get('industry')}
MARKET CAP: {info.get('marketCap') or fast_info.get('market_cap')}

Summarize the stock fundamentals, recent performance, and possible outlook.
Mention key indicators (RSI, MACD, SMA) if available and end with 3 short follow-up research questions.
"""
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.3,
                    )
                    st.markdown(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"AI request failed: {e}")
else:
    st.info("Enable the AI assistant from the sidebar to analyze the selected stock.")

st.markdown("---")
st.caption("Data via Yahoo Finance, logos via Clearbit/Finnhub. Educational use only — not financial advice.")
