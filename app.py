# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import requests, os
import openai

st.set_page_config(layout="wide", page_title="Pro Stock Analyzer")

# OpenAI key
os.environ["OPENAI_API_KEY"] = "sk-abcdef1234567890abcdef1234567890abcdef12"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sidebar inputs
st.sidebar.header("Stock & Settings")
ticker_input = st.sidebar.text_input("Ticker", "AAPL").upper()
period_choice = st.sidebar.selectbox("Historical Period", ["1mo","3mo","6mo","1y","2y","5y","max"], index=3)
interval_choice = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
chart_type = st.sidebar.selectbox("Chart Type", ["candlestick","line","bar"])
show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_indicators = st.sidebar.checkbox("Show Indicators (RSI, MACD)", True)

st.sidebar.header("AI Assistant")
use_ai = st.sidebar.checkbox("Enable AI Stock Assistant")

# -----------------------------
# FETCH DATA
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    # Historical
    try:
        df = yf.download(ticker, period=period_choice, interval=interval_choice, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
    except:
        df = pd.DataFrame()

    # Company profile & fundamentals (Finnhub)
    FINNHUB_TOKEN = "cd7v7iad3i8s1h8j7o30"
    company, fundamentals = {}, {}
    try:
        profile = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_TOKEN}").json()
        metrics = requests.get(f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_TOKEN}").json().get("metric",{})
        company = {
            "name": profile.get("name") or ticker,
            "sector": profile.get("finnhubIndustry"),
            "industry": profile.get("industry"),
            "website": profile.get("weburl"),
            "logo": profile.get("logo")
        }
        fundamentals = {
            "marketCap": metrics.get("marketCapitalization"),
            "peRatio": metrics.get("peBasicExclExtraTTM"),
            "eps": metrics.get("epsExclExtraItemsTTM"),
            "dividendYield": metrics.get("dividendsPerShareTTM")/metrics.get("price",1) if metrics.get("dividendsPerShareTTM") else None
        }
    except:
        company = {"name": ticker}

    return df, company, fundamentals

df, company, fundamentals = fetch_stock_data(ticker_input)

# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
def compute_indicators(df):
    if df.empty or 'Close' not in df: return df
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]
    try:
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_SIGNAL'] = macd.macd_signal()
        df['RSI_14'] = RSIIndicator(close).rsi()
    except:
        st.warning("⚠️ Could not compute technical indicators.")
    df.dropna(inplace=True)
    return df

df = compute_indicators(df)

def compute_signal(df):
    if df.empty or len(df)<2: return "No data"
    latest, prev = df.iloc[-1], df.iloc[-2]
    macd_up = prev.get('MACD',0)<prev.get('MACD_SIGNAL',0) and latest.get('MACD',0)>latest.get('MACD_SIGNAL',0)
    macd_down = prev.get('MACD',0)>prev.get('MACD_SIGNAL',0) and latest.get('MACD',0)<latest.get('MACD_SIGNAL',0)
    price_above = latest['Close']>latest.get('SMA_50', latest['Close'])
    price_below = latest['Close']<latest.get('SMA_50', latest['Close'])
    rsi = latest.get('RSI_14',50)
    if macd_up and rsi<70 and price_above: return "BUY"
    if macd_down and rsi>30 and price_below: return "SELL"
    return "HOLD"

signal = compute_signal(df)

# -----------------------------
# CHART
# -----------------------------
def build_chart(df, chart_type="candlestick"):
    if df.empty: return go.Figure()
    fig = go.Figure()
    if chart_type=="candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    elif chart_type=="line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines"))
    elif chart_type=="bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Close']))
    if 'SMA_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode="lines", name="SMA 20", line=dict(color="orange")))
    if 'SMA_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode="lines", name="SMA 50", line=dict(color="purple")))
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    return fig

# -----------------------------
# AI ANALYSIS (OpenAI >=1.0)
# -----------------------------
def ai_stock_analysis(ticker, company, fundamentals, df):
    if df.empty: return "No data for AI analysis."
    last_close = df['Close'].iloc[-1]
    latest_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 'N/A'
    latest_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 'N/A'
    latest_signal = df['MACD_SIGNAL'].iloc[-1] if 'MACD_SIGNAL' in df.columns else 'N/A'

    prompt = f"""
Analyze {ticker} ({company.get('name')}) stock.
Last close: {last_close}
RSI: {latest_rsi}, MACD/Signal: {latest_macd}/{latest_signal}
Fundamentals: {fundamentals}
Sector: {company.get('sector')}, Industry: {company.get('industry')}

1. Predict short-term trend
2. Suggest Buy/Hold/Sell with reasoning
3. Highlight key indicators
4. Provide 3 follow-up research questions
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {e}"

# -----------------------------
# DISPLAY DASHBOARD
# -----------------------------
col1, col2 = st.columns([4,1])
with col1:
    st.subheader(f"{company.get('name', ticker_input)} ({ticker_input})")
    subinfo = [x for x in [company.get('sector'), company.get('industry')] if x]
    if subinfo: st.caption(" · ".join(subinfo))
with col2:
    if company.get('logo'): st.image(company['logo'], width=80)

st.markdown("---")
left, right = st.columns([3,1])
with left:
    if df.empty: st.error("No price data available")
    else: st.plotly_chart(build_chart(df, chart_type), use_container_width=True)
    if show_indicators:
        if 'RSI_14' in df.columns:
            st.line_chart(df['RSI_14'], height=200)
        if 'MACD' in df.columns:
            macd_fig = go.Figure(); macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'])); macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'])); st.plotly_chart(macd_fig)

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

if use_ai:
    st.markdown("---")
    st.header("🤖 AI Stock Analysis")
    st.write(ai_stock_analysis(ticker_input, company, fundamentals, df))

st.markdown("---")
st.caption("Data via Yahoo Finance & Finnhub, logos via Clearbit, AI included. Not financial advice.")
