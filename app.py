import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import date, timedelta

st.set_page_config(
    page_title="Pro Stock Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# ⚙️ Sidebar – User Controls
# ==============================
st.sidebar.header("📈 Stock Selector")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA, NVDA):", "AAPL").upper()
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
chart_type = st.sidebar.radio("Chart Type", ["candlestick", "line", "bar"], index=0)
show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_volume = st.sidebar.checkbox("Show Volume", True)
show_indicators = st.sidebar.checkbox("Show RSI / MACD Panels", True)

st.sidebar.markdown("---")
st.sidebar.write("💡 *Built with Yahoo Finance, TA-Lib, and AI analytics (coming soon).*")

# ==============================
# 📦 Fetch Data
# ==============================
@st.cache_data(show_spinner=False)
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        st.error("No data found for this ticker.")
    return df

@st.cache_data(show_spinner=False)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return info

df = get_data(ticker, period, interval)
info = get_info(ticker)

# ==============================
# 📊 Compute Indicators
# ==============================
def compute_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # Fix potential MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    try:
        df["SMA_20"] = SMAIndicator(close, 20).sma_indicator()
        df["SMA_50"] = SMAIndicator(close, 50).sma_indicator()
        df["EMA_20"] = EMAIndicator(close, 20).ema_indicator()
        macd = MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_HIST"] = macd.macd_diff()
        df["RSI_14"] = RSIIndicator(close).rsi()
    except Exception as e:
        st.warning(f"Indicator computation failed: {e}")

    return df.dropna()

df = compute_indicators(df)

# ==============================
# 🎨 Main Chart with Volume + Buy/Sell
# ==============================
def build_chart(df):
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Main Price Chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))

    # Moving Averages
    if show_ma:
        if "SMA_20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines",
                                     name="SMA 20", line=dict(width=1.2, color="orange")))
        if "SMA_50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines",
                                     name="SMA 50", line=dict(width=1.2, color="purple")))

    # Buy/Sell markers
    buy, sell = [], []
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        macd_up = prev["MACD"] < prev["MACD_SIGNAL"] and curr["MACD"] > curr["MACD_SIGNAL"]
        macd_down = prev["MACD"] > prev["MACD_SIGNAL"] and curr["MACD"] < curr["MACD_SIGNAL"]
        if macd_up and curr["RSI_14"] < 70:
            buy.append((df.index[i], curr["Low"] * 0.98))
        elif macd_down and curr["RSI_14"] > 30:
            sell.append((df.index[i], curr["High"] * 1.02))

    fig.add_trace(go.Scatter(x=[x[0] for x in buy], y=[x[1] for x in buy],
                             mode="markers", name="BUY", marker=dict(symbol="triangle-up", color="lime", size=12)))
    fig.add_trace(go.Scatter(x=[x[0] for x in sell], y=[x[1] for x in sell],
                             mode="markers", name="SELL", marker=dict(symbol="triangle-down", color="red", size=12)))

    # Volume Bars
    if show_volume and "Volume" in df:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color="rgba(100,149,237,0.4)", yaxis="y2"))

        fig.update_layout(
            yaxis=dict(domain=[0.25, 1], title="Price"),
            yaxis2=dict(domain=[0, 0.2], title="Volume", showgrid=False)
        )
    else:
        fig.update_layout(yaxis=dict(title="Price"))

    fig.update_layout(
        template="plotly_dark", height=700, hovermode="x unified",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    return fig

# ==============================
# 📉 RSI & MACD Panels
# ==============================
def indicator_panels(df):
    if df.empty or not show_indicators:
        return None

    fig = go.Figure()

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI (14)",
                             line=dict(color="#42a5f5", width=2)))
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="lime", opacity=0.2, line_width=0)

    # MACD
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD Histogram",
                         marker_color="rgba(255,255,255,0.4)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="purple", width=1.5)))

    fig.update_layout(
        template="plotly_dark", height=400,
        hovermode="x unified",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    return fig

# ==============================
# 🏢 Display Info + Charts
# ==============================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(f"## {info.get('longName', ticker)} ({ticker})")
    if "logo_url" in info:
        st.image(info["logo_url"], width=120)
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
    st.markdown(f"**Country:** {info.get('country', 'N/A')}")
    st.markdown("---")
    st.info("💬 AI Insight (coming soon): Personalized stock prediction & market news")

with col2:
    st.plotly_chart(build_chart(df), use_container_width=True)
    if show_indicators:
        st.plotly_chart(indicator_panels(df), use_container_width=True)

st.markdown("---")
st.caption("⚙️ Data source: Yahoo Finance | Technicals: ta | Built with ❤️ using Streamlit & Plotly")
