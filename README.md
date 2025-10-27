# Streamlit Stock Analyzer

A Streamlit app that fetches market data from Yahoo Finance (via `yfinance`), plots interactive charts (candlestick/line/bar) with indicators (SMA, EMA, MACD, RSI), shows fundamentals and a simple buy/sell/hold heuristic, and an optional OpenAI-powered assistant.

## Quick start

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd your-repo
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."

streamlit run app.py
