# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 16:08:47 2025

@author: manju
"""

# save this as app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import requests

@st.cache_data
def download_nifty500_list():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

st.title("📈 Automated Swing Trading Scanner")

mode = st.radio("Select mode:", ["Live", "Backdated"])

if mode == "Backdated":
    analysis_date = st.date_input("Enter analysis date")

if st.button("Run Scanner"):
    tickers = download_nifty500_list()
    st.write(f"✅ Retrieved {len(tickers)} stock tickers")

    progress = st.progress(0)
    results = []

    for i, ticker in enumerate(tickers, start=1):
        try:
            data = yf.download(ticker + ".NS", period="5d", interval="1d", progress=False)
            if not data.empty:
                close_price = data['Close'].iloc[-1]
                results.append({"Ticker": ticker, "Close": close_price})
        except Exception as e:
            pass
        progress.progress(i / len(tickers))

    st.success("✅ Analysis Completed!")
    if results:
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("⚠️ No results generated.")
def send_to_telegram(csv_file, bot_token, chat_id):
    df = pd.read_csv(csv_file)
    top5 = df.head(5).to_string(index=False)

    message = f"📊 Daily Swing Trade Top 5 Picks:\n\n{top5}"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"@Manjunath1113": chat_id, "text": message}
    requests.post(url, data=payload)