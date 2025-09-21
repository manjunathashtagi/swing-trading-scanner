# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 23:24:31 2025

@author: manju
"""

import streamlit as st
import pandas as pd

st.title("📊 Swing Trading Recommendations")

DATA_URL = "https://raw.githubusercontent.com/USERNAME/REPO/main/top_10_stocks.csv"

try:
    df = pd.read_csv(DATA_URL)
    st.success("Latest recommendations loaded ✅")
    st.dataframe(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
