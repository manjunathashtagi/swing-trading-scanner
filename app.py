# untitled4_fixed.py
# Automated NIFTY500 scanner — Live or Backdated mode, with progress bar and Excel output.

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from datetime import datetime, timedelta
import time
import sys
import re
from tqdm import tqdm          # pip install tqdm
import os

# --------- Configuration ----------
NSE_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
DAILY_BUDGET = 5000.0        # ₹ per day for allocation
TOP_N = 5
HIST_DAYS = 180              # historical window for indicators
YF_PERIOD = "6mo"            # fallback period (live)
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN = 0.12         # polite delay between ticker downloads


def get_mode():
    """
    Get mode from environment variable MODE if available,
    otherwise ask user interactively.
    """
    mode_in = os.getenv("MODE")
    if mode_in:
        print(f"MODE detected from environment: {mode_in}")
        return mode_in.strip()
    else:
        return input("Select mode (1 = Live, 2 = Backdated): ").strip()

def main():
    print("Welcome to the Automated Swing Trading Scanner! 📈")
    
    mode_in = get_mode()   # <-- updated line

    if mode_in == "1":
        print("Running in Live mode...")
        # your existing live mode code here
    elif mode_in == "2":
        print("Running in Backdated mode...")
        # your existing backdated mode code here
    else:
        print("Invalid mode selected. Exiting...")
        return

    # continue with rest of your script...


# --------- Utilities ----------
def download_nifty500_list(retries=3, timeout=REQUEST_TIMEOUT):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"📥 Downloading latest NIFTY 500 list... (Attempt {attempt}/{retries})")
            resp = requests.get(NSE_CSV_URL, headers=headers, timeout=timeout)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            if "Symbol" not in df.columns:
                raise ValueError("CSV parsed but 'Symbol' column not found.")
            # Clean symbols: remove blanks, header-like rows, DUMMY entries, keep alnum/hyphen/dot
            symbols = df["Symbol"].dropna().astype(str).map(str.strip).tolist()
            cleaned = []
            bad_tokens = {"COMPANY", "NAME", "INDUSTRY", "CODE", "ISIN", "SERIES"}
            for s in symbols:
                if not s:
                    continue
                su = s.upper()
                if any(tok in su for tok in bad_tokens):
                    continue
                if "DUMMY" in su:
                    continue
                # Exclude entries that contain spaces or slashes or are too long
                if " " in su or "/" in su or len(su) > 12:
                    continue
                # require at least one alnum
                if not re.search(r"[A-Z0-9]", su):
                    continue
                cleaned.append(su)
            cleaned = list(dict.fromkeys(cleaned))  # preserve order, unique
            print(f"✅ Retrieved {len(cleaned)} tickers (cleaned) from NIFTY 500 list.")
            return cleaned
        except Exception as e:
            print(f"⚠️ Error fetching NIFTY500 CSV (attempt {attempt}): {e}")
            if attempt < retries:
                wait = attempt * 4
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("❌ Failed to download NIFTY500 CSV after retries.")
                raise

# --------- Indicator helpers ----------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/window, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# Composite score (simple, extendable)
def compute_score(df):
    if df is None or df.empty:
        return None
    if 'Adj Close' not in df.columns and 'Close' not in df.columns:
        return None
    # ensure using Adj Close if available
    close = (df.get('Adj Close') if 'Adj Close' in df.columns else df['Close']).dropna()
    if len(close) < 60:
        return None

    s20 = sma(close, 20)
    s50 = sma(close, 50)
    rsi14 = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)

    last = close.iloc[-1]
    prev_idx = -2 if len(close) >= 2 else -1

    score = 0.0
    reasons = []

    # SMA relationship
    try:
        if s20.iloc[-1] > s50.iloc[-1]:
            score += 1.0
            reasons.append("SMA20 > SMA50")
        else:
            score -= 0.2
        # Recent SMA cross up
        if s20.iloc[prev_idx] <= s50.iloc[prev_idx] and s20.iloc[-1] > s50.iloc[-1]:
            score += 1.0
            reasons.append("Recent SMA cross up")
    except Exception:
        pass

    # RSI zones
    last_rsi = float(rsi14.iloc[-1]) if len(rsi14.dropna()) else 50.0
    if last_rsi < 30:
        score += 0.9; reasons.append(f"RSI {last_rsi:.1f} <30")
    elif 30 <= last_rsi <= 55:
        score += 0.6
    elif 55 < last_rsi <= 70:
        score += 0.2
    else:
        score -= 0.6; reasons.append(f"RSI {last_rsi:.1f} high")

    # MACD hist turn positive
    try:
        if macd_hist.iloc[prev_idx] <= 0 and macd_hist.iloc[-1] > 0:
            score += 0.9; reasons.append("MACD hist turned +")
        elif macd_hist.iloc[-1] > 0:
            score += 0.3
    except Exception:
        pass

    return {
        "score": float(score),
        "last_close": float(last),
        "rsi": float(last_rsi),
        "sma20": float(s20.iloc[-1]) if len(s20) else np.nan,
        "sma50": float(s50.iloc[-1]) if len(s50) else np.nan,
        "macd_hist": float(macd_hist.iloc[-1]) if len(macd_hist) else np.nan,
        "reasons": "; ".join(reasons)
    }

# --------- Fetch data safely ----------
def fetch_history(symbol, end_date=None, hist_days=HIST_DAYS):
    """
    symbol: ticker without .NS (e.g., 'INFY')
    end_date: 'YYYY-MM-DD' (string) for backdated runs. If None => live (use YF period)
    """
    tkr = symbol + ".NS"
    try:
        if end_date:
            # get history window ending on end_date (inclusive)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            start_dt = end_dt - timedelta(days=hist_days)
            # yfinance end is exclusive -> pass end_dt + 1 day
            yf_start = start_dt.strftime("%Y-%m-%d")
            yf_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(tkr, start=yf_start, end=yf_end, progress=False, auto_adjust=True, threads=False)
        else:
            df = yf.download(tkr, period=YF_PERIOD, interval="1d", progress=False, auto_adjust=True, threads=False)
        return df
    except Exception as e:
        # yfinance may raise on delisted symbols; return None and let caller continue
        return None

# --------- Main runner ----------
def main():
    print("Welcome to the Automated Swing Trading Scanner! 📈")
    # Mode selection
    while True:
        mode_in = input("Select mode (1 = Live, 2 = Backdated): ").strip()
        if mode_in in ("1", "2"):
            break
        print("Please enter 1 or 2.")
    mode = "live" if mode_in == "1" else "backdated"
    backdate = None
    if mode == "backdated":
        while True:
            backdate = input("Enter analysis date (YYYY-MM-DD): ").strip()
            try:
                # validate date
                datetime.strptime(backdate, "%Y-%m-%d")
                break
            except Exception:
                print("Invalid date format. Use YYYY-MM-DD.")
    print(f"\nMode = {mode.upper()}", ("", f", backdate = {backdate}")[mode=='backdated'])

    # Download list
    try:
        symbols = download_nifty500_list()
    except Exception as e:
        print("Fatal: could not retrieve NIFTY500 list. Exiting.")
        sys.exit(1)

    if not symbols:
        print("No symbols found. Exiting.")
        sys.exit(1)

    total = len(symbols)
    print(f"\nScanning {total} symbols. This may take ~3-6 minutes depending on network.\n")

    results = []
    failed_fetch = []
    pbar = tqdm(total=total, desc="Scanning", unit="stk", ncols=100)
    start_time = time.time()
    for sym in symbols:
        # fetch data
        df = fetch_history(sym, end_date=backdate if mode=="backdated" else None)
        if df is None or df.empty:
            failed_fetch.append(sym)
            pbar.update(1)
            time.sleep(SLEEP_BETWEEN)
            continue

        # compute score
        score = compute_score(df)
        if score:
            row = {"symbol": sym, **score}
            results.append(row)

        pbar.update(1)
        time.sleep(SLEEP_BETWEEN)

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\nDone. Time elapsed: {elapsed:.1f}s. Fetched data for {len(results)} symbols, {len(failed_fetch)} failed.")

    if not results:
        print("⚠️ No scored results — either filters are strict or data insufficient.")
        # still write failed list for debugging
        if failed_fetch:
            pd.Series(failed_fetch, name="failed_ticker").to_csv("failed_downloads.csv", index=False)
            print("Saved failed tickers to failed_downloads.csv")
        sys.exit(0)

    df_all = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    # Top N selection + allocation
    top = df_all.head(TOP_N).copy()
    cash_per = DAILY_BUDGET / TOP_N
    top["qty"] = (cash_per / top["last_close"]).astype(int)
    top["buy_price"] = top["last_close"]
    top["target_price"] = (top["buy_price"] * 1.03).round(2)
    top["stop_loss"] = (top["buy_price"] * 0.98).round(2)

    # Save Excel report
    tag = (datetime.now().strftime("%Y%m%d")) if mode=="live" else f"backtest_{backdate}"
    fname = f"nifty_scan_{tag}.xlsx"
    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_scored", index=False)
        top.to_excel(writer, sheet_name=f"top_{TOP_N}", index=False)

    print(f"\n✅ Report saved: {fname}")
    print("\nTop picks:")
    print(top[["symbol","score","last_close","qty","buy_price","target_price","stop_loss","reasons"]].to_string(index=False))

if __name__ == "__main__":
    main()
