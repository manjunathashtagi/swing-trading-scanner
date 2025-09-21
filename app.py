import os
import sys
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from io import StringIO
import time

NSE_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
OUTPUT_FILE = "swing_trading_results.csv"''
# --------- Configuration ----------

DAILY_BUDGET = 5000.0        # ₹ per day for allocation
TOP_N = 5
HIST_DAYS = 180              # historical window for indicators
YF_PERIOD = "6mo"            # fallback period (live)
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN = 0.12         # polite delay between ticker downloads

# Telegram secrets (set in GitHub or env)
TELEGRAM_TOKEN = os.getenv("8394166743:AAEASt_4WYwlNB6LP3j6_4VM-qPkPTt4Erg")
TELEGRAM_CHAT_ID = os.getenv("1384256039")


# ---------------- Utilities ----------------
def download_nifty500_list():
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(NSE_CSV_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    symbols = df["Symbol"].dropna().astype(str).map(str.strip).tolist()
    cleaned = [s for s in symbols if " " not in s and "DUMMY" not in s]
    return list(dict.fromkeys(cleaned))

def sma(series, window): return series.rolling(window, min_periods=1).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/window, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = series.ewm(span=fast).mean(), series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_score(df):
    if df is None or df.empty:
        return None

    # ensure we have a valid close column
    if 'Adj Close' in df.columns:
        close = df['Adj Close'].dropna()
    elif 'Close' in df.columns:
        close = df['Close'].dropna()
    else:
        return None

    if close.empty or len(close) < 60:
        return None

    s20 = sma(close, 20)
    s50 = sma(close, 50)
    rsi14 = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)

    last = float(close.iloc[-1])
    prev_idx = -2 if len(close) >= 2 else -1

    score = 0.0
    reasons = []

    # --- SMA relationship ---
    if len(s20) > 0 and len(s50) > 0:
        if float(s20.iloc[-1]) > float(s50.iloc[-1]):
            score += 1.0
            reasons.append("SMA20 > SMA50")
        else:
            score -= 0.2

        if len(s20) >= 2 and len(s50) >= 2:
            if float(s20.iloc[prev_idx]) <= float(s50.iloc[prev_idx]) and float(s20.iloc[-1]) > float(s50.iloc[-1]):
                score += 1.0
                reasons.append("Recent SMA cross up")

    # --- RSI zones ---
    last_rsi = float(rsi14.iloc[-1]) if not rsi14.empty else 50.0
    if last_rsi < 30:
        score += 0.9; reasons.append(f"RSI {last_rsi:.1f} <30")
    elif 30 <= last_rsi <= 55:
        score += 0.6
    elif 55 < last_rsi <= 70:
        score += 0.2
    else:
        score -= 0.6; reasons.append(f"RSI {last_rsi:.1f} high")

    # --- MACD hist ---
    if not macd_hist.empty:
        if len(macd_hist) >= 2:
            if float(macd_hist.iloc[prev_idx]) <= 0 and float(macd_hist.iloc[-1]) > 0:
                score += 0.9; reasons.append("MACD hist turned +")
        if float(macd_hist.iloc[-1]) > 0:
            score += 0.3

    return {
        "score": float(score),
        "last_close": last,
        "rsi": last_rsi,
        "sma20": float(s20.iloc[-1]) if not s20.empty else np.nan,
        "sma50": float(s50.iloc[-1]) if not s50.empty else np.nan,
        "macd_hist": float(macd_hist.iloc[-1]) if not macd_hist.empty else np.nan,
        "reasons": "; ".join(reasons)
    }


def fetch_history(symbol, end_date=None):
    tkr = symbol + ".NS"
    try:
        if end_date:
            end_dt = datetime.strptime(end_date,"%Y-%m-%d").date()
            start_dt = end_dt - timedelta(days=HIST_DAYS)
            df = yf.download(tkr, start=start_dt, end=end_dt+timedelta(1), progress=False)
        else:
            df = yf.download(tkr, period=YF_PERIOD, progress=False)
        return df
    except: return None

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# ---------------- Main ----------------
def main():
    print("📈 Automated Swing Trading Scanner")

    # --- Mode selection ---
    mode_env = os.getenv("MODE")
    backdate_env = os.getenv("BACKDATE")

    if mode_env:  # Auto mode (GitHub Actions / cron)
        mode = "backdated" if mode_env.strip() == "2" else "live"
        backdate = backdate_env
        if mode == "backdated" and not backdate:
            print("⚠ BACKDATE must be set when MODE=2")
            sys.exit(1)
        print(f"🔧 Running in AUTO mode: {mode.upper()} {('('+backdate+')') if backdate else ''}")
    else:  # Interactive mode (local runs)
        while True:
            mode_in = input("Select mode (1 = Live, 2 = Backdated): ").strip()
            if mode_in in ("1", "2"):
                break
            print("Please enter 1 or 2.")
        mode = "backdated" if mode_in == "2" else "live"
        backdate = None
        if mode == "backdated":
            while True:
                backdate = input("Enter backdate (YYYY-MM-DD): ").strip()
                try:
                    datetime.strptime(backdate, "%Y-%m-%d")
                    break
                except ValueError:
                    print("Invalid format, use YYYY-MM-DD.")

    # --- Download stock list ---
    symbols = download_nifty500_list()
    results, failed = [], []
    pbar = tqdm(total=len(symbols), desc="Scanning", unit="stk")
    for sym in symbols:
        df = fetch_history(sym, end_date=backdate if mode=="backdated" else None)
        if df is None or df.empty:
            failed.append(sym)
        else:
            score = compute_score(df)
            if score:
                results.append({"symbol": sym, **score})
        pbar.update(1)
        time.sleep(SLEEP_BETWEEN)
    pbar.close()
    if not results:
        print("⚠ No results"); return

    df_all = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    # Top N selection + allocation
    top = df_all.head(TOP_N).copy()
    cash_per = DAILY_BUDGET / TOP_N
    top["qty"] = (cash_per / top["last_close"]).astype(int)
    top["buy_price"] = top["last_close"]
    top["target_price"] = (top["buy_price"] * 1.03).round(2)
    top["stop_loss"] = (top["buy_price"] * 0.98).round(2)
    
    # Save to one CSV with 2 sheets
    tag = datetime.now().strftime("%Y%m%d") if mode=="live" else f"backtest_{backdate}"
    fname = f"nifty_scan_{tag}.xlsx"
    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_scored", index=False)
        top.to_excel(writer, sheet_name=f"top_{TOP_N}", index=False)
    
    print(f"\n✅ Report saved: {fname}")
    print("\nTop picks:")
    print(top[["symbol","score","last_close","qty","buy_price","target_price","stop_loss","reasons"]].to_string(index=False))
    
    # --- Telegram summary ---
    msg = f"📊 Top {TOP_N} picks ({mode.upper()}{' ' + backdate if mode=='backdated' else ''}):\n\n"
    for r in top.itertuples():
        msg += f"{r.symbol} | Buy ₹{r.buy_price:.2f} | Target ₹{r.target_price:.2f} | SL ₹{r.stop_loss:.2f}\n"
    
    send_telegram(msg)

if __name__=="__main__":
    main()
