import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_TO = os.getenv("MAIL_TO", "")

STATE_PATH = Path("state/last_signals.json")
HK_TZ = ZoneInfo("Asia/Hong_Kong")

# 第一版先用高流动性大盘股池，后续可换成S&P500全量
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD",
    "NFLX", "JPM", "GS", "V", "MA", "COST", "WMT", "LLY", "UNH", "XOM",
    "CVX", "BAC", "PLTR", "INTC", "QCOM", "ADBE", "CRM", "ORCL", "MU", "UBER", "DIS"
]


def require_env():
    required = {
        "ALPACA_KEY": ALPACA_KEY,
        "ALPACA_SECRET": ALPACA_SECRET,
        "TG_BOT_TOKEN": TG_BOT_TOKEN,
        "TG_CHAT_ID": TG_CHAT_ID,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


def fetch_hourly_bars(symbols: List[str], limit: int = 120) -> Dict[str, pd.DataFrame]:
    url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Hour",
        "limit": limit,
        "adjustment": "raw",
        "feed": "iex",   # 免费层可用
        "sort": "asc",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    out: Dict[str, pd.DataFrame] = {}
    bars_map = data.get("bars", {})
    for sym, bars in bars_map.items():
        if not bars:
            continue
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={
                "t": "time",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        out[sym] = df

    return out


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_s = pd.Series(up, index=series.index).rolling(period).mean()
    down_s = pd.Series(down, index=series.index).rolling(period).mean()
    rs = up_s / (down_s + 1e-9)
    return 100 - (100 / (1 + rs))


def evaluate_symbol(df: pd.DataFrame) -> Tuple[str, float, str]:
    """
    返回: (signal, score, reason)
    signal: BUY / SELL / HOLD
    """
    if len(df) < 50:
        return "HOLD", 0.0, "数据不足"

    c = df["close"]
    h = df["high"]
    v = df["volume"]

    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    r = rsi(c, 14)
    vol_ma20 = v.rolling(20).mean()

    last = len(df) - 1
    prev_20_high = h.iloc[max(0, last - 20):last].max()

    close_now = float(c.iloc[last])
    ema8_now = float(ema8.iloc[last])
    ema21_now = float(ema21.iloc[last])
    rsi_now = float(r.iloc[last]) if not np.isnan(r.iloc[last]) else 50.0
    vol_ratio = float(v.iloc[last] / (vol_ma20.iloc[last] + 1e-9))

    uptrend = close_now > ema21_now and ema8_now > ema21_now
    breakout = close_now > prev_20_high
    vol_ok = vol_ratio >= 1.1

    buy_cond = uptrend and breakout and vol_ok and (50 <= rsi_now <= 75)
    sell_cond = (close_now < ema8_now) or ((ema8_now < ema21_now) and rsi_now < 45)

    if buy_cond:
        score = (close_now / (ema21_now + 1e-9) - 1) * 100 + max(0.0, (rsi_now - 50) * 0.2) + max(0.0, (vol_ratio - 1) * 5)
        reason = f"突破+趋势, RSI={rsi_now:.1f}, 量比={vol_ratio:.2f}"
        return "BUY", float(score), reason

    if sell_cond:
        score = (ema8_now / (ema21_now + 1e-9) - 1) * 100 - max(0.0, (45 - rsi_now) * 0.2)
        reason = f"转弱/失守EMA8, RSI={rsi_now:.1f}"
        return "SELL", float(score), reason

    return "HOLD", 0.0, f"观望, RSI={rsi_now:.1f}, 量比={vol_ratio:.2f}"


def load_prev_state() -> Dict:
    if not STATE_PATH.exists():
        return {"buy": [], "sell": [], "ts": ""}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"buy": [], "sell": [], "ts": ""}


def save_state(buy: List[str], sell: List[str]):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now_hk = datetime.now(HK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    payload = {"buy": buy, "sell": sell, "ts": now_hk}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(results: Dict[str, Tuple[str, float, str]]) -> str:
    now_hk = datetime.now(HK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    prev = load_prev_state()

    buys = [(s, sc, rs) for s, (sig, sc, rs) in results.items() if sig == "BUY"]
    sells = [(s, sc, rs) for s, (sig, sc, rs) in results.items() if sig == "SELL"]

    buys.sort(key=lambda x: x[1], reverse=True)
    sells.sort(key=lambda x: x[1])  # 越负越弱

    top_buys = buys[:8]
    top_sells = sells[:8]

    curr_buy_syms = [x[0] for x in top_buys]
    curr_sell_syms = [x[0] for x in top_sells]

    prev_buy = set(prev.get("buy", []))
    prev_sell = set(prev.get("sell", []))
    new_buy = [s for s in curr_buy_syms if s not in prev_buy]
    new_sell = [s for s in curr_sell_syms if s not in prev_sell]

    lines = []
    lines.append(f"[US Stock Hourly Signals] {now_hk}")
    lines.append("仅供研究记录，非投资建议。")
    lines.append("")
    lines.append(f"上次快照时间: {prev.get('ts', 'N/A')}")
    lines.append(f"本次 BUY 候选: {', '.join(curr_buy_syms) if curr_buy_syms else '无'}")
    lines.append(f"本次 SELL/减仓 候选: {', '.join(curr_sell_syms) if curr_sell_syms else '无'}")
    lines.append("")
    lines.append(f"新增 BUY: {', '.join(new_buy) if new_buy else '无'}")
    lines.append(f"新增 SELL: {', '.join(new_sell) if new_sell else '无'}")
    lines.append("")
    lines.append("BUY 详情:")
    if top_buys:
        for s, sc, rs in top_buys:
            lines.append(f"- {s}: score={sc:.2f} | {rs}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("SELL 详情:")
    if top_sells:
        for s, sc, rs in top_sells:
            lines.append(f"- {s}: score={sc:.2f} | {rs}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("风险参数(手动执行时遵守): 账户最大回撤15%, 单票<=50%, 总仓位<=100%")

    save_state(curr_buy_syms, curr_sell_syms)
    return "\n".join(lines)


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()


def send_email(text: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_TO):
        print("Skip email: SMTP env not complete")
        return

    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = "US Stock Hourly Signals"
    msg["From"] = SMTP_USER
    msg["To"] = MAIL_TO

    if SMTP_PORT == 465:
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20)
    else:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        server.starttls()

    with server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [MAIL_TO], msg.as_string())


def main():
    require_env()
    bars = fetch_hourly_bars(SYMBOLS, limit=120)

    results = {}
    for sym in SYMBOLS:
        df = bars.get(sym)
        if df is None or df.empty:
            continue
        results[sym] = evaluate_symbol(df)

    report = build_report(results)
    print(report)

    send_telegram(report)
    send_email(report)


if __name__ == "__main__":
    main()
