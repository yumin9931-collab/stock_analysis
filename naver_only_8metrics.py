from __future__ import annotations

from datetime import timedelta
import glob
import json
import re

import numpy as np
import pandas as pd
import requests

CODE = "035420"  # NAVER
YEARS = 2.5
FIN_PARQUET_GLOB = "/home/s202423331/fin_parquet/*.parquet"

OP_KEYS = ("영업이익",)
DEBT_KEYS = ("부채비율",)
PBR_KEYS = ("PBR(배)", "PBR", "pbr")

# 2026년 11월 기준 참고 수치 반영 (영업이익: 십억 원 단위, 부채비율: %, PBR: 배)
FUNDAMENTALS_OVERRIDE: dict[str, dict[str, float]] = {
    "035420": {
        "5_latest_op": 451.3,  # 최근 실적(영업이익, 2026.3Q 약 4,513억 원 기준 십억 원 환산)
        "6_est_op": 489.6,  # 추정치(영업이익 (E), 2026.3Q 컨센서스 약 4,896억 원 기준 십억 원 환산)
        "7_debt_ratio": 40.0,  # 부채비율 (%)
        "8_pbr": 1.0,  # PBR (배, 2026E 기준)
    }
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}


def _num(txt: object) -> float:
  if txt is None:
    return np.nan
  s = str(txt).strip().replace(",", "").replace("%", "")
  if s in ("", "-", "N/A", "nan", "None"):
    return np.nan
  neg = s.startswith("(") and s.endswith(")")
  if neg:
    s = s[1:-1]
  try:
    v = float(s)
    return -v if neg else v
  except ValueError:
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def fetch_daily_from_sise_json(
    sess: requests.Session,
    code: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
  url = "https://api.finance.naver.com/siseJson.naver"
  params = {
      "symbol": code,
      "requestType": "1",
      "startTime": start_dt.strftime("%Y%m%d"),
      "endTime": end_dt.strftime("%Y%m%d"),
      "timeframe": "day",
  }
  resp = sess.get(url, params=params, headers=HEADERS, timeout=15)
  resp.raise_for_status()

  raw_text = resp.text.strip().replace("'", '"')
  raw_text = re.sub(r",\s*(\]|\})", r"\1", raw_text)
  data = json.loads(raw_text)
  if not data or len(data) < 2:
    return pd.DataFrame(columns=["date", "close", "volume"])

  cols = [str(c).strip() for c in data[0]]
  df = pd.DataFrame(data[1:], columns=cols)

  date_col = next(
      (c for c in df.columns if "날짜" in c or "date" in c.lower()), None
  )
  close_col = next(
      (c for c in df.columns if "종가" in c or "close" in c.lower()), None
  )
  vol_col = next(
      (c for c in df.columns if "거래량" in c or "volume" in c.lower()), None
  )
  if not date_col or not close_col:
    return pd.DataFrame(columns=["date", "close", "volume"])

  out = pd.DataFrame({
      "date": pd.to_datetime(
          df[date_col].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
      ),
      "close": df[close_col].map(_num),
      "volume": df[vol_col].map(_num) if vol_col else np.nan,
  })
  out = (
      out.dropna(subset=["date", "close"])
      .sort_values("date")
      .drop_duplicates("date")
  )
  return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(
      drop=True
  )


def load_fin_metrics_from_parquet(code: str) -> dict[str, float]:
  base = FUNDAMENTALS_OVERRIDE.get(
      code.zfill(6),
      {
          "5_latest_op": np.nan,
          "6_est_op": np.nan,
          "7_debt_ratio": np.nan,
          "8_pbr": np.nan,
      },
  )
  out = dict(base)

  files = sorted(glob.glob(FIN_PARQUET_GLOB))
  if not files:
    return out

  fin = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
  if fin.empty or "code" not in fin.columns:
    return out

  sub = fin[fin["code"].astype(str).str.zfill(6) == code.zfill(6)].copy()
  if sub.empty:
    return out

  sub["val_num"] = sub["value"].map(_num)
  sub["dt"] = (
      pd.to_datetime(sub["date"], errors="coerce")
      if "date" in sub.columns
      else pd.NaT
  )

  metric_str = sub["metric"].astype(str).str.strip()
  est_mask = sub["is_est"].astype(str).str.lower().isin(
      {"true", "1", "yes", "y", "e"}
  ) | (sub["is_est"] == True)

  def latest_val(mask: pd.Series) -> float:
    cand = sub[mask & sub["val_num"].notna()].copy()
    if cand.empty:
      return np.nan
    cand = cand.sort_values(["dt", "period"], ascending=[True, True])
    return float(cand["val_num"].iloc[-1])

  op_mask = metric_str.apply(
      lambda m: any(k in m for k in OP_KEYS) and "영업이익률" not in m
  )
  debt_mask = metric_str.apply(lambda m: any(k in m for k in DEBT_KEYS))
  pbr_mask = metric_str.apply(lambda m: any(k in m for k in PBR_KEYS))

  for key, val in [
      ("5_latest_op", latest_val(op_mask & (~est_mask))),
      ("6_est_op", latest_val(op_mask & est_mask)),
      ("7_debt_ratio", latest_val(debt_mask)),
      ("8_pbr", latest_val(pbr_mask)),
  ]:
    if not np.isnan(val):
      out[key] = val
  return out


def get_naver_8metrics(
    code: str = CODE, years: float = YEARS
) -> pd.DataFrame:
  end_dt = pd.Timestamp.today().normalize()
  start_dt = end_dt - timedelta(days=int(round(years * 365.25)))
  sess = requests.Session()

  df = fetch_daily_from_sise_json(sess, code, start_dt, end_dt)
  if df.empty or len(df) < 2:
    raise RuntimeError(
        f"{code} 일별 시세 데이터를 가져오지 못했습니다. (rows={len(df)})"
    )

  df["ret"] = df["close"].pct_change()

  cum_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
  ann_vol = df["ret"].std() * np.sqrt(252)
  avg_vol = df["volume"].mean()
  vol_ma20 = df["volume"].rolling(20, min_periods=5).mean().shift(1)
  surge_days = int((df["volume"] >= 2.0 * vol_ma20).sum())

  fin_metrics = load_fin_metrics_from_parquet(code)

  metrics_8 = {
      "1_cum_ret_2_5y": cum_ret,
      "2_ann_vol": ann_vol,
      "3_avg_volume": avg_vol,
      "4_vol_surge_days_2x": surge_days,
      **fin_metrics,
  }
  return pd.DataFrame([metrics_8])


if __name__ == "__main__":
  res = get_naver_8metrics()
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 200)
  print(res.to_string(index=False))
