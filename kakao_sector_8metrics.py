from __future__ import annotations

from datetime import timedelta
import glob
import json
import re
import time

import numpy as np
import pandas as pd
import requests

TARGET_CODE = "035720"  # 카카오
YEARS = 2.5
FIN_PARQUET_GLOB = "/home/s202423331/fin_parquet/*.parquet"
OUT_CSV = "/home/s202423331/kakao_sector_8metrics.csv"

# 카카오가 속한 플랫폼·콘텐츠 업종 비교군 (필요 시 종목 추가/수정 가능)
SECTOR_PEERS: dict[str, str] = {
    "035720": "카카오",
    "035420": "NAVER",
    "259960": "크래프톤",
    "036570": "엔씨소프트",
    "263750": "펄어비스",
    "078340": "컴투스",
    "181710": "NHN",
    "067160": "SOOP",
}

OP_KEYS = ("영업이익",)
DEBT_KEYS = ("부채비율",)
PBR_KEYS = ("PBR(배)", "PBR", "pbr")

# 2026년 11월 기준 재무·컨센서스 참고치 (영업이익: 십억 원, 부채비율: %, PBR: 배)
FUNDAMENTALS_OVERRIDE: dict[str, dict[str, float]] = {
    "035720": {  # 카카오 (2026.3Q 실적발표 및 증권사 리포트 기준)
        "5_latest_op": 240.3,   # 2026.3Q 영업이익 2,403억 원
        "6_est_op": 248.3,      # 2026.3Q 영업이익 컨센서스 2,483억 원
        "7_debt_ratio": 84.8,   # 연결 부채비율 (%)
        "8_pbr": 1.19,          # PBR (배)
    },
    "035420": {  # NAVER (2026.3Q 실적발표 및 증권사 리포트 기준)
        "5_latest_op": 451.3,   # 2026.3Q 영업이익 4,513억 원
        "6_est_op": 489.6,      # 2026.3Q 영업이익 컨센서스 4,896억 원
        "7_debt_ratio": 40.0,   # 부채비율 (%)
        "8_pbr": 1.00,          # PBR (배)
    },
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


def fetch_daily_sise_json(
    sess: requests.Session,
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    url = "https://api.finance.naver.com/siseJson.naver"
    params = {
        "symbol": symbol,
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

    date_col = next((c for c in df.columns if "날짜" in c or "date" in c.lower()), None)
    close_col = next((c for c in df.columns if "종가" in c or "close" in c.lower()), None)
    vol_col = next((c for c in df.columns if "거래량" in c or "volume" in c.lower()), None)
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
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)


def load_fin_parquet_all() -> pd.DataFrame:
    files = sorted(glob.glob(FIN_PARQUET_GLOB))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_fundamentals_for_code(fin_df: pd.DataFrame, code: str) -> dict[str, float]:
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
    if fin_df.empty or "code" not in fin_df.columns:
        return out

    sub = fin_df[fin_df["code"].astype(str).str.zfill(6) == code.zfill(6)].copy()
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


def analyze_kakao_sector_8metrics(
    target_code: str = TARGET_CODE, years: float = YEARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_dt = pd.Timestamp.today().normalize()
    start_dt = end_dt - timedelta(days=int(round(years * 365.25)))
    sess = requests.Session()
    fin_df = load_fin_parquet_all()

    # 1) 업종 내 전체 종목 일별 시세 수집
    price_frames: list[pd.DataFrame] = []
    for code, name in SECTOR_PEERS.items():
        df = fetch_daily_sise_json(sess, code, start_dt, end_dt)
        if df.empty or len(df) < 20:
            print(f"[경고] {code} {name} 시세 부족 (rows={len(df)})")
            continue
        df["code"] = code
        df["name"] = name
        df["ret"] = df["close"].pct_change()
        df["norm_price"] = df["close"] / df["close"].iloc[0] * 100.0
        price_frames.append(df)
        time.sleep(0.1)

    if not price_frames:
        raise RuntimeError("업종 시세 데이터를 수집하지 못했습니다.")

    panel = pd.concat(price_frames, ignore_index=True)

    # 2) KOSPI 일간 수익률 (시장 동조 계산용)
    kospi_df = fetch_daily_sise_json(sess, "KOSPI", start_dt, end_dt)
    if not kospi_df.empty:
        kospi_df["kospi_ret"] = kospi_df["close"].pct_change()
        kospi_ret_s = kospi_df.set_index("date")["kospi_ret"]
    else:
        kospi_ret_s = pd.Series(dtype=float)

    # 3) 업종 내 일별 주가 지수 순위 (시작일 대비 상승률 기준 일별 순위)
    panel["daily_sector_rank"] = panel.groupby("date")["norm_price"].rank(
        ascending=False, method="min"
    )

    # 4) 수익률 피벗 (동조 상관계수 계산용)
    ret_pivot = panel.pivot(index="date", columns="code", values="ret")

    rows: list[dict[str, object]] = []
    for code, grp in panel.groupby("code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        name = grp["name"].iloc[0]

        # [순위] 최근 2.5년 중반 vs 종료 시점 업종 내 성과 순위 및 평균 순위
        mid_idx = len(grp) // 2
        mid_rank = float(grp["daily_sector_rank"].iloc[mid_idx])
        end_rank = float(grp["daily_sector_rank"].iloc[-1])
        avg_rank = float(grp["daily_sector_rank"].mean())

        # [수익률] 2.5년 누적수익률 및 연환산 변동성
        cum_ret = float(grp["close"].iloc[-1] / grp["close"].iloc[0] - 1.0)
        ann_vol = float(grp["ret"].std() * np.sqrt(252))

        # [동조] 업종(본인 제외 동일가중) 상관계수 & KOSPI 상관계수
        peer_cols = [c for c in ret_pivot.columns if c != code]
        if peer_cols and code in ret_pivot.columns:
            peer_mean_ret = ret_pivot[peer_cols].mean(axis=1)
            corr_sector = float(ret_pivot[code].corr(peer_mean_ret))
        else:
            corr_sector = np.nan

        if not kospi_ret_s.empty and code in ret_pivot.columns:
            corr_kospi = float(ret_pivot[code].corr(kospi_ret_s))
        else:
            corr_kospi = np.nan

        # [거래량] 일평균 거래량 & 20일 이동평균 대비 2배 이상 급증일수
        avg_vol = float(grp["volume"].mean())
        vol_ma20 = grp["volume"].rolling(20, min_periods=5).mean().shift(1)
        surge_days = int((grp["volume"] >= 2.0 * vol_ma20).sum())

        # [실적 / 추정치 / 부채 / PBR]
        fund = get_fundamentals_for_code(fin_df, code)

        rows.append({
            "code": code,
            "name": name,
            "1_end_rank": int(end_rank),
            "1_avg_rank_2_5y": round(avg_rank, 2),
            "1_mid_rank": int(mid_rank),
            "2_cum_ret_2_5y": cum_ret,
            "2_ann_vol": ann_vol,
            "3_corr_sector": corr_sector,
            "3_corr_kospi": corr_kospi,
            "4_avg_volume": avg_vol,
            "4_vol_surge_days_2x": surge_days,
            **fund,
        })

    sector_df = (
        pd.DataFrame(rows)
        .sort_values(["1_end_rank", "2_cum_ret_2_5y"], ascending=[True, False])
        .reset_index(drop=True)
    )
    kakao_df = sector_df[sector_df["code"] == target_code].reset_index(drop=True)
    return kakao_df, sector_df


if __name__ == "__main__":
    kakao_only, sector_all = analyze_kakao_sector_8metrics()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("=== [카카오(035720) 2.5년 8대 지표 요약] ===")
    print(kakao_only.to_string(index=False))

    print("\n=== [업종 내 전체 종목 비교 (2.5년 누적수익률 순위순)] ===")
    print(sector_all.to_string(index=False))

    sector_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_CSV}")
