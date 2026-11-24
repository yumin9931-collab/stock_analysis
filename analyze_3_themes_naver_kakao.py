from __future__ import annotations

from datetime import timedelta
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import requests

PARQUET_GLOB = "/home/s202423331/parquet/*.parquet"
OUT_DIR = "/home/s202423331/results_3themes"
YEARS = 2.5

# ~/parquet에 sector가 없거나 폴백 수집 시 사용할 기본 종목군
FALLBACK_UNIVERSE: dict[str, tuple[str, str]] = {
    "035420": ("NAVER", "인터넷/플랫폼"),
    "035720": ("카카오", "인터넷/플랫폼"),
    "259960": ("크래프톤", "게임/콘텐츠"),
    "036570": ("엔씨소프트", "게임/콘텐츠"),
    "263750": ("펄어비스", "게임/콘텐츠"),
    "078340": ("컴투스", "게임/콘텐츠"),
    "181710": ("NHN", "인터넷/플랫폼"),
    "067160": ("SOOP", "인터넷/플랫폼"),
    "005930": ("삼성전자", "반도체/IT"),
    "000660": ("SK하이닉스", "반도체/IT"),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}


def _num(x: object) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else np.nan


def fetch_sise_json(code: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    url = "https://api.finance.naver.com/siseJson.naver"
    params = {
        "symbol": code,
        "requestType": "1",
        "startTime": start_dt.strftime("%Y%m%d"),
        "endTime": end_dt.strftime("%Y%m%d"),
        "timeframe": "day",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    raw = re.sub(r",\s*(\]|\})", r"\1", resp.text.strip().replace("'", '"'))
    data = json.loads(raw)
    if not data or len(data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(data[1:], columns=[str(c).strip() for c in data[0]])
    c_date = next((c for c in df.columns if "날짜" in c or "date" in c.lower()), None)
    c_close = next((c for c in df.columns if "종가" in c or "close" in c.lower()), None)
    c_vol = next((c for c in df.columns if "거래량" in c or "volume" in c.lower()), None)
    if not c_date or not c_close:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(df[c_date].astype(str).str.strip(), format="%Y%m%d", errors="coerce"),
        "close": df[c_close].map(_num),
        "volume": df[c_vol].map(_num) if c_vol else np.nan,
    })
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def load_panel(years: float = YEARS) -> pd.DataFrame:
    end_dt = pd.Timestamp.today().normalize()
    start_dt = end_dt - timedelta(days=int(round(years * 365.25)))
    files = sorted(glob.glob(PARQUET_GLOB))

    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["close"] = df["close"].map(_num)
        df["volume"] = df["volume"].map(_num)
        if "mktcap_100mn" in df.columns:
            df["mktcap_100mn"] = df["mktcap_100mn"].map(_num)
        else:
            df["mktcap_100mn"] = np.nan
        if "sector" not in df.columns:
            df["sector"] = df["code"].map(lambda c: FALLBACK_UNIVERSE.get(c, ("기타", "기타"))[1])
        df = df.dropna(subset=["date", "code", "sector", "close"]).sort_values(["code", "date"]).reset_index(drop=True)
        return df

    frames = []
    for code, (name, sector) in FALLBACK_UNIVERSE.items():
        sub = fetch_sise_json(code, start_dt, end_dt)
        if sub.empty:
            continue
        sub["code"] = code
        sub["name"] = name
        sub["sector"] = sector
        sub["mktcap_100mn"] = np.nan
        frames.append(sub)
    if not frames:
        raise RuntimeError("시세 데이터를 불러오지 못했습니다.")
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)


def analyze_q1_sector_spread(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1. 업종별 누적 수익률 / 업종 간 분산의 시간 변화 / 차이를 만든 상위 업종"""
    p = panel.copy()
    p["ret"] = p.groupby("code")["close"].pct_change()
    p["cum_ret"] = p.groupby("code")["close"].transform(lambda s: s / s.iloc[0] - 1.0)

    # 일별·종목별 수익률 -> 일별·업종별 동일가중(및 시총가중) 수익률/누적수익률
    sec_daily = (
        p.groupby(["date", "sector"], as_index=False)
        .agg(
            ew_ret=("ret", "mean"),
            ew_cum_ret=("cum_ret", "mean"),
            n_stocks=("code", "nunique"),
        )
        .sort_values(["date", "sector"])
    )

    # 업종 간 분산(cross-sectional variance) 및 표준편차(dispersion)의 시간 변화
    disp_daily = (
        sec_daily.groupby("date", as_index=False)
        .agg(
            n_sectors=("sector", "nunique"),
            cum_ret_var=("ew_cum_ret", lambda s: np.var(s, ddof=1) if len(s) > 1 else np.nan),
            cum_ret_std=("ew_cum_ret", lambda s: np.std(s, ddof=1) if len(s) > 1 else np.nan),
            cum_ret_range=("ew_cum_ret", lambda s: np.max(s) - np.min(s) if len(s) > 0 else np.nan),
            daily_ret_var=("ew_ret", lambda s: np.var(s, ddof=1) if len(s) > 1 else np.nan),
        )
        .sort_values("date")
    )
    disp_daily["ym"] = disp_daily["date"].dt.to_period("M").astype(str)
    disp_monthly = disp_daily.groupby("ym", as_index=False).agg(
        mean_cum_ret_var=("cum_ret_var", "mean"),
        end_cum_ret_var=("cum_ret_var", "last"),
        mean_cum_ret_range=("cum_ret_range", "mean"),
        mean_daily_ret_var=("daily_ret_var", "mean"),
    )

    # 기간 전체 업종별 누적수익률 및 전체 업종 평균 대비 기여도(차이를 만든 상위 업종)
    sec_summary = (
        sec_daily.groupby("sector", as_index=False)
        .agg(final_cum_ret=("ew_cum_ret", "last"), mean_daily_ret=("ew_ret", "mean"))
    )
    overall_mean = sec_summary["final_cum_ret"].mean()
    sec_summary["diff_from_mean"] = sec_summary["final_cum_ret"] - overall_mean
    sec_summary["abs_diff_from_mean"] = sec_summary["diff_from_mean"].abs()
    sec_summary["sq_diff_from_mean"] = sec_summary["diff_from_mean"] ** 2
    var_sum = sec_summary["sq_diff_from_mean"].sum()
    sec_summary["var_share_pct"] = np.where(var_sum > 0, sec_summary["sq_diff_from_mean"] / var_sum * 100.0, np.nan)
    sec_summary = sec_summary.sort_values("sq_diff_from_mean", ascending=False).reset_index(drop=True)

    return sec_summary, disp_monthly, disp_daily


def analyze_q2_comovement(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """2. 업종 간 일별 수익률 상관 / 상관의 시간 변화 / 시장 충격일 vs 보통일 상관 비교"""
    p = panel.copy()
    p["ret"] = p.groupby("code")["close"].pct_change()

    sec_ret_long = (
        p.groupby(["date", "sector"], as_index=False)["ret"]
        .mean()
        .dropna(subset=["ret"])
    )
    ret_mat = sec_ret_long.pivot(index="date", columns="sector", values="ret").dropna(how="all")

    # (a) 전체 기간 업종 간 일별 수익률 상관계수 행렬
    corr_all = ret_mat.corr()

    # (b) 상관의 시간 변화 (분기별/월별 평균 페어와이즈 상관계수)
    cols = list(ret_mat.columns)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    time_rows = []
    for ym, grp in ret_mat.groupby(ret_mat.index.to_period("M")):
        if len(grp) < 5 or len(pairs) == 0:
            continue
        cmat = grp.corr()
        vals = [cmat.loc[a, b] for a, b in pairs if a in cmat.index and b in cmat.columns]
        vals = [v for v in vals if pd.notna(v)]
        time_rows.append({
            "ym": str(ym),
            "n_days": len(grp),
            "mean_pairwise_corr": float(np.mean(vals)) if vals else np.nan,
            "median_pairwise_corr": float(np.median(vals)) if vals else np.nan,
        })
    corr_time = pd.DataFrame(time_rows)

    # (c) 시장이 크게 흔들린 날(동일가중 시장수익률 하위 10% 충격일) vs 보통 날 상관 비교
    mkt_ret = ret_mat.mean(axis=1)
    q10 = mkt_ret.quantile(0.10)
    shock_mask = mkt_ret <= q10
    normal_mask = ~shock_mask

    corr_shock = ret_mat.loc[shock_mask].corr()
    corr_normal = ret_mat.loc[normal_mask].corr()

    shock_vals = [
        corr_shock.loc[a, b] for a, b in pairs if a in corr_shock.index and b in corr_shock.columns
    ]
    normal_vals = [
        corr_normal.loc[a, b] for a, b in pairs if a in corr_normal.index and b in corr_normal.columns
    ]
    shock_vals = [v for v in shock_vals if pd.notna(v)]
    normal_vals = [v for v in normal_vals if pd.notna(v)]

    regime_comp = pd.DataFrame([
        {
            "regime": f"충격일 (하위 10%, <= {q10:.2%})",
            "n_days": int(shock_mask.sum()),
            "mean_pairwise_corr": float(np.mean(shock_vals)) if shock_vals else np.nan,
        },
        {
            "regime": "보통일 (상위 90%)",
            "n_days": int(normal_mask.sum()),
            "mean_pairwise_corr": float(np.mean(normal_vals)) if normal_vals else np.nan,
        },
    ])

    return corr_all, corr_time, regime_comp


def analyze_q3_volume_vs_price(panel: pd.DataFrame, focus_codes: list[str] | None = None) -> pd.DataFrame:
    """3. 거래량 급증 정의 / 급증 뒤 5·10·20일 수익률 / 정의 변경 시 결과 변화"""
    if focus_codes is None:
        focus_codes = ["035420", "035720"]  # NAVER, 카카오

    p = panel[panel["code"].isin(focus_codes)].copy().sort_values(["code", "date"])
    rows = []

    for code, grp in p.groupby("code"):
        g = grp.sort_values("date").reset_index(drop=True)
        name = g["name"].iloc[0]
        vol_ma20 = g["volume"].rolling(20, min_periods=5).mean().shift(1)
        vol_q95 = g["volume"].rolling(60, min_periods=20).quantile(0.95).shift(1)

        defs: dict[str, pd.Series] = {
            "MA20_2.0x (직전20일 평균 2배↑)": g["volume"] >= 2.0 * vol_ma20,
            "MA20_1.5x (직전20일 평균 1.5배↑)": g["volume"] >= 1.5 * vol_ma20,
            "MA20_3.0x (직전20일 평균 3배↑)": g["volume"] >= 3.0 * vol_ma20,
            "Q95_60d (직전60일 상위 5%↑)": g["volume"] >= vol_q95,
        }

        fwd_5 = g["close"].shift(-5) / g["close"] - 1.0
        fwd_10 = g["close"].shift(-10) / g["close"] - 1.0
        fwd_20 = g["close"].shift(-20) / g["close"] - 1.0

        for def_name, cond in defs.items():
            cond = cond.fillna(False)
            n_ev = int(cond.sum())
            rows.append({
                "code": code,
                "name": name,
                "surge_def": def_name,
                "n_events": n_ev,
                "fwd_5d_ret_mean": float(fwd_5.loc[cond].mean()) if n_ev > 0 else np.nan,
                "fwd_10d_ret_mean": float(fwd_10.loc[cond].mean()) if n_ev > 0 else np.nan,
                "fwd_20d_ret_mean": float(fwd_20.loc[cond].mean()) if n_ev > 0 else np.nan,
                "fwd_5d_win_rate": float((fwd_5.loc[cond] > 0).mean()) if n_ev > 0 else np.nan,
                "fwd_20d_win_rate": float((fwd_20.loc[cond] > 0).mean()) if n_ev > 0 else np.nan,
            })

    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel(YEARS)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print(f"데이터 기간: {panel['date'].min().date()} ~ {panel['date'].max().date()} | 종목수: {panel['code'].nunique()} | 업종수: {panel['sector'].nunique()}")

    # [1] 업종 간 수익률 차이
    sec_sum, disp_m, disp_d = analyze_q1_sector_spread(panel)
    print("\n=== [1-A] 업종별 누적수익률 및 분산 기여도 상위 업종 ===")
    print(sec_sum.to_string(index=False))
    print("\n=== [1-B] 업종 간 분산 시간 변화 (최근 12개월) ===")
    print(disp_m.tail(12).to_string(index=False))

    # [2] 업종 간 동조화
    corr_all, corr_time, regime_comp = analyze_q2_comovement(panel)
    print("\n=== [2-A] 업종 간 일별 수익률 상관계수 행렬 ===")
    print(corr_all.round(3).to_string())
    print("\n=== [2-B] 업종 간 평균 상관계수 시간 변화 (최근 12개월) ===")
    print(corr_time.tail(12).to_string(index=False))
    print("\n=== [2-C] 시장 충격일 vs 보통일 업종 간 평균 상관 비교 ===")
    print(regime_comp.to_string(index=False))

    # [3] 거래량 급증과 이후 수익률(NAVER, 카카오)
    q3_df = analyze_q3_volume_vs_price(panel, focus_codes=["035420", "035720"])
    print("\n=== [3] 거래량 급증 정의별 이후 5·10·20일 평균 수익률 (NAVER·카카오) ===")
    print(q3_df.to_string(index=False))

    sec_sum.to_csv(f"{OUT_DIR}/q1_sector_contribution.csv", index=False, encoding="utf-8-sig")
    disp_m.to_csv(f"{OUT_DIR}/q1_dispersion_monthly.csv", index=False, encoding="utf-8-sig")
    disp_d.to_csv(f"{OUT_DIR}/q1_dispersion_daily.csv", index=False, encoding="utf-8-sig")
    corr_all.to_csv(f"{OUT_DIR}/q2_corr_matrix.csv", encoding="utf-8-sig")
    corr_time.to_csv(f"{OUT_DIR}/q2_corr_monthly.csv", index=False, encoding="utf-8-sig")
    regime_comp.to_csv(f"{OUT_DIR}/q2_shock_vs_normal.csv", index=False, encoding="utf-8-sig")
    q3_df.to_csv(f"{OUT_DIR}/q3_volume_surge_fwd_ret.csv", index=False, encoding="utf-8-sig")
    print(f"\n결과 CSV 저장 완료: {OUT_DIR}/")


if __name__ == "__main__":
    main()
