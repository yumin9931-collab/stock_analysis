from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

PARQUET_GLOB = "/home/s202423331/parquet/*.parquet"
OUT_DIR = "/home/s202423331/results_spec1_spec2"
MIN_STOCKS_PER_SECTOR = 3
EXTREME_RET_THRESHOLD = 0.50  # ±50% 초과 일별 수익률 제외 및 결함 기록


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


def _clean_sector(val: object) -> str:
    """'(업종명 :' 잘림, 날짜/시간 문자열, 문장형 설명 등 파싱 오류 라벨을 '미분류'로 정제"""
    if val is None:
        return "미분류"
    s = str(val).strip()
    if s.startswith("(업종명"):
        parts = s.split(":", 1)
        s = parts[1].strip(" )") if len(parts) > 1 else ""
    if (
        not s
        or s in {"nan", "None", "미분류", "N/A", "-"}
        or "/" in s
        or ":" in s
        or len(s) > 25
    ):
        return "미분류"
    return s


def _mad(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    med = np.median(valid)
    return float(np.median(np.abs(valid - med)))


def _iqr(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    q75, q25 = np.percentile(valid, [75, 25])
    return float(q75 - q25)


def load_and_clean_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(glob.glob(PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(f"Parquet 파일이 없습니다: {PARQUET_GLOB}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip() if "name" in df.columns else df["code"]
    df["raw_sector"] = df["sector"].astype(str).str.strip() if "sector" in df.columns else "미분류"
    df["sector"] = df["raw_sector"].map(_clean_sector)
    df["close"] = df["close"].map(_num)
    df["mktcap_100mn"] = df["mktcap_100mn"].map(_num) if "mktcap_100mn" in df.columns else np.nan

    df = (
        df.dropna(subset=["date", "code", "sector", "close"])
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"])
        .reset_index(drop=True)
    )

    # 종목별 산술 일별 수익률 및 전일 시가총액 가중치
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["w_mcap"] = df.groupby("code")["mktcap_100mn"].shift(1)

    # 결함 기록: 일별 수익률이 ±50% 밖인 종목·날짜 (액면분할·감자·파싱 오류 등)
    extreme_mask = df["ret"].abs() > EXTREME_RET_THRESHOLD
    defects_df = df.loc[
        extreme_mask, ["date", "code", "name", "sector", "raw_sector", "close", "ret", "w_mcap"]
    ].copy()
    defects_df["reason"] = "abs(daily_ret) > 0.50 (액면분할/감자/급등락 의심)"

    clean_df = df.loc[~extreme_mask].copy().reset_index(drop=True)

    # '미분류' 및 종목 수 하한(3개 미만) 업종 확인 및 필터링
    non_unclass = clean_df[clean_df["sector"] != "미분류"]
    sec_counts = non_unclass.groupby("sector")["code"].nunique()
    excluded_sectors = sec_counts[sec_counts < MIN_STOCKS_PER_SECTOR]
    valid_sectors = set(sec_counts[sec_counts >= MIN_STOCKS_PER_SECTOR].index)

    valid_df = clean_df[clean_df["sector"].isin(valid_sectors)].copy().reset_index(drop=True)
    return valid_df, defects_df, excluded_sectors.reset_index(name="n_stocks")


def calc_spec1_sector_returns(valid_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """견본 1: (a) 시가총액 가중, (b) 동일 가중, (c) 기간 양 끝 비교"""
    mcap_sub = valid_df.dropna(subset=["ret", "w_mcap"]).copy()
    mcap_sub = mcap_sub[mcap_sub["w_mcap"] > 0]
    mcap_sub["weighted_ret"] = mcap_sub["ret"] * mcap_sub["w_mcap"]

    mcap_grp = mcap_sub.groupby(["date", "sector"], as_index=False).agg(
        w_sum=("w_mcap", "sum"),
        wr_sum=("weighted_ret", "sum"),
    )
    mcap_grp["ret_mcap"] = np.where(mcap_grp["w_sum"] > 0, mcap_grp["wr_sum"] / mcap_grp["w_sum"], np.nan)
    sector_ret_mcap = mcap_grp.pivot(index="date", columns="sector", values="ret_mcap").sort_index()
    sector_cum_mcap = (1.0 + sector_ret_mcap.fillna(0.0)).cumprod() - 1.0

    ew_grp = valid_df.dropna(subset=["ret"]).groupby(["date", "sector"], as_index=False).agg(
        ret_ew=("ret", "mean")
    )
    sector_ret_ew = ew_grp.pivot(index="date", columns="sector", values="ret_ew").sort_index()
    sector_cum_ew = (1.0 + sector_ret_ew.fillna(0.0)).cumprod() - 1.0

    ep_stk = (
        valid_df.sort_values(["code", "date"])
        .groupby(["code", "sector"], as_index=False)
        .agg(start_close=("close", "first"), end_close=("close", "last"))
    )
    ep_stk["endpoint_ret"] = ep_stk["end_close"] / ep_stk["start_close"] - 1.0
    endpoint_sec = (
        ep_stk.groupby("sector", as_index=False)
        .agg(n_surviving_stocks=("code", "count"), endpoint_mean_ret=("endpoint_ret", "mean"))
    )

    final_mcap = sector_cum_mcap.iloc[-1].rename("cum_ret_mcap_w")
    final_ew = sector_cum_ew.iloc[-1].rename("cum_ret_equal_w")
    comp = (
        pd.concat([final_mcap, final_ew], axis=1)
        .reset_index()
        .merge(endpoint_sec, on="sector", how="left")
    )
    comp["diff_mcap_minus_ew"] = comp["cum_ret_mcap_w"] - comp["cum_ret_equal_w"]
    comp["survivorship_note"] = "기간 내 생존 종목 기준(상장폐지 제외 편향 가능)"
    comp = comp.sort_values("cum_ret_mcap_w", ascending=False).reset_index(drop=True)

    focus_rows = []
    for code, label in [("035420", "NAVER"), ("035720", "카카오")]:
        sub = valid_df[valid_df["code"] == code].sort_values("date")
        if sub.empty:
            continue
        sec = sub["sector"].iloc[-1]
        stock_cum = float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0)
        sec_mcap_cum = float(sector_cum_mcap[sec].iloc[-1]) if sec in sector_cum_mcap.columns else np.nan
        sec_ew_cum = float(sector_cum_ew[sec].iloc[-1]) if sec in sector_cum_ew.columns else np.nan
        focus_rows.append({
            "code": code,
            "name": label,
            "sector": sec,
            "stock_cum_ret": stock_cum,
            "sector_cum_ret_mcap_w": sec_mcap_cum,
            "sector_cum_ret_equal_w": sec_ew_cum,
            "excess_vs_sector_mcap_w": stock_cum - sec_mcap_cum if pd.notna(sec_mcap_cum) else np.nan,
        })

    return {
        "sector_ret_mcap": sector_ret_mcap,
        "sector_cum_mcap": sector_cum_mcap,
        "sector_ret_ew": sector_ret_ew,
        "sector_cum_ew": sector_cum_ew,
        "sector_comparison": comp,
        "focus_naver_kakao": pd.DataFrame(focus_rows),
    }


def calc_spec2_dispersion(
    sector_ret_mcap: pd.DataFrame,
    sector_cum_mcap: pd.DataFrame,
    sector_ret_ew: pd.DataFrame,
    sector_cum_ew: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """견본 2: (a) 월별 횡단면 분산 및 민감도(IQR/MAD/초과수익률), (b) 누적수익률 퍼짐 및 상위/하위 업종"""
    monthly_mcap = sector_ret_mcap.resample("ME").apply(lambda x: (1.0 + x.fillna(0.0)).prod() - 1.0)
    mkt_monthly_mcap = monthly_mcap.mean(axis=1)
    excess_monthly_mcap = monthly_mcap.sub(mkt_monthly_mcap, axis=0)
    monthly_ew = sector_ret_ew.resample("ME").apply(lambda x: (1.0 + x.fillna(0.0)).prod() - 1.0)

    disp_monthly = pd.DataFrame({
        "month": monthly_mcap.index.strftime("%Y-%m"),
        "std_mcap_w": monthly_mcap.std(axis=1, ddof=1).values,
        "iqr_mcap_w": monthly_mcap.apply(lambda r: _iqr(r.to_numpy(dtype=float)), axis=1).values,
        "mad_mcap_w": monthly_mcap.apply(lambda r: _mad(r.to_numpy(dtype=float)), axis=1).values,
        "std_excess_vs_mkt": excess_monthly_mcap.std(axis=1, ddof=1).values,
        "std_equal_w": monthly_ew.std(axis=1, ddof=1).values,
    })

    spread_daily = pd.DataFrame({
        "date": sector_cum_mcap.index,
        "cum_std_mcap_w": sector_cum_mcap.std(axis=1, ddof=1).values,
        "cum_range_mcap_w": (sector_cum_mcap.max(axis=1) - sector_cum_mcap.min(axis=1)).values,
        "cum_std_equal_w": sector_cum_ew.std(axis=1, ddof=1).values,
        "cum_range_equal_w": (sector_cum_ew.max(axis=1) - sector_cum_ew.min(axis=1)).values,
    })

    mid_m = len(disp_monthly) // 2
    first_half_std = float(disp_monthly["std_mcap_w"].iloc[:mid_m].mean()) if mid_m > 0 else np.nan
    second_half_std = float(disp_monthly["std_mcap_w"].iloc[mid_m:].mean()) if mid_m > 0 else np.nan

    mid_d = len(spread_daily) // 2
    first_half_cum_std = float(spread_daily["cum_std_mcap_w"].iloc[:mid_d].mean()) if mid_d > 0 else np.nan
    second_half_cum_std = float(spread_daily["cum_std_mcap_w"].iloc[mid_d:].mean()) if mid_d > 0 else np.nan

    trend_summary = pd.DataFrame([{
        "monthly_std_first_half": first_half_std,
        "monthly_std_second_half": second_half_std,
        "monthly_std_change": second_half_std - first_half_std if pd.notna(first_half_std) and pd.notna(second_half_std) else np.nan,
        "cum_std_first_half": first_half_cum_std,
        "cum_std_second_half": second_half_cum_std,
        "cum_std_change": second_half_cum_std - first_half_cum_std if pd.notna(first_half_cum_std) and pd.notna(second_half_cum_std) else np.nan,
    }])

    end_cum = sector_cum_mcap.iloc[-1].dropna().sort_values(ascending=False)
    top_bottom_df = pd.DataFrame({
        "rank_high_to_low": range(1, len(end_cum) + 1),
        "sector": end_cum.index,
        "final_cum_ret_mcap_w": end_cum.values,
        "diff_from_sector_mean": (end_cum - end_cum.mean()).values,
    })

    return {
        "disp_monthly": disp_monthly,
        "spread_daily": spread_daily,
        "trend_summary": trend_summary,
        "top_bottom_sectors": top_bottom_df,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    valid_df, defects_df, excluded_sec = load_and_clean_panel()
    print(f"유효 시세 기간: {valid_df['date'].min().date()} ~ {valid_df['date'].max().date()}")
    print(f"분석 대상 종목수: {valid_df['code'].nunique()} | 유효 업종수(>= {MIN_STOCKS_PER_SECTOR}종목): {valid_df['sector'].nunique()}")
    print(f"제외된 극단 일별수익률(|ret| > {EXTREME_RET_THRESHOLD:.0%}) 행수: {len(defects_df)}")
    print(f"종목수 {MIN_STOCKS_PER_SECTOR} 미만 제외 업종수: {len(excluded_sec)}")

    s1 = calc_spec1_sector_returns(valid_df)
    print("\n=== [견본 1-A] NAVER · 카카오 및 소속 업종 누적수익률 비교 ===")
    print(s1["focus_naver_kakao"].to_string(index=False))

    print("\n=== [견본 1-B] 업종별 누적수익률 비교 ((a)시총가중 vs (b)동일가중 vs (c)양끝비교, 상위 15개) ===")
    print(s1["sector_comparison"].head(15).to_string(index=False))

    s2 = calc_spec2_dispersion(
        s1["sector_ret_mcap"],
        s1["sector_cum_mcap"],
        s1["sector_ret_ew"],
        s1["sector_cum_ew"],
    )
    print("\n=== [견본 2-A] 전반부 vs 후반부 분산 변화 요약 ===")
    print(s2["trend_summary"].to_string(index=False))

    print("\n=== [견본 2-B] 월별 업종 간 분산 및 민감도(IQR/MAD/초과수익률, 최근 12개월) ===")
    print(s2["disp_monthly"].tail(12).to_string(index=False))

    print("\n=== [견본 2-C] 누적수익률 기준 격차를 만든 상위 5개 / 하위 5개 업종 ===")
    tb = s2["top_bottom_sectors"]
    print(pd.concat([tb.head(5), tb.tail(5)], ignore_index=True).to_string(index=False))

    s1["sector_comparison"].to_csv(f"{OUT_DIR}/spec1_sector_comparison.csv", index=False, encoding="utf-8-sig")
    s1["focus_naver_kakao"].to_csv(f"{OUT_DIR}/spec1_naver_kakao_focus.csv", index=False, encoding="utf-8-sig")
    s2["disp_monthly"].to_csv(f"{OUT_DIR}/spec2_dispersion_monthly.csv", index=False, encoding="utf-8-sig")
    s2["spread_daily"].to_csv(f"{OUT_DIR}/spec2_spread_daily.csv", index=False, encoding="utf-8-sig")
    s2["trend_summary"].to_csv(f"{OUT_DIR}/spec2_trend_summary.csv", index=False, encoding="utf-8-sig")
    s2["top_bottom_sectors"].to_csv(f"{OUT_DIR}/spec2_top_bottom_sectors.csv", index=False, encoding="utf-8-sig")
    defects_df.to_csv(f"{OUT_DIR}/defects_extreme_ret.csv", index=False, encoding="utf-8-sig")
    excluded_sec.to_csv(f"{OUT_DIR}/defects_excluded_small_sectors.csv", index=False, encoding="utf-8-sig")
    print(f"\n결과 및 결함 기록 CSV 저장 완료: {OUT_DIR}/")


if __name__ == "__main__":
    main()

