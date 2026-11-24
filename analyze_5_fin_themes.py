from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

PRICE_PARQUET_GLOB = "/home/s202423331/parquet/*.parquet"
FIN_PARQUET_GLOB = "/home/s202423331/fin_parquet/*.parquet"
OUT_DIR = "/home/s202423331/results_5fin_themes"


def _num(x: object) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("%", "")
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


def _is_est_flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "e"}) | (s == True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    p_files = sorted(glob.glob(PRICE_PARQUET_GLOB))
    f_files = sorted(glob.glob(FIN_PARQUET_GLOB))
    if not p_files:
        raise FileNotFoundError(f"시세 Parquet 파일이 없습니다: {PRICE_PARQUET_GLOB}")
    if not f_files:
        raise FileNotFoundError(f"재무 Parquet 파일이 없습니다: {FIN_PARQUET_GLOB}")

    price = pd.concat([pd.read_parquet(f) for f in p_files], ignore_index=True)
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["code"] = price["code"].astype(str).str.zfill(6)
    price["close"] = price["close"].map(_num)
    price["volume"] = price["volume"].map(_num)
    if "mktcap_100mn" in price.columns:
        price["mktcap_100mn"] = price["mktcap_100mn"].map(_num)
    if "sector" not in price.columns:
        price["sector"] = "기타"
    price = (
        price.dropna(subset=["date", "code", "close"])
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"])
        .reset_index(drop=True)
    )

    fin = pd.concat([pd.read_parquet(f) for f in f_files], ignore_index=True)
    fin["date"] = pd.to_datetime(fin["date"], errors="coerce")
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin["period"] = fin["period"].astype(str).str.strip()
    fin["metric"] = fin["metric"].astype(str).str.strip()
    fin["val_num"] = fin["value"].map(_num)
    fin["is_est_bool"] = _is_est_flag(fin["is_est"])

    # fin에 sector가 없으면 price의 대표 sector 매핑
    code_sec = (
        price.dropna(subset=["sector"])
        .groupby("code")["sector"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
    )
    code_name = (
        price.dropna(subset=["name"])
        .groupby("code")["name"]
        .last()
        if "name" in price.columns
        else pd.Series(dtype=str)
    )
    fin["sector"] = fin["code"].map(code_sec).fillna("기타")
    if "name" not in fin.columns:
        fin["name"] = fin["code"].map(code_name).fillna(fin["code"])
    else:
        fin["name"] = fin["name"].fillna(fin["code"].map(code_name)).fillna(fin["code"])

    return price, fin


def analyze_q1_opm_vs_return(price: pd.DataFrame, fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """1. 영업이익률이 높은 업종이 주가 수익률도 높았는가"""
    # 종목별 2.5년 누적 수익률
    stk_ret = (
        price.groupby(["code", "name", "sector"], as_index=False)
        .agg(
            start_close=("close", "first"),
            end_close=("close", "last"),
            n_days=("close", "count"),
        )
    )
    stk_ret = stk_ret[stk_ret["n_days"] >= 20].copy()
    stk_ret["cum_ret"] = stk_ret["end_close"] / stk_ret["start_close"] - 1.0

    # 확정 실적(is_est == False) 기준 영업이익률 추출
    opm = fin[(fin["metric"] == "영업이익률") & (~fin["is_est_bool"]) & fin["val_num"].notna()].copy()
    opm = opm.sort_values(["code", "period", "date"]).drop_duplicates(["code", "period"], keep="last")

    opm_stk = (
        opm.sort_values(["code", "period"])
        .groupby("code", as_index=False)
        .agg(
            opm_first=("val_num", "first"),
            opm_last=("val_num", "last"),
            opm_mean=("val_num", "mean"),
            n_periods=("val_num", "count"),
        )
    )
    opm_stk["opm_change"] = opm_stk["opm_last"] - opm_stk["opm_first"]

    merged = stk_ret.merge(opm_stk, on="code", how="inner")

    # (a) 업종별 평균 영업이익률 vs 누적 수익률
    sec_df = (
        merged.groupby("sector", as_index=False)
        .agg(
            n_stocks=("code", "nunique"),
            sec_opm_mean=("opm_mean", "mean"),
            sec_opm_first=("opm_first", "mean"),
            sec_opm_change=("opm_change", "mean"),
            sec_cum_ret=("cum_ret", "mean"),
        )
        .sort_values("sec_opm_mean", ascending=False)
        .reset_index(drop=True)
    )
    sec_corr_level = sec_df["sec_opm_mean"].corr(sec_df["sec_cum_ret"]) if len(sec_df) > 2 else np.nan
    sec_corr_chg = sec_df["sec_opm_change"].corr(sec_df["sec_cum_ret"]) if len(sec_df) > 2 else np.nan

    # (b) 업종 내부(within-sector)에서도 같은 방향인가
    within_rows = []
    for sec, grp in merged.groupby("sector"):
        if len(grp) < 3:
            continue
        within_rows.append({
            "sector": sec,
            "n_stocks": len(grp),
            "within_corr_opm_ret": float(grp["opm_mean"].corr(grp["cum_ret"])),
            "within_corr_opm_chg_ret": float(grp["opm_change"].corr(grp["cum_ret"])),
        })
    within_df = (
        pd.DataFrame(within_rows).sort_values("within_corr_opm_ret", ascending=False)
        if within_rows
        else pd.DataFrame(columns=["sector", "n_stocks", "within_corr_opm_ret", "within_corr_opm_chg_ret"])
    )

    # (c) 이익률이 올라간 업종 vs 원래 높았던 업종 2x2 비교
    med_init = sec_df["sec_opm_first"].median()
    med_chg = sec_df["sec_opm_change"].median()
    sec_df["high_initial_opm"] = sec_df["sec_opm_first"] >= med_init
    sec_df["improved_opm"] = sec_df["sec_opm_change"] > 0

    quad_df = (
        sec_df.groupby(["high_initial_opm", "improved_opm"], as_index=False)
        .agg(
            n_sectors=("sector", "count"),
            mean_cum_ret=("sec_cum_ret", "mean"),
            median_cum_ret=("sec_cum_ret", "median"),
        )
    )
    summary_corr = pd.DataFrame([{
        "sector_corr_opm_level_vs_ret": sec_corr_level,
        "sector_corr_opm_change_vs_ret": sec_corr_chg,
        "median_within_sector_corr": float(within_df["within_corr_opm_ret"].median()) if not within_df.empty else np.nan,
    }])

    return {
        "sector_opm_ret": sec_df,
        "within_sector": within_df,
        "quadrant_2x2": quad_df,
        "summary_corr": summary_corr,
    }


def analyze_q2_earnings_event(price: pd.DataFrame, fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """2. 실적 발표 뒤 주가는 며칠 만에 반응을 끝내는가 (실적 표 전환일 기준 전후 10일 CAR/수익률)"""
    p = price.copy()
    p["ret"] = p.groupby("code")["close"].pct_change()
    mkt_ret = p.groupby("date")["ret"].mean().rename("mkt_ret")
    p = p.merge(mkt_ret, on="date", how="left")
    p["abn_ret"] = p["ret"] - p["mkt_ret"]

    # 같은 (code, period)의 영업이익이 is_est=True -> is_est=False로 바뀐 첫 날짜 = 실적 표 전환일(발표일 근사)
    op = fin[(fin["metric"] == "영업이익") & fin["val_num"].notna()].sort_values(["code", "period", "date"])
    events = []
    for (code, period), grp in op.groupby(["code", "period"]):
        est_rows = grp[grp["is_est_bool"]]
        act_rows = grp[~grp["is_est_bool"]]
        if est_rows.empty or act_rows.empty:
            continue
        announce_dt = act_rows["date"].iloc[0]
        # 발표 직전 마지막 추정치 vs 확정 실적
        prior_est = est_rows[est_rows["date"] <= announce_dt]
        if prior_est.empty:
            continue
        est_val = float(prior_est["val_num"].iloc[-1])
        act_val = float(act_rows["val_num"].iloc[0])
        if abs(est_val) < 1e-9:
            continue
        surprise = (act_val - est_val) / abs(est_val)
        events.append({
            "code": code,
            "name": grp["name"].iloc[-1],
            "sector": grp["sector"].iloc[-1],
            "period": period,
            "announce_date": announce_dt,
            "est_op": est_val,
            "act_op": act_val,
            "surprise_pct": surprise,
            "beat": act_val >= est_val,
        })

    ev_df = pd.DataFrame(events)
    if ev_df.empty:
        return {"events": ev_df, "event_window": pd.DataFrame()}

    # 종목별 거래일 인덱스 매핑 후 [-10, +10] 윈도우 수익률 계산
    win_rows = []
    for code, grp in p.groupby("code"):
        sub_ev = ev_df[ev_df["code"] == code]
        if sub_ev.empty:
            continue
        g = grp.sort_values("date").reset_index(drop=True)
        dates = g["date"].values
        for _, erow in sub_ev.iterrows():
            idx = int(np.searchsorted(dates, np.datetime64(erow["announce_date"])))
            if idx - 10 < 0 or idx + 10 >= len(g):
                continue
            base_close = g["close"].iloc[idx - 1]
            if pd.isna(base_close) or base_close <= 0:
                continue
            for tau in range(-10, 11):
                c_tau = g["close"].iloc[idx + tau]
                cum_raw = c_tau / base_close - 1.0
                cum_abn = g["abn_ret"].iloc[idx - 10 : idx + tau + 1].sum()
                win_rows.append({
                    "code": code,
                    "period": erow["period"],
                    "beat": erow["beat"],
                    "tau": tau,
                    "cum_ret_from_m1": cum_raw,
                    "car_from_m10": cum_abn,
                })

    win_df = pd.DataFrame(win_rows)
    if win_df.empty:
        return {"events": ev_df, "event_window": pd.DataFrame()}

    tau_summary = (
        win_df.groupby(["beat", "tau"], as_index=False)
        .agg(
            n_events=("code", "count"),
            mean_cum_ret_from_m1=("cum_ret_from_m1", "mean"),
            mean_car_from_m10=("car_from_m10", "mean"),
        )
        .sort_values(["beat", "tau"])
    )
    return {"events": ev_df, "event_window": tau_summary}


def analyze_q3_estimate_errors(fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """3. 추정치는 어느 업종에서 가장 많이 빗나갔는가 / 발표 전 수정 경로 / 방향 쏠림"""
    op = fin[(fin["metric"] == "영업이익") & fin["val_num"].notna()].sort_values(["code", "period", "date"])
    err_rows = []
    rev_rows = []

    for (code, period), grp in op.groupby(["code", "period"]):
        est_rows = grp[grp["is_est_bool"]].sort_values("date")
        act_rows = grp[~grp["is_est_bool"]].sort_values("date")
        if est_rows.empty or act_rows.empty:
            continue
        act_dt = act_rows["date"].iloc[0]
        act_val = float(act_rows["val_num"].iloc[0])
        if abs(act_val) < 1e-9:
            continue

        pre_est = est_rows[est_rows["date"] <= act_dt]
        if pre_est.empty:
            continue
        last_est = float(pre_est["val_num"].iloc[-1])
        first_est = float(pre_est["val_num"].iloc[0])

        # 오차 = (실적 - 추정치) / |실적|  (음수면 추정치가 실적보다 과대 낙관)
        signed_err = (act_val - last_est) / abs(act_val)
        abs_err = abs(signed_err)
        err_rows.append({
            "code": code,
            "name": grp["name"].iloc[-1],
            "sector": grp["sector"].iloc[-1],
            "period": period,
            "first_est": first_est,
            "last_est": last_est,
            "actual_op": act_val,
            "signed_error": signed_err,
            "abs_error": abs_err,
            "over_estimated": last_est > act_val,
        })

        # 발표 전 몇 개월 동안의 수정 경로 (발표일 기준 D-180 ~ D-0 월별 버킷)
        for _, er in pre_est.iterrows():
            days_before = (act_dt - er["date"]).days
            months_before = int(days_before // 30)
            if 0 <= months_before <= 6:
                rev_rows.append({
                    "sector": grp["sector"].iloc[-1],
                    "code": code,
                    "period": period,
                    "months_before": months_before,
                    "est_to_actual_ratio": float(er["val_num"]) / abs(act_val),
                    "est_bias_pct": (float(er["val_num"]) - act_val) / abs(act_val),
                })

    err_df = pd.DataFrame(err_rows)
    rev_df = pd.DataFrame(rev_rows)
    if err_df.empty:
        return {"sector_error": pd.DataFrame(), "revision_path": pd.DataFrame(), "raw_errors": err_df}

    sec_err = (
        err_df.groupby("sector", as_index=False)
        .agg(
            n_obs=("code", "count"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_signed_error=("signed_error", "mean"),
            over_est_share=("over_estimated", "mean"),
        )
        .sort_values("mean_abs_error", ascending=False)
        .reset_index(drop=True)
    )

    rev_path = (
        rev_df.groupby("months_before", as_index=False)
        .agg(
            n_obs=("code", "count"),
            mean_est_bias_pct=("est_bias_pct", "mean"),
            median_est_bias_pct=("est_bias_pct", "median"),
        )
        .sort_values("months_before", ascending=False)
        if not rev_df.empty
        else pd.DataFrame()
    )
    return {"sector_error": sec_err, "revision_path": rev_path, "raw_errors": err_df}


def analyze_q4_debt_vs_volatility(price: pd.DataFrame, fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """4. 부채비율이 높은 업종이 더 크게 흔들렸는가 (전체 vs 시장 급락 구간 비교)"""
    p = price.copy()
    p["ret"] = p.groupby("code")["close"].pct_change()
    mkt_ret = p.groupby("date")["ret"].mean()
    q10 = mkt_ret.quantile(0.10)
    crash_dates = set(mkt_ret[mkt_ret <= q10].index)

    p["is_crash"] = p["date"].isin(crash_dates)

    # 종목별 전체 변동성 / 급락일 평균 수익률 & 변동성 / 보통일 변동성
    stk_vol = (
        p.groupby(["code", "name", "sector"], as_index=False)
        .agg(
            ann_vol_all=("ret", lambda s: s.std() * np.sqrt(252)),
            vol_crash=("ret", lambda s: s[p.loc[s.index, "is_crash"]].std() * np.sqrt(252)),
            vol_normal=("ret", lambda s: s[~p.loc[s.index, "is_crash"]].std() * np.sqrt(252)),
            mean_ret_crash=("ret", lambda s: s[p.loc[s.index, "is_crash"]].mean()),
            mean_ret_normal=("ret", lambda s: s[~p.loc[s.index, "is_crash"]].mean()),
        )
    )

    debt = fin[(fin["metric"] == "부채비율") & (~fin["is_est_bool"]) & fin["val_num"].notna()].copy()
    debt_stk = debt.groupby("code", as_index=False).agg(debt_ratio=("val_num", "mean"))

    merged = stk_vol.merge(debt_stk, on="code", how="inner")
    sec_debt = (
        merged.groupby("sector", as_index=False)
        .agg(
            n_stocks=("code", "nunique"),
            sec_debt_ratio=("debt_ratio", "mean"),
            sec_ann_vol=("ann_vol_all", "mean"),
            sec_vol_crash=("vol_crash", "mean"),
            sec_vol_normal=("vol_normal", "mean"),
            sec_crash_drop=("mean_ret_crash", "mean"),
        )
        .sort_values("sec_debt_ratio", ascending=False)
        .reset_index(drop=True)
    )

    corr_summary = pd.DataFrame([{
        "corr_debt_vs_all_vol": float(sec_debt["sec_debt_ratio"].corr(sec_debt["sec_ann_vol"])) if len(sec_debt) > 2 else np.nan,
        "corr_debt_vs_crash_vol": float(sec_debt["sec_debt_ratio"].corr(sec_debt["sec_vol_crash"])) if len(sec_debt) > 2 else np.nan,
        "corr_debt_vs_normal_vol": float(sec_debt["sec_debt_ratio"].corr(sec_debt["sec_vol_normal"])) if len(sec_debt) > 2 else np.nan,
        "corr_debt_vs_crash_drop": float(sec_debt["sec_debt_ratio"].corr(sec_debt["sec_crash_drop"])) if len(sec_debt) > 2 else np.nan,
    }])
    return {"sector_debt_vol": sec_debt, "corr_summary": corr_summary}


def analyze_q5_low_pbr(price: pd.DataFrame, fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """5. PBR < 1 종목은 2년 반 뒤 어떻게 됐는가 (업종 집중 / ROE 양수 vs 음수 / 저PBR 해소 여부)"""
    stk_ret = (
        price.groupby(["code", "name", "sector"], as_index=False)
        .agg(start_close=("close", "first"), end_close=("close", "last"), n_days=("close", "count"))
    )
    stk_ret = stk_ret[stk_ret["n_days"] >= 20].copy()
    stk_ret["cum_ret"] = stk_ret["end_close"] / stk_ret["start_close"] - 1.0

    # 시작 시점(초기) 및 종료 시점(최근) PBR(배), ROE(지배주주) 추출
    pbr = fin[(fin["metric"].isin(["PBR(배)", "PBR"])) & (~fin["is_est_bool"]) & fin["val_num"].notna()].sort_values(["code", "date", "period"])
    roe = fin[(fin["metric"].str.startswith("ROE")) & (~fin["is_est_bool"]) & fin["val_num"].notna()].sort_values(["code", "date", "period"])

    pbr_se = pbr.groupby("code", as_index=False).agg(pbr_start=("val_num", "first"), pbr_end=("val_num", "last"))
    roe_se = roe.groupby("code", as_index=False).agg(roe_start=("val_num", "first"), roe_mean=("val_num", "mean"))

    df = stk_ret.merge(pbr_se, on="code", how="inner").merge(roe_se, on="code", how="left")
    df["is_low_pbr_start"] = df["pbr_start"] < 1.0
    low_df = df[df["is_low_pbr_start"]].copy()

    if low_df.empty:
        return {"low_pbr_stocks": low_df}

    low_df["roe_positive"] = low_df["roe_start"] > 0
    low_df["pbr_resolved"] = low_df["pbr_end"] >= 1.0

    # (a) 저PBR 종목이 몰린 업종
    sec_conc = (
        df.groupby("sector", as_index=False)
        .agg(
            total_stocks=("code", "count"),
            low_pbr_count=("is_low_pbr_start", "sum"),
            low_pbr_ratio=("is_low_pbr_start", "mean"),
        )
        .sort_values(["low_pbr_count", "low_pbr_ratio"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # (b) ROE 양수 vs 음수 이후 수익률 차이
    roe_comp = (
        low_df.dropna(subset=["roe_start"])
        .groupby("roe_positive", as_index=False)
        .agg(
            n_stocks=("code", "count"),
            mean_cum_ret=("cum_ret", "mean"),
            median_cum_ret=("cum_ret", "median"),
            mean_pbr_start=("pbr_start", "mean"),
            mean_pbr_end=("pbr_end", "mean"),
        )
    )

    # (c) 저PBR 해소(pbr_end >= 1.0) vs 미해소(pbr_end < 1.0) 비교
    res_comp = (
        low_df.groupby("pbr_resolved", as_index=False)
        .agg(
            n_stocks=("code", "count"),
            mean_cum_ret=("cum_ret", "mean"),
            median_cum_ret=("cum_ret", "median"),
            mean_roe_start=("roe_start", "mean"),
            mean_pbr_start=("pbr_start", "mean"),
            mean_pbr_end=("pbr_end", "mean"),
        )
    )

    return {
        "low_pbr_stocks": low_df,
        "sector_concentration": sec_conc,
        "roe_pos_vs_neg": roe_comp,
        "resolved_vs_unresolved": res_comp,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    price, fin = load_data()
    print(f"시세 데이터: {price['date'].min().date()} ~ {price['date'].max().date()} ({price['code'].nunique()}종목)")
    print(f"재무 데이터: {fin['date'].min().date()} ~ {fin['date'].max().date()} ({fin['code'].nunique()}종목, {len(fin)}행)")

    # 1. 영업이익률 vs 수익률
    q1 = analyze_q1_opm_vs_return(price, fin)
    print("\n=== [1-A] 업종별 영업이익률 vs 누적 수익률 (상위 10개 업종) ===")
    print(q1["sector_opm_ret"].head(10).to_string(index=False))
    print("\n=== [1-B] 영업이익률-수익률 상관계수 요약 ===")
    print(q1["summary_corr"].to_string(index=False))
    print("\n=== [1-C] 원래 고이익률 업종 vs 이익률 상승 업종 2x2 수익률 ===")
    print(q1["quadrant_2x2"].to_string(index=False))

    # 2. 실적 발표 전후 주가 반응
    q2 = analyze_q2_earnings_event(price, fin)
    print("\n=== [2] 실적 표 전환일 전후 10일 누적 수익률 (Beat vs Miss) ===")
    if not q2["event_window"].empty:
        print(q2["event_window"].to_string(index=False))
    else:
        print("추정치->확정실적 전환 이벤트가 부족합니다.")

    # 3. 추정치 오차 및 수정 경로
    q3 = analyze_q3_estimate_errors(fin)
    print("\n=== [3-A] 업종별 영업이익 추정 오차 (상위 10개 업종) ===")
    if not q3["sector_error"].empty:
        print(q3["sector_error"].head(10).to_string(index=False))
    print("\n=== [3-B] 발표 전 개월수(months_before)별 추정치 편향 수정 경로 ===")
    if not q3["revision_path"].empty:
        print(q3["revision_path"].to_string(index=False))

    # 4. 부채비율과 변동성
    q4 = analyze_q4_debt_vs_volatility(price, fin)
    print("\n=== [4-A] 부채비율과 변동성 상관계수 (전체 vs 급락 구간 vs 보통 구간) ===")
    print(q4["corr_summary"].to_string(index=False))
    print("\n=== [4-B] 부채비율 상위 10개 업종 변동성 비교 ===")
    print(q4["sector_debt_vol"].head(10).to_string(index=False))

    # 5. 저PBR 종목 2.5년 후 결과
    q5 = analyze_q5_low_pbr(price, fin)
    print("\n=== [5-A] 저PBR(PBR < 1) 집중 업종 상위 10개 ===")
    if "sector_concentration" in q5:
        print(q5["sector_concentration"].head(10).to_string(index=False))
    print("\n=== [5-B] 저PBR 종목 중 ROE 양수 vs 음수 수익률 차이 ===")
    if "roe_pos_vs_neg" in q5:
        print(q5["roe_pos_vs_neg"].to_string(index=False))
    print("\n=== [5-C] 저PBR 해소 종목 vs 미해소 종목 수익률 차이 ===")
    if "resolved_vs_unresolved" in q5:
        print(q5["resolved_vs_unresolved"].to_string(index=False))

    # CSV 저장
    for name, df in [
        ("q1_sector_opm_ret", q1["sector_opm_ret"]),
        ("q1_within_sector", q1["within_sector"]),
        ("q1_quadrant_2x2", q1["quadrant_2x2"]),
        ("q2_events", q2["events"]),
        ("q2_event_window", q2["event_window"]),
        ("q3_sector_error", q3["sector_error"]),
        ("q3_revision_path", q3["revision_path"]),
        ("q4_sector_debt_vol", q4["sector_debt_vol"]),
        ("q4_corr_summary", q4["corr_summary"]),
        ("q5_low_pbr_stocks", q5.get("low_pbr_stocks", pd.DataFrame())),
        ("q5_sector_conc", q5.get("sector_concentration", pd.DataFrame())),
        ("q5_roe_pos_vs_neg", q5.get("roe_pos_vs_neg", pd.DataFrame())),
        ("q5_resolved_vs_unresolved", q5.get("resolved_vs_unresolved", pd.DataFrame())),
    ]:
        if not df.empty:
            df.to_csv(f"{OUT_DIR}/{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n모든 결과 CSV 저장 완료: {OUT_DIR}/")


if __name__ == "__main__":
    main()
