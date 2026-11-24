import argparse
import glob
import os
import numpy as np
import pandas as pd

TARGETS = {
    "035420": "NAVER",
    "035720": "Kakao",
}


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    return None


def clean_sector_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["code"] = out["code"].astype(str).str.strip().str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "code"]).sort_values(["code", "date"])

    sec = out["sector"].astype(str).str.strip()
    invalid = (
        out["sector"].isna()
        | (sec == "")
        | (sec == "미분류")
        | sec.str.contains("업종명", na=False)
        | sec.str.contains(r"^\d{1,2}/\d{1,2}$", regex=True, na=False)
    )
    out.loc[invalid, "sector"] = np.nan

    mode_sector = (
        out.dropna(subset=["sector"])
        .groupby("code")["sector"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
    )
    out["sector"] = out["code"].map(mode_sector).fillna("미분류")
    return out


def load_prices_from_dir(price_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(price_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"{price_dir}/ 에 .parquet 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    code_col = pick_col(df, ["code", "ticker", "종목코드"])
    date_col = pick_col(df, ["date", "dt", "날짜", "일자"])
    close_col = pick_col(df, ["close", "adj_close", "종가"])
    mcap_col = pick_col(df, ["market_cap", "mktcap_100mn", "mcap", "시가총액"])
    vol_col = pick_col(df, ["volume", "vol", "거래량", "acc_trdvol"])
    sec_col = pick_col(df, ["sector", "sector_final", "업종", "업종명", "industry", "ind_nm"])
    name_col = pick_col(df, ["name", "종목명"])
    ret_col = pick_col(df, ["ret", "return", "daily_ret", "수익률"])

    if not code_col or not date_col or not close_col:
        raise ValueError(f"필수 컬럼 누락: 현재 컬럼 {list(df.columns)}")

    out = pd.DataFrame({
        "code": df[code_col],
        "date": df[date_col],
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "market_cap": pd.to_numeric(df[mcap_col], errors="coerce") if mcap_col else np.nan,
        "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan,
        "sector": df[sec_col] if sec_col else "미분류",
        "name": df[name_col] if name_col else "",
    })
    out = clean_sector_series(out)
    if ret_col and ret_col in df.columns:
        tmp_ret = pd.DataFrame({
            "code": df[code_col].astype(str).str.strip().str.zfill(6),
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "_ret": pd.to_numeric(df[ret_col], errors="coerce"),
        })
        out = out.merge(tmp_ret.drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
        out["ret"] = out["_ret"].fillna(out.groupby("code")["close"].pct_change())
        out = out.drop(columns=["_ret"])
    else:
        out["ret"] = out.groupby("code")["close"].pct_change()
    return out


def _pivot_metric_edge(sub: pd.DataFrame, metric_col: str, how: str = "last") -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    clean = sub.dropna(subset=["_val"]).sort_values(["code", "_dt"])
    if clean.empty:
        return pd.DataFrame()
    grp = clean.groupby(["code", metric_col], as_index=False)["_val"]
    agg_df = grp.last() if how == "last" else grp.first()
    return agg_df.pivot(index="code", columns=metric_col, values="_val")


def _extract_metric(piv: pd.DataFrame, code_series: pd.Series, candidates: list[str]) -> pd.Series:
    if piv.empty:
        return pd.Series(np.nan, index=code_series.index)
    col = pick_col(piv, candidates)
    if not col:
        return pd.Series(np.nan, index=code_series.index)
    return code_series.map(piv[col])


def load_financials_summary(fin_dir: str, codes_filter: set[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(fin_dir, "*.parquet")))
    res = pd.DataFrame({"code": sorted(codes_filter)})
    if not files:
        return res

    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    code_col = pick_col(f, ["code", "ticker", "종목코드"])
    metric_col = pick_col(f, ["metric", "item", "account", "지표", "항목명"])
    val_col = pick_col(f, ["value", "val", "amount", "값"])
    date_col = pick_col(f, ["date", "dt", "period", "quarter", "날짜", "기준일"])
    est_col = pick_col(f, ["is_est", "estimate_flag", "is_estimate"])

    if not code_col or not metric_col or not val_col:
        return res

    f["code"] = f[code_col].astype(str).str.strip().str.zfill(6)
    f = f[f["code"].isin(codes_filter)].copy()
    if f.empty:
        return res

    f["_dt"] = pd.to_datetime(f[date_col], errors="coerce") if date_col else pd.NaT
    f["_val"] = pd.to_numeric(f[val_col], errors="coerce")

    if est_col:
        est_bool = f[est_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        actual_df = f[~est_bool]
        est_df = f[est_bool]
    else:
        actual_df = f
        est_df = pd.DataFrame(columns=f.columns)

    act_first = _pivot_metric_edge(actual_df, metric_col, "first")
    act_last = _pivot_metric_edge(actual_df, metric_col, "last")
    est_last = _pivot_metric_edge(est_df, metric_col, "last")

    # 5. 실적
    res["rev_last"] = _extract_metric(act_last, res["code"], ["매출액"])
    res["op_last"] = _extract_metric(act_last, res["code"], ["영업이익"])
    res["ni_last"] = _extract_metric(act_last, res["code"], ["당기순이익"])
    res["opm_last"] = _extract_metric(act_last, res["code"], ["영업이익률"])
    res["npm_last"] = _extract_metric(act_last, res["code"], ["순이익률"])
    res["roe_last"] = _extract_metric(act_last, res["code"], ["ROE(지배주주)"])
    res["eps_last"] = _extract_metric(act_last, res["code"], ["EPS(원)"])
    res["bps_last"] = _extract_metric(act_last, res["code"], ["BPS(원)"])

    # 6. 추정치 (is_est == True가 fin_parquet에 없는 경우 NaN 처리)
    res["est_rev_last"] = _extract_metric(est_last, res["code"], ["매출액"])
    res["est_op_last"] = _extract_metric(est_last, res["code"], ["영업이익"])
    res["op_surprise_pct"] = np.where(
        res["est_op_last"].abs() > 0,
        (res["op_last"] - res["est_op_last"]) / res["est_op_last"].abs(),
        np.nan,
    )

    # 7. 부채
    res["debt_first"] = _extract_metric(act_first, res["code"], ["부채비율"])
    res["debt_last"] = _extract_metric(act_last, res["code"], ["부채비율"])
    res["debt_change"] = res["debt_last"] - res["debt_first"]

    # 8. PBR
    res["pbr_first"] = _extract_metric(act_first, res["code"], ["PBR(배)"])
    res["pbr_last"] = _extract_metric(act_last, res["code"], ["PBR(배)"])
    res["per_last"] = _extract_metric(act_last, res["code"], ["PER(배)"])
    res["pbr_change"] = res["pbr_last"] - res["pbr_first"]

    return res


def analyze_target_sector(df: pd.DataFrame, fin_all: pd.DataFrame, target_code: str, target_name: str, out_dir: str):
    t_rows = df[df["code"] == target_code]
    if t_rows.empty:
        print(f"[경고] {target_name} ({target_code}) 데이터가 없습니다.")
        return None

    sector_name = t_rows["sector"].mode().iloc[0]
    peer = df[df["sector"] == sector_name].copy()
    if peer.empty:
        print(f"[경고] {target_name} 업종({sector_name}) 데이터가 비어 있습니다.")
        return None

    first_dt, last_dt = peer["date"].min(), peer["date"].max()

    # 1. 순위
    s_mcap = (
        peer[peer["date"] == first_dt]
        .groupby("code", as_index=False)["market_cap"]
        .last()
        .assign(rank_start=lambda x: x["market_cap"].rank(ascending=False, method="min"))
        .rename(columns={"market_cap": "mcap_start"})
    )
    e_mcap = (
        peer[peer["date"] == last_dt]
        .groupby("code", as_index=False)["market_cap"]
        .last()
        .assign(rank_end=lambda x: x["market_cap"].rank(ascending=False, method="min"))
        .rename(columns={"market_cap": "mcap_end"})
    )
    rank_df = s_mcap.merge(e_mcap, on="code", how="outer")
    rank_df["rank_shift"] = rank_df["rank_end"] - rank_df["rank_start"]

    # 2. 수익률
    ret_df = peer.groupby("code", as_index=False).agg(
        name=("name", lambda s: s.dropna().iloc[-1] if not s.dropna().empty else ""),
        obs_days=("ret", "count"),
        mean_daily_ret=("ret", "mean"),
        vol_daily_ret=("ret", "std"),
        cum_ret=("ret", lambda s: float(np.prod(1.0 + s.dropna().to_numpy()) - 1.0) if s.dropna().size > 0 else np.nan),
    )

    # 3. 동조
    sec_ret = peer.groupby("date")["ret"].mean().rename("sec_ret")
    tgt_ret = df[df["code"] == target_code].groupby("date")["ret"].mean().rename("tgt_ret")
    sec_var = sec_ret.var()

    sync_records = []
    for code, g in peer.groupby("code"):
        s = g.groupby("date")["ret"].mean().rename("r")
        c_sec = pd.concat([s, sec_ret], axis=1).dropna()
        corr_sec = c_sec["r"].corr(c_sec["sec_ret"]) if len(c_sec) >= 5 else np.nan
        beta_sec = (
            c_sec["r"].cov(c_sec["sec_ret"]) / sec_var
            if (len(c_sec) >= 5 and pd.notna(sec_var) and sec_var > 0)
            else np.nan
        )
        c_tgt = pd.concat([s, tgt_ret], axis=1).dropna()
        corr_tgt = c_tgt["r"].corr(c_tgt["tgt_ret"]) if len(c_tgt) >= 5 else np.nan
        sync_records.append({
            "code": code,
            "corr_with_sector": corr_sec,
            "beta_with_sector": beta_sec,
            "corr_with_target": corr_tgt,
        })
    sync_df = pd.DataFrame(sync_records)

    # 4. 거래량
    peer_sorted = peer.sort_values(["code", "date"]).copy()
    prev20 = peer_sorted.groupby("code")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
    )
    peer_sorted["vol_surge_ratio"] = np.where(prev20 > 0, peer_sorted["volume"] / prev20, np.nan)
    vol_df = peer_sorted.groupby("code", as_index=False).agg(
        avg_volume=("volume", "mean"),
        median_volume=("volume", "median"),
        mean_vol_surge_20d=("vol_surge_ratio", "mean"),
        max_vol_surge_20d=("vol_surge_ratio", "max"),
    )

    peer_codes = set(ret_df["code"].unique())
    fin_sub = fin_all[fin_all["code"].isin(peer_codes)] if not fin_all.empty else pd.DataFrame({"code": sorted(peer_codes)})

    merged = (
        ret_df.merge(rank_df, on="code", how="left")
        .merge(sync_df, on="code", how="left")
        .merge(vol_df, on="code", how="left")
        .merge(fin_sub, on="code", how="left")
    )
    merged["sector"] = sector_name
    merged["target_group"] = target_name
    merged["is_target"] = merged["code"] == target_code
    merged = merged.sort_values(["rank_end", "mcap_end"], ascending=[True, False], na_position="last")

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"peer_8metrics_{target_name.lower()}_{target_code}.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n=== [{target_name} ({target_code})] 업종: {sector_name} (동종 종목 수: {len(merged)}) ===")
    print(f"저장 파일: {out_csv}")
    cols_show = [
        "code", "name", "rank_start", "rank_end", "rank_shift",
        "cum_ret", "vol_daily_ret", "corr_with_sector", "beta_with_sector",
        "avg_volume", "mean_vol_surge_20d",
        "rev_last", "op_last", "opm_last", "roe_last",
        "est_op_last", "debt_last", "pbr_first", "pbr_last", "pbr_change",
    ]
    avail_cols = [c for c in cols_show if c in merged.columns]
    print(merged[merged["is_target"]][avail_cols].to_string(index=False))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="parquet", help="주가 파케이 폴더")
    parser.add_argument("--fin-dir", default="fin_parquet", help="재무 파케이 폴더")
    parser.add_argument("--out-dir", default="results_naver_kakao_peer", help="결과 저장 폴더")
    args = parser.parse_args()

    df = load_prices_from_dir(args.parquet_dir)
    all_codes = set(df["code"].unique())
    fin_all = load_financials_summary(args.fin_dir, all_codes)

    print(f"전체 기간: {df['date'].min().date()} ~ {df['date'].max().date()} (총 행: {len(df):,})")

    for code, name in TARGETS.items():
        analyze_target_sector(df, fin_all, code, name, args.out_dir)


if __name__ == "__main__":
    main()
