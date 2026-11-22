import argparse
import glob
import os
import numpy as np
import pandas as pd


def load_archive_df(parquet_dir: str) -> pd.DataFrame:
    if os.path.exists("stock_archive.py"):
        try:
            import stock_archive
            if hasattr(stock_archive, "load_all"):
                df = stock_archive.load_all()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        except Exception as e:
            print(f"[안내] stock_archive.load_all() 직접 호출 생략 ({e}), parquet 파일로 읽습니다.")

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"'{parquet_dir}' 경로에 .parquet 파일이 없습니다. 먼저 --step 4를 실행하세요.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    candidates = {
        "date": ["date", "날짜", "dt", "trd_dd"],
        "code": ["code", "ticker", "종목코드", "isu_srt_cd"],
        "name": ["name", "종목명", "isu_abbrv"],
        "sector": ["sector", "업종", "업종명", "industry", "ind_nm"],
        "market_cap": ["market_cap", "시가총액", "mktcap", "marcap", "mktcap_100mn"],
        "close": ["close", "종가", "price", "tdd_clsprc"],
        "volume": ["volume", "거래량", "acc_trdvol"],
    }
    lower_cols = {c.lower(): c for c in df.columns}
    for std_name, opts in candidates.items():
        if std_name in df.columns:
            continue
        for opt in opts:
            if opt.lower() in lower_cols:
                col_map[lower_cols[opt.lower()]] = std_name
                break

    df = df.rename(columns=col_map).copy()
    required = ["date", "code", "sector", "market_cap", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing} / 현재 컬럼: {list(df.columns)}")

    if "name" not in df.columns:
        df["name"] = df["code"].astype(str)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["sector"] = df["sector"].fillna("미분류").astype(str).str.strip()

    for num_col in ["market_cap", "close", "volume"]:
        df[num_col] = pd.to_numeric(
            df[num_col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    df = df.dropna(subset=["date", "code", "close"]).sort_values(["code", "date"]).reset_index(drop=True)
    return df


def analyze_q1_market_cap_rank(df: pd.DataFrame, out_dir: str):
    valid = df.dropna(subset=["market_cap"]).copy()
    dates = sorted(valid["date"].unique())
    start_dt, end_dt = dates[0], dates[-1]

    s_df = valid[valid["date"] == start_dt].copy()
    e_df = valid[valid["date"] == end_dt].copy()

    s_df["rank_start"] = s_df["market_cap"].rank(ascending=False, method="min")
    e_df["rank_end"] = e_df["market_cap"].rank(ascending=False, method="min")

    top30_start = set(s_df[s_df["rank_start"] <= 30]["code"])
    top30_end = set(e_df[e_df["rank_end"] <= 30]["code"])

    retained_codes = top30_start & top30_end
    new_codes = top30_end - top30_start

    new_entries = (
        e_df[e_df["code"].isin(new_codes)][["code", "name", "sector", "rank_end", "market_cap"]]
        .sort_values("rank_end")
    )

    merged = pd.merge(
        s_df[["code", "name", "sector", "rank_start"]],
        e_df[["code", "sector", "rank_end"]],
        on="code",
        how="inner",
        suffixes=("_start", "_end"),
    )
    merged["sector_final"] = merged["sector_end"].fillna(merged["sector_start"])
    merged["abs_rank_shift"] = (merged["rank_end"] - merged["rank_start"]).abs()
    merged["signed_rank_shift"] = merged["rank_start"] - merged["rank_end"]

    sector_shift = (
        merged.groupby("sector_final", as_index=False)
        .agg(
            n_stocks=("code", "count"),
            mean_abs_shift=("abs_rank_shift", "mean"),
            median_abs_shift=("abs_rank_shift", "median"),
            mean_signed_shift=("signed_rank_shift", "mean"),
        )
        .sort_values("mean_abs_shift", ascending=False)
    )

    print("\n================ [1. 시가총액 순위 변화] ================")
    print(f"기간: {start_dt.date()} ~ {end_dt.date()}")
    print(f"초기 상위 30위 중 최종 상위 30위 유지 종목 수: {len(retained_codes)} / 30")
    print(f"최종 상위 30위 새로 진입 종목 수: {len(new_entries)}")
    print("\n[신규 진입 종목 및 업종]")
    print(new_entries.to_string(index=False))
    print("\n[업종별 순위 이동 크기 상위 10개]")
    print(sector_shift.head(10).to_string(index=False))

    new_entries.to_csv(os.path.join(out_dir, "q1_top30_new_entries.csv"), index=False, encoding="utf-8-sig")
    sector_shift.to_csv(os.path.join(out_dir, "q1_sector_rank_shift.csv"), index=False, encoding="utf-8-sig")


def analyze_q2_sector_dispersion(df: pd.DataFrame, out_dir: str):
    sub = df.sort_values(["code", "date"]).copy()
    sub["ret"] = sub.groupby("code")["close"].pct_change()
    sub = sub.dropna(subset=["ret"])

    sub["w_ret"] = sub["ret"] * sub["market_cap"].fillna(0)
    sec_daily = (
        sub.groupby(["date", "sector"], as_index=False)
        .agg(w_ret_sum=("w_ret", "sum"), mcap_sum=("market_cap", "sum"), eq_ret=("ret", "mean"))
    )
    sec_daily["sec_ret"] = np.where(
        sec_daily["mcap_sum"] > 0,
        sec_daily["w_ret_sum"] / sec_daily["mcap_sum"],
        sec_daily["eq_ret"],
    )

    piv = sec_daily.pivot(index="date", columns="sector", values="sec_ret").sort_index()
    cum_ret = (1.0 + piv.fillna(0.0)).cumprod() - 1.0

    daily_disp = piv.var(axis=1, ddof=1).rename("cross_sectional_var")
    monthly_disp = daily_disp.resample("ME").agg(["mean", "median", "max"]).reset_index()

    final_cum = cum_ret.iloc[-1].dropna()
    mean_final_cum = final_cum.mean()
    contrib_df = pd.DataFrame({
        "sector": final_cum.index,
        "cum_return": final_cum.values,
        "diff_from_mean": (final_cum - mean_final_cum).values,
        "abs_diff_from_mean": (final_cum - mean_final_cum).abs().values,
    }).sort_values("abs_diff_from_mean", ascending=False)

    mid_idx = len(daily_disp) // 2
    first_half_var = daily_disp.iloc[:mid_idx].mean()
    second_half_var = daily_disp.iloc[mid_idx:].mean()

    print("\n================ [2. 업종 간 수익률 차이] ================")
    print(f"전반기 평균 업종 간 분산: {first_half_var:.6f} -> 후반기 평균: {second_half_var:.6f}")
    print("2.5년 누적 수익률 기준 업종 평균과 괴리가 큰 상위 8개 업종:")
    print(contrib_df.head(8).to_string(index=False))

    cum_ret.to_csv(os.path.join(out_dir, "q2_sector_cum_return.csv"), encoding="utf-8-sig")
    monthly_disp.to_csv(os.path.join(out_dir, "q2_monthly_dispersion.csv"), index=False, encoding="utf-8-sig")
    contrib_df.to_csv(os.path.join(out_dir, "q2_sector_dispersion_drivers.csv"), index=False, encoding="utf-8-sig")


def analyze_q3_sector_comovement(df: pd.DataFrame, out_dir: str):
    sub = df.sort_values(["code", "date"]).copy()
    sub["ret"] = sub.groupby("code")["close"].pct_change()
    sub = sub.dropna(subset=["ret"])

    mkt_daily = (
        sub.groupby("date", as_index=False)
        .apply(
            lambda g: pd.Series({
                "mkt_ret": (
                    (g["ret"] * g["market_cap"]).sum() / g["market_cap"].sum()
                    if g["market_cap"].sum() > 0 else g["ret"].mean()
                )
            }),
            include_groups=False,
        )
        .set_index("date")["mkt_ret"]
        .sort_index()
    )

    sec_daily = (
        sub.groupby(["date", "sector"], as_index=False)["ret"]
        .mean()
        .pivot(index="date", columns="sector", values="ret")
        .sort_index()
    )

    corr_mat = sec_daily.corr()
    upper_vals = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)).stack()
    overall_mean_corr = upper_vals.mean()

    rolling_corrs = []
    dates = sec_daily.index
    win = 60
    for i in range(win - 1, len(dates)):
        w_df = sec_daily.iloc[i - win + 1 : i + 1]
        c = w_df.corr()
        vals = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
        rolling_corrs.append({"date": dates[i], "rolling_60d_mean_corr": vals.mean()})
    roll_df = pd.DataFrame(rolling_corrs)

    q10 = mkt_daily.quantile(0.10)
    shock_dates = mkt_daily[mkt_daily <= q10].index.intersection(sec_daily.index)
    normal_dates = mkt_daily[mkt_daily > q10].index.intersection(sec_daily.index)

    def mean_pair_corr(d_idx):
        if len(d_idx) < 5:
            return np.nan
        c = sec_daily.loc[d_idx].corr()
        v = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
        return v.mean()

    shock_corr = mean_pair_corr(shock_dates)
    normal_corr = mean_pair_corr(normal_dates)

    print("\n================ [3. 업종 간 동조화(상관)] ================")
    print(f"전체 기간 평균 업종 쌍별 일별 상관계수: {overall_mean_corr:.4f}")
    print(f"시장 하락 충격일(하위 10%, {len(shock_dates)}일) 평균 상관: {shock_corr:.4f}")
    print(f"보통일({len(normal_dates)}일) 평균 상관: {normal_corr:.4f}")

    corr_mat.to_csv(os.path.join(out_dir, "q3_sector_corr_matrix.csv"), encoding="utf-8-sig")
    roll_df.to_csv(os.path.join(out_dir, "q3_rolling_60d_corr.csv"), index=False, encoding="utf-8-sig")


def analyze_q4_volume_lead_lag(df: pd.DataFrame, out_dir: str):
    sub = df.sort_values(["code", "date"]).copy()
    g = sub.groupby("code")

    sub["ret_1d"] = g["close"].pct_change()
    for h in [5, 10, 20]:
        sub[f"fwd_ret_{h}d"] = g["close"].shift(-h) / sub["close"] - 1.0

    prev20_mean = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    prev20_med = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    prev60_mean = g["volume"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).mean())

    definitions = {
        "Def1_20dMean_2.0x": sub["volume"] >= 2.0 * prev20_mean,
        "Def2_20dMean_3.0x": sub["volume"] >= 3.0 * prev20_mean,
        "Def3_20dMed_2.5x": sub["volume"] >= 2.5 * prev20_med,
        "Def4_60dMean_2.0x": sub["volume"] >= 2.0 * prev60_mean,
    }

    rows = []
    base_means = {h: sub[f"fwd_ret_{h}d"].mean() for h in [5, 10, 20]}

    for name, mask in definitions.items():
        m = mask.fillna(False)
        row = {"definition": name, "n_events": int(m.sum())}
        for h in [5, 10, 20]:
            ev_mean = sub.loc[m, f"fwd_ret_{h}d"].mean()
            ev_med = sub.loc[m, f"fwd_ret_{h}d"].median()
            row[f"fwd_{h}d_mean"] = ev_mean
            row[f"fwd_{h}d_excess_vs_uncond"] = ev_mean - base_means[h]
            row[f"fwd_{h}d_median"] = ev_med
        rows.append(row)

    res_df = pd.DataFrame(rows)
    print("\n================ [4. 거래량 급증 후 5·10·20일 수익률] ================")
    print(f"무조건부 평균 forward 수익률 -> 5일: {base_means[5]:.4%}, 10일: {base_means[10]:.4%}, 20일: {base_means[20]:.4%}")
    print(res_df.to_string(index=False))

    res_df.to_csv(os.path.join(out_dir, "q4_volume_surge_sensitivity.csv"), index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="2.5년치 주식 아카이브 1~4번 문항 분석")
    parser.add_argument("--parquet-dir", default="parquet", help="날짜별 parquet 폴더 경로")
    parser.add_argument("--out-dir", default="results_q1_q4", help="결과 CSV 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_df = load_archive_df(args.parquet_dir)
    df = normalize_columns(raw_df)

    analyze_q1_market_cap_rank(df, args.out_dir)
    analyze_q2_sector_dispersion(df, args.out_dir)
    analyze_q3_sector_comovement(df, args.out_dir)
    analyze_q4_volume_lead_lag(df, args.out_dir)
    print(f"\n완료: 세부 CSV 파일이 '{args.out_dir}/' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
