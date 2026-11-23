import os
import glob
import numpy as np
import pandas as pd

PRICE_DIR = "parquet"
FIN_DIR = "fin_parquet"
OUT_DIR = "results_all_8q"
os.makedirs(OUT_DIR, exist_ok=True)


def load_prices():
    files = sorted(glob.glob(os.path.join(PRICE_DIR, "*.parquet")))
    if not files:
        raise RuntimeError("parquet/ 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    return df


def calc_q1_q4(price_df):
    dates = sorted(price_df["date"].unique())
    d_start, d_end = dates[0], dates[-1]
    mid_idx = len(dates) // 2
    d_mid = dates[mid_idx]

    # 1. 시가총액 순위 변화
    p_start = price_df[price_df["date"] == d_start].copy()
    p_end = price_df[price_df["date"] == d_end].copy()
    p_start["rank_start"] = p_start["mktcap_100mn"].rank(ascending=False, method="min")
    p_end["rank_end"] = p_end["mktcap_100mn"].rank(ascending=False, method="min")

    top30_start = set(p_start.nsmallest(30, "rank_start")["code"])
    top30_end_df = p_end.nsmallest(30, "rank_end")
    top30_end = set(top30_end_df["code"])
    retained = len(top30_start & top30_end)
    new_entrants = top30_end_df[~top30_end_df["code"].isin(top30_start)][
        ["code", "name", "sector", "rank_end", "mktcap_100mn"]
    ]
    new_entrants.to_csv(os.path.join(OUT_DIR, "q1_new_top30.csv"), index=False, encoding="utf-8-sig")

    rank_cmp = p_end[["code", "name", "sector", "rank_end", "mktcap_100mn"]].merge(
        p_start[["code", "rank_start"]], on="code", how="inner"
    )
    rank_cmp["abs_shift"] = (rank_cmp["rank_end"] - rank_cmp["rank_start"]).abs()
    rank_cmp["signed_shift"] = rank_cmp["rank_start"] - rank_cmp["rank_end"]  # +면 순위 상승
    q1_sec = rank_cmp.groupby("sector", as_index=False).agg(
        n_stocks=("code", "count"),
        mean_abs_shift=("abs_shift", "mean"),
        median_abs_shift=("abs_shift", "median"),
        mean_signed_shift=("signed_shift", "mean"),
    ).sort_values("mean_abs_shift", ascending=False)
    q1_sec.to_csv(os.path.join(OUT_DIR, "q1_sector_rank_shift.csv"), index=False, encoding="utf-8-sig")

    # 2. 업종 간 수익률 차이 (누적 수익률, 분산의 시간 변화, 상위 괴리 업종)
    sec_daily = price_df.groupby(["date", "sector"], as_index=False)["ret"].mean()
    sec_piv = sec_daily.pivot(index="date", columns="sector", values="ret")
    disp_series = sec_piv.var(axis=1, skipna=True)
    disp_df = disp_series.reset_index(name="cross_sec_var")
    disp_df["half"] = np.where(disp_df["date"] <= d_mid, "FirstHalf", "SecondHalf")
    disp_df.to_csv(os.path.join(OUT_DIR, "q2_daily_dispersion.csv"), index=False, encoding="utf-8-sig")

    sec_cum = ((1.0 + sec_piv.fillna(0.0)).prod(axis=0) - 1.0).reset_index(name="cum_return")
    mkt_mean_cum = sec_cum["cum_return"].mean()
    sec_cum["diff_from_mean"] = sec_cum["cum_return"] - mkt_mean_cum
    sec_cum["abs_diff"] = sec_cum["diff_from_mean"].abs()
    sec_cum = sec_cum.sort_values("abs_diff", ascending=False)
    sec_cum.to_csv(os.path.join(OUT_DIR, "q2_sector_cum_return_gap.csv"), index=False, encoding="utf-8-sig")

    # 3. 업종 간 동조화 (일별 상관, 시간 변화, 충격일 vs 보통일)
    corr_all = sec_piv.corr()
    corr_all.to_csv(os.path.join(OUT_DIR, "q3_sector_corr_all.csv"), encoding="utf-8-sig")

    roll_corr = []
    w = min(20, max(3, len(sec_piv) // 2))
    for i in range(w - 1, len(sec_piv)):
        sub = sec_piv.iloc[i - w + 1 : i + 1].corr().values
        tri = sub[np.triu_indices_from(sub, k=1)]
        roll_corr.append({"date": sec_piv.index[i], "win": w, "mean_pair_corr": np.nanmean(tri)})
    pd.DataFrame(roll_corr).to_csv(os.path.join(OUT_DIR, "q3_rolling_corr.csv"), index=False, encoding="utf-8-sig")

    mkt_ret = price_df.groupby("date")["ret"].mean()
    shock_cut = mkt_ret.quantile(0.10)
    shock_dates = mkt_ret[mkt_ret <= shock_cut].index
    normal_dates = mkt_ret[mkt_ret > shock_cut].index

    def avg_pair_corr(idx_list):
        if len(idx_list) < 2:
            return np.nan
        m = sec_piv.loc[idx_list].corr().values
        return float(np.nanmean(m[np.triu_indices_from(m, k=1)]))

    q3_regime = pd.DataFrame([{
        "all_days_corr": avg_pair_corr(sec_piv.index),
        "shock_bottom10_corr": avg_pair_corr(shock_dates),
        "normal_days_corr": avg_pair_corr(normal_dates),
        "n_shock_days": len(shock_dates),
        "n_normal_days": len(normal_dates),
    }])
    q3_regime.to_csv(os.path.join(OUT_DIR, "q3_shock_vs_normal_corr.csv"), index=False, encoding="utf-8-sig")

    # 4. 거래량 급증 정의별 5·10·20일 수익률 민감도
    df_v = price_df.copy()
    df_v["vol_ma20"] = df_v.groupby("code")["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df_v["vol_med20"] = df_v.groupby("code")["volume"].transform(lambda s: s.rolling(20, min_periods=5).median())
    df_v["vol_ma60"] = df_v.groupby("code")["volume"].transform(lambda s: s.rolling(60, min_periods=10).mean())

    for h in [5, 10, 20]:
        df_v[f"fwd_{h}d"] = df_v.groupby("code")["close"].transform(lambda s: s.shift(-h) / s - 1.0)

    defs = {
        "Def1_MA20_2.0x": df_v["volume"] >= 2.0 * df_v["vol_ma20"],
        "Def2_MA20_3.0x": df_v["volume"] >= 3.0 * df_v["vol_ma20"],
        "Def3_Med20_2.5x": df_v["volume"] >= 2.5 * df_v["vol_med20"],
        "Def4_MA60_2.0x": df_v["volume"] >= 2.0 * df_v["vol_ma60"],
    }
    uncond = {h: df_v[f"fwd_{h}d"].mean() for h in [5, 10, 20]}
    q4_rows = []
    for name, cond in defs.items():
        sub = df_v[cond]
        row = {"definition": name, "n_events": int(cond.sum())}
        for h in [5, 10, 20]:
            m_val = sub[f"fwd_{h}d"].mean()
            row[f"fwd_{h}d_mean"] = m_val
            row[f"fwd_{h}d_excess"] = m_val - uncond[h]
        q4_rows.append(row)
    pd.DataFrame(q4_rows).to_csv(os.path.join(OUT_DIR, "q4_volume_surge_all_defs.csv"), index=False, encoding="utf-8-sig")

    summary_lines = [
        f"기간: {d_start.date()} ~ {d_end.date()} (거래일 {len(dates)}일)",
        f"Q1 상위30 유지: {retained}/30, 신규진입: {len(new_entrants)}개",
        f"Q2 업종간분산 전반기평균: {disp_df[disp_df['half']=='FirstHalf']['cross_sec_var'].mean():.6f}, 후반기평균: {disp_df[disp_df['half']=='SecondHalf']['cross_sec_var'].mean():.6f}",
        f"Q3 쌍별상관 전체: {q3_regime.loc[0, 'all_days_corr']:.4f}, 하락충격일: {q3_regime.loc[0, 'shock_bottom10_corr']:.4f}, 보통일: {q3_regime.loc[0, 'normal_days_corr']:.4f}",
    ]
    return summary_lines, rank_cmp


def calc_q5_q8(price_df, rank_cmp):
    files = sorted(glob.glob(os.path.join(FIN_DIR, "*.parquet")))
    if not files:
        return ["Q5~Q8: fin_parquet/ 데이터가 아직 없어 생략됨(parse_financials.py 실행 필요)."]

    fin = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fin["date"] = pd.to_datetime(fin["date"])

    # 종목별 초기/최종 누적수익률 및 변동성
    first_c = price_df.groupby("code")["close"].transform("first")
    last_c = price_df.groupby("code")["close"].transform("last")
    price_df["cum_ret"] = last_c / first_c - 1.0

    st = price_df.groupby(["code", "name", "sector"], as_index=False).agg(
        cum_ret=("cum_ret", "first"),
        vol=("ret", "std"),
    )

    # 최근 관측치 기준 metric별 피벗
    latest_f = fin.sort_values("date").groupby(["code", "metric"], as_index=False).last()
    piv_val = latest_f.pivot(index="code", columns="metric", values="value")

    opm_col = next((c for c in piv_val.columns if "영업이익률" in c), None)
    debt_col = next((c for c in piv_val.columns if "부채비율" in c), None)
    pbr_col = next((c for c in piv_val.columns if "PBR" in c), None)
    roe_col = next((c for c in piv_val.columns if "ROE" in c), None)

    # 5. 영업이익률과 주가 수익률 (업종별 / 업종 내 / 개선 vs 고수준)
    m5 = st.merge(piv_val, on="code", how="left")
    if opm_col:
        q5_sec = m5.groupby("sector", as_index=False).agg(
            n_stocks=("code", "count"),
            mean_opm=(opm_col, "mean"),
            mean_cum_ret=("cum_ret", "mean"),
            within_corr=(opm_col, lambda s: s.corr(m5.loc[s.index, "cum_ret"])),
        )
        q5_sec.to_csv(os.path.join(OUT_DIR, "q5_opm_vs_return_sector.csv"), index=False, encoding="utf-8-sig")

        # 초기 vs 후기 영업이익률 변화 비교
        early_f = fin.sort_values("date").groupby(["code", "metric"], as_index=False).first().pivot(
            index="code", columns="metric", values="value"
        )
        if opm_col in early_f.columns:
            m5_ch = m5.merge(early_f[[opm_col]].rename(columns={opm_col: "opm_early"}), on="code", how="left")
            m5_ch["opm_chg"] = m5_ch[opm_col] - m5_ch["opm_early"]
            m5_ch["high_init"] = m5_ch["opm_early"] >= m5_ch["opm_early"].median()
            m5_ch["improved"] = m5_ch["opm_chg"] > 0
            q5_2x2 = m5_ch.groupby(["high_init", "improved"], as_index=False)["cum_ret"].agg(["count", "mean", "median"])
            q5_2x2.to_csv(os.path.join(OUT_DIR, "q5_opm_level_vs_change.csv"), index=False, encoding="utf-8-sig")

    # 6. 실적 발표일(표 변경일) 탐지 & 전후 10일 누적수익률 (Beat vs Miss)
    f_chg = fin.sort_values(["code", "metric", "period", "date"]).copy()
    f_chg["prev_val"] = f_chg.groupby(["code", "metric", "period"])["value"].shift(1)
    f_chg["prev_est"] = f_chg.groupby(["code", "metric", "period"])["is_est"].shift(1)
    changed = f_chg[
        (f_chg["prev_val"].notna()) & (f_chg["value"].notna()) & (f_chg["value"] != f_chg["prev_val"])
    ].copy()
    changed["surprise"] = changed["value"] - changed["prev_val"]
    changed["beat_miss"] = np.where(changed["surprise"] > 0, "Beat", np.where(changed["surprise"] < 0, "Miss", "Meet"))
    changed.to_csv(os.path.join(OUT_DIR, "q6_table_change_events.csv"), index=False, encoding="utf-8-sig")

    ret_piv = price_df.pivot(index="date", columns="code", values="ret")
    t_dates = ret_piv.index.sort_values()
    d2i = {d: i for i, d in enumerate(t_dates)}
    op_ev = changed[changed["metric"].str.contains("영업이익", na=False) & (~changed["metric"].str.contains("률", na=False))]
    car_rows = []
    for _, r in op_ev.iterrows():
        c, d = r["code"], r["date"]
        if c not in ret_piv.columns or d not in d2i:
            continue
        idx = d2i[d]
        if idx < 10 or idx + 10 >= len(t_dates):
            continue
        w_rets = ret_piv[c].iloc[idx - 10 : idx + 11].fillna(0.0).values
        car = np.cumprod(1.0 + w_rets) - 1.0
        rec = {"code": c, "date": d, "beat_miss": r["beat_miss"]}
        for k, rel in enumerate(range(-10, 11)):
            rec[f"car_t{rel}"] = car[k]
        car_rows.append(rec)
    if car_rows:
        pd.DataFrame(car_rows).groupby("beat_miss").mean(numeric_only=True).to_csv(
            os.path.join(OUT_DIR, "q6_event_car_10d.csv"), encoding="utf-8-sig"
        )

    # 7. 추정치 오차 분포, 발표 전 수정 추이, 쏠림 방향
    est_hist = fin[fin["is_est"] == True].copy()
    if not est_hist.empty and not changed.empty:
        # 같은 (code, period, metric)에서 발표 전 추정치 변화 기록
        est_rev = est_hist.sort_values(["code", "metric", "period", "date"]).copy()
        est_rev["prev_est_val"] = est_rev.groupby(["code", "metric", "period"])["value"].shift(1)
        est_rev["est_revision"] = est_rev["value"] - est_rev["prev_est_val"]
        est_rev.dropna(subset=["est_revision"]).to_csv(
            os.path.join(OUT_DIR, "q7_pre_announcement_revisions.csv"), index=False, encoding="utf-8-sig"
        )

        sec_m = price_df[["code", "sector"]].drop_duplicates("code")
        ch_sec = changed.merge(sec_m, on="code", how="left")
        q7_err = ch_sec.groupby(["sector", "metric"], as_index=False).agg(
            n_events=("surprise", "count"),
            mae=("surprise", lambda s: s.abs().mean()),
            bias_mean=("surprise", "mean"),
            pos_share=("surprise", lambda s: (s > 0).mean()),
        )
        q7_err.to_csv(os.path.join(OUT_DIR, "q7_forecast_error_bias.csv"), index=False, encoding="utf-8-sig")

    # 8. 부채비율과 변동성 (급락 vs 보통) & 저PBR(<1) 2.5년 후 성과 (업종, ROE +/-, PBR 해소 여부)
    mkt_r = price_df.groupby("date")["ret"].mean()
    crash_d = mkt_r[mkt_r <= mkt_r.quantile(0.05)].index
    price_df["is_crash"] = np.where(price_df["date"].isin(crash_d), "Crash", "Normal")
    vol_c = price_df.groupby(["code", "is_crash"], as_index=False)["ret"].std().pivot(
        index="code", columns="is_crash", values="ret"
    ).reset_index()
    m8 = st.merge(piv_val, on="code", how="left").merge(vol_c, on="code", how="left")
    if debt_col:
        q8_debt = m8.groupby("sector", as_index=False).agg(
            mean_debt=(debt_col, "mean"),
            vol_normal=("Normal", "mean"),
            vol_crash=("Crash", "mean"),
        )
        q8_debt.to_csv(os.path.join(OUT_DIR, "q8_debt_vs_vol.csv"), index=False, encoding="utf-8-sig")

    if pbr_col and roe_col:
        early_f = fin.sort_values("date").groupby(["code", "metric"], as_index=False).first().pivot(
            index="code", columns="metric", values="value"
        )
        if pbr_col in early_f.columns:
            m8_pbr = m8.merge(early_f[[pbr_col]].rename(columns={pbr_col: "pbr_start"}), on="code", how="left")
            low_pbr = m8_pbr[m8_pbr["pbr_start"] < 1.0].copy()
            low_pbr["roe_pos"] = np.where(low_pbr[roe_col] > 0, "ROE_Positive", "ROE_Negative_or_Zero")
            low_pbr["pbr_resolved"] = np.where(low_pbr[pbr_col] >= 1.0, "Resolved_ge1", "Remained_lt1")

            low_pbr.groupby("sector", as_index=False).agg(
                n=("code", "count"),
                mean_cum_ret=("cum_ret", "mean"),
            ).sort_values("n", ascending=False).to_csv(
                os.path.join(OUT_DIR, "q8_low_pbr_by_sector.csv"), index=False, encoding="utf-8-sig"
            )
            low_pbr.groupby("roe_pos", as_index=False)["cum_ret"].agg(["count", "mean", "median"]).to_csv(
                os.path.join(OUT_DIR, "q8_low_pbr_by_roe.csv"), index=False, encoding="utf-8-sig"
            )
            low_pbr.groupby("pbr_resolved", as_index=False)["cum_ret"].agg(["count", "mean", "median"]).to_csv(
                os.path.join(OUT_DIR, "q8_low_pbr_resolved.csv"), index=False, encoding="utf-8-sig"
            )

    return [f"Q5~Q8 완료: 표 변경 감지 {len(changed)}건"]


def main():
    price_df = load_prices()
    s1, rank_cmp = calc_q1_q4(price_df)
    s2 = calc_q5_q8(price_df, rank_cmp)
    out_txt = os.path.join(OUT_DIR, "summary_1_to_8.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in s1 + s2:
            print(line)
            f.write(line + "\n")
    print("요약 저장:", out_txt)


if __name__ == "__main__":
    main()
