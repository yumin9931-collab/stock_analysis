import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


SECTOR_MAP = {
    "양방향미디어와서비스": "인터넷/플랫폼",
    "인터넷": "인터넷/플랫폼",
    "인터넷/포털": "인터넷/플랫폼",
    "소프트웨어": "IT/소프트웨어",
    "컴퓨터서비스": "IT/소프트웨어",
    "반도체": "반도체/IT",
    "반도체와반도체장비": "반도체/IT",
    "제약": "바이오/제약",
    "생물공학": "바이오/제약",
    "은행": "금융/지주",
    "증권": "금융/지주",
    "보험": "금융/지주",
    "자동차": "자동차/부품",
    "자동차부품": "자동차/부품",
    "기계": "조선/기계/방산",
    "조선": "조선/기계/방산",
    "방위산업": "조선/기계/방산",
    "화학": "에너지/화학",
    "석유와가스": "에너지/화학",
    "게임엔터테인먼트": "게임/콘텐츠",
}


def setup_korean_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    else:
        plt.rcParams["font.sans-serif"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def clean_sector_series(s: pd.Series) -> pd.Series:
    cleaned = s.fillna("").astype(str).str.strip()
    cleaned = cleaned.apply(lambda x: SECTOR_MAP.get(x, x))

    noise_mask = (
        cleaned.eq("")
        | cleaned.str.lower().isin({"nan", "none", "null", "미분류", "소형업종"})
        | cleaned.str.contains(r"업종명|0\d/\d\d|1\d/\d\d|\d{2}:\d{2}|/|:", regex=True)
        | cleaned.str.startswith("(")
        | (cleaned.str.len() < 2)
    )
    return cleaned.where(~noise_mask, np.nan)


def safe_read_csv(path: str) -> pd.DataFrame:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_code_sector_lookup() -> dict:
    """다른 CSV나 parquet에서 유효한 (code -> sector) 매핑을 수집"""
    lookup = {}
    cand_csvs = [
        "kakao_sector_8metrics.csv",
        "results_spec1_spec2/spec1_naver_kakao_focus.csv",
        "results_spec1_spec2/spec1_sector_comparison.csv",
        "results_all_8q/q5_opm_vs_return_sector.csv",
        "results_all_8q/q8_debt_vs_vol.csv",
    ]
    for path in cand_csvs:
        df = safe_read_csv(path)
        if df.empty or "code" not in df.columns:
            continue
        sec_c = next((c for c in ["sector_clean", "sector_final", "sector"] if c in df.columns), None)
        if not sec_c:
            continue
        s_clean = clean_sector_series(df[sec_c])
        for c, sec in zip(df["code"].astype(str).str.zfill(6), s_clean):
            if pd.notna(sec) and c not in lookup:
                lookup[c] = sec
    return lookup


def compute_from_parquet() -> tuple[pd.DataFrame, pd.DataFrame]:
    p_files = sorted(glob.glob(os.path.expanduser("~/parquet/*.parquet")))
    if not p_files:
        p_files = sorted(glob.glob("parquet/*.parquet"))
    if not p_files:
        return pd.DataFrame(columns=["sector_clean", "shift_val"]), pd.DataFrame(columns=["sector_clean", "cnt"])

    dfs = []
    for fp in p_files:
        try:
            dfs.append(pd.read_parquet(fp))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=["sector_clean", "shift_val"]), pd.DataFrame(columns=["sector_clean", "cnt"])

    raw = pd.concat(dfs, ignore_index=True)
    mcap_col = next((c for c in ["market_cap", "mcap", "시가총액"] if c in raw.columns), None)
    sec_col = next((c for c in ["sector", "raw_sector", "업종"] if c in raw.columns), None)
    if not mcap_col or "date" not in raw.columns or "code" not in raw.columns:
        return pd.DataFrame(columns=["sector_clean", "shift_val"]), pd.DataFrame(columns=["sector_clean", "cnt"])

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code_z"] = raw["code"].astype(str).str.strip().str.zfill(6)
    raw[mcap_col] = pd.to_numeric(raw[mcap_col], errors="coerce")

    valid = raw.dropna(subset=["date", "code_z", mcap_col]).copy()
    dates = sorted(valid["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["sector_clean", "shift_val"]), pd.DataFrame(columns=["sector_clean", "cnt"])

    lookup = build_code_sector_lookup()
    if sec_col and sec_col in valid.columns:
        sec_clean_all = clean_sector_series(valid[sec_col])
        for c, sec in zip(valid["code_z"], sec_clean_all):
            if pd.notna(sec) and c not in lookup:
                lookup[c] = sec

    valid["sector_clean"] = valid["code_z"].map(lookup)
    # 매핑이 없는 경우 정제된 원시 라벨 fallback, 그래도 없으면 '기타/미분류'로 구분 집계
    if sec_col and sec_col in valid.columns:
        fallback_sec = clean_sector_series(valid[sec_col])
        valid["sector_clean"] = valid["sector_clean"].fillna(fallback_sec)
    valid["sector_clean"] = valid["sector_clean"].fillna("기타/미분류")

    s_df = valid[valid["date"] == dates[0]].copy()
    e_df = valid[valid["date"] == dates[-1]].copy()

    s_df["rank_start"] = s_df[mcap_col].rank(ascending=False, method="min")
    e_df["rank_end"] = e_df[mcap_col].rank(ascending=False, method="min")

    top30_start = set(s_df[s_df["rank_start"] <= 30]["code_z"])
    top30_end = set(e_df[e_df["rank_end"] <= 30]["code_z"])
    new_codes = top30_end - top30_start

    new_df = e_df[e_df["code_z"].isin(new_codes)].copy()
    new_counts = (
        new_df.groupby("sector_clean", as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
        .sort_values("cnt", ascending=False)
    )

    merged = pd.merge(
        s_df[["code_z", "sector_clean", "rank_start"]],
        e_df[["code_z", "sector_clean", "rank_end"]],
        on="code_z",
        how="inner",
        suffixes=("_start", "_end"),
    )
    merged["sector_final"] = merged["sector_clean_end"].fillna(merged["sector_clean_start"])
    merged["abs_rank_shift"] = (merged["rank_end"] - merged["rank_start"]).abs()

    grp = merged.groupby("sector_final", as_index=False).agg(
        n_stocks=("code_z", "count"),
        mean_abs_shift=("abs_rank_shift", "mean"),
    )
    # 종목 수 3개 이상 우선, 부족하면 전체 집계 사용
    grp_3 = grp[grp["n_stocks"] >= 3]
    use_grp = grp_3 if len(grp_3) >= 2 else grp
    shift_out = (
        use_grp.rename(columns={"sector_final": "sector_clean", "mean_abs_shift": "shift_val"})
        .dropna(subset=["sector_clean", "shift_val"])
        .sort_values("shift_val", ascending=False)
    )

    return shift_out[["sector_clean", "shift_val"]], new_counts[["sector_clean", "cnt"]]


def load_or_compute_q1_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_clean = pd.DataFrame(columns=["sector_clean", "shift_val"])
    new_counts = pd.DataFrame(columns=["sector_clean", "cnt"])

    # 1) CSV 시도
    for d in ["results_q1_q4", "results_all_8q", "results_spec1_spec2"]:
        s_cand = safe_read_csv(os.path.join(d, "q1_sector_rank_shift.csv"))
        if not s_cand.empty:
            sec_col = next((c for c in ["sector_final", "sector", "raw_sector", "업종"] if c in s_cand.columns), s_cand.columns[0])
            val_col = next((c for c in ["mean_abs_shift", "abs_rank_shift", "rank_shift", "mean_shift"] if c in s_cand.columns), None)
            if val_col:
                tmp = s_cand.copy()
                tmp["sector_clean"] = clean_sector_series(tmp[sec_col])
                tmp["shift_val"] = pd.to_numeric(tmp[val_col], errors="coerce")
                tmp = tmp.dropna(subset=["sector_clean", "shift_val"]).sort_values("shift_val", ascending=False)
                if len(tmp) >= 2:
                    shift_clean = tmp[["sector_clean", "shift_val"]]
                    break

    for d in ["results_q1_q4", "results_all_8q", "results_spec1_spec2"]:
        for fn in ["q1_top30_new_entries.csv", "q1_new_top30.csv"]:
            n_cand = safe_read_csv(os.path.join(d, fn))
            if not n_cand.empty:
                lookup = build_code_sector_lookup()
                tmp = n_cand.copy()
                if "code" in tmp.columns:
                    tmp["sector_clean"] = tmp["code"].astype(str).str.strip().str.zfill(6).map(lookup)
                if "sector_clean" not in tmp.columns or tmp["sector_clean"].isna().all():
                    n_sec_col = next((c for c in ["sector", "raw_sector", "업종"] if c in tmp.columns), None)
                    if n_sec_col:
                        tmp["sector_clean"] = clean_sector_series(tmp[n_sec_col])
                if "sector_clean" in tmp.columns:
                    tmp = tmp.dropna(subset=["sector_clean"])
                    if len(tmp) >= 1:
                        new_counts = (
                            tmp.groupby("sector_clean", as_index=False)
                            .size()
                            .rename(columns={"size": "cnt"})
                            .sort_values("cnt", ascending=False)
                        )
                        break
        if not new_counts.empty:
            break

    # 2) CSV가 비었거나 부족하면 parquet에서 직접 계산
    if len(shift_clean) < 2 or new_counts.empty:
        pq_shift, pq_new = compute_from_parquet()
        if len(shift_clean) < 2 and not pq_shift.empty:
            shift_clean = pq_shift
        if new_counts.empty and not pq_new.empty:
            new_counts = pq_new

    return shift_clean.head(10).iloc[::-1], new_counts.head(10).iloc[::-1]


def build_plot(out_path: str = "plots_q1_q4/q1_market_cap_rank_shift.png") -> None:
    setup_korean_font()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shift_plot, new_plot = load_or_compute_q1_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("white")

    if not shift_plot.empty:
        ax1.barh(
            shift_plot["sector_clean"],
            shift_plot["shift_val"],
            color="#2b6cb0",
            edgecolor="none",
        )
        max_v = float(shift_plot["shift_val"].max())
        ax1.set_xlim(0, max_v * 1.18 if max_v > 0 else 1.0)
        for idx, (_, r) in enumerate(shift_plot.iterrows()):
            ax1.text(
                float(r["shift_val"]) + (max_v * 0.02 if max_v > 0 else 0.02),
                idx,
                f"{float(r['shift_val']):.1f}",
                va="center",
                fontsize=9,
                color="#1f2937",
            )
    else:
        ax1.text(0.5, 0.5, "유효 순위 이동 데이터 없음", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title("업종별 평균 순위 이동 크기", fontsize=12, fontweight="bold")
    ax1.set_xlabel("평균 순위 변동 폭", fontsize=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.45)

    if not new_plot.empty:
        ax2.barh(
            new_plot["sector_clean"],
            new_plot["cnt"],
            color="#2f855a",
            edgecolor="none",
        )
        max_c = float(new_plot["cnt"].max())
        ax2.set_xlim(0, max_c * 1.18 if max_c > 0 else 1.0)
        for idx, (_, r) in enumerate(new_plot.iterrows()):
            ax2.text(
                float(r["cnt"]) + (max_c * 0.02 if max_c > 0 else 0.2),
                idx,
                f"{int(r['cnt'])}",
                va="center",
                fontsize=9,
                color="#1f2937",
            )
    else:
        ax2.text(0.5, 0.5, "신규 진입 업종 데이터 없음", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("Top 30 신규 진입 종목 수 (업종별)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("종목 수", fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)

    fig.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    build_plot()
