import glob
import os
import re
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
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_or_compute_q1_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_df = pd.DataFrame()
    new_df = pd.DataFrame()

    for d in ["results_q1_q4", "results_all_8q", "results_spec1_spec2"]:
        s_cand = safe_read_csv(os.path.join(d, "q1_sector_rank_shift.csv"))
        if not s_cand.empty:
            shift_df = s_cand
            break

    for d in ["results_q1_q4", "results_all_8q", "results_spec1_spec2"]:
        for fn in ["q1_top30_new_entries.csv", "q1_new_top30.csv"]:
            n_cand = safe_read_csv(os.path.join(d, fn))
            if not n_cand.empty:
                new_df = n_cand
                break
        if not new_df.empty:
            break

    # CSV가 없거나 유효한 순위 이동 값이 부족한 경우 parquet에서 직접 산출
    need_recompute = shift_df.empty
    if not need_recompute:
        val_col = next(
            (c for c in ["mean_abs_shift", "abs_rank_shift", "rank_shift", "mean_shift"] if c in shift_df.columns),
            None,
        )
        if not val_col or pd.to_numeric(shift_df[val_col], errors="coerce").dropna().empty:
            need_recompute = True

    if need_recompute:
        p_files = sorted(glob.glob(os.path.expanduser("~/parquet/*.parquet")))
        if not p_files:
            p_files = sorted(glob.glob("parquet/*.parquet"))

        dfs = []
        for fp in p_files:
            try:
                dfs.append(pd.read_parquet(fp))
            except Exception:
                continue

        if dfs:
            raw = pd.concat(dfs, ignore_index=True)
            mcap_col = next((c for c in ["market_cap", "mcap", "시가총액"] if c in raw.columns), None)
            sec_col = next((c for c in ["sector", "raw_sector", "업종"] if c in raw.columns), None)
            if mcap_col and "date" in raw.columns and "code" in raw.columns:
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                raw[mcap_col] = pd.to_numeric(raw[mcap_col], errors="coerce")
                valid = raw.dropna(subset=["date", "code", mcap_col]).copy()
                dates = sorted(valid["date"].unique())
                if len(dates) >= 2:
                    s_df = valid[valid["date"] == dates[0]].copy()
                    e_df = valid[valid["date"] == dates[-1]].copy()
                    s_df["rank_start"] = s_df[mcap_col].rank(ascending=False, method="min")
                    e_df["rank_end"] = e_df[mcap_col].rank(ascending=False, method="min")

                    top30_start = set(s_df[s_df["rank_start"] <= 30]["code"])
                    top30_end = set(e_df[e_df["rank_end"] <= 30]["code"])
                    new_codes = top30_end - top30_start

                    name_col = "name" if "name" in e_df.columns else "code"
                    new_df = e_df[e_df["code"].isin(new_codes)].copy()
                    if sec_col and sec_col in new_df.columns:
                        new_df["sector_clean"] = clean_sector_series(new_df[sec_col])
                    else:
                        new_df["sector_clean"] = np.nan

                    s_sec = clean_sector_series(s_df[sec_col]) if (sec_col and sec_col in s_df.columns) else pd.Series(index=s_df.index, dtype=object)
                    e_sec = clean_sector_series(e_df[sec_col]) if (sec_col and sec_col in e_df.columns) else pd.Series(index=e_df.index, dtype=object)
                    s_df["sector_clean"] = s_sec
                    e_df["sector_clean"] = e_sec

                    merged = pd.merge(
                        s_df[["code", "sector_clean", "rank_start"]],
                        e_df[["code", "sector_clean", "rank_end"]],
                        on="code",
                        how="inner",
                        suffixes=("_start", "_end"),
                    )
                    merged["sector_final"] = merged["sector_clean_end"].fillna(merged["sector_clean_start"])
                    merged["abs_rank_shift"] = (merged["rank_end"] - merged["rank_start"]).abs()
                    merged = merged.dropna(subset=["sector_final", "abs_rank_shift"])

                    grp = merged.groupby("sector_final", as_index=False).agg(
                        n_stocks=("code", "count"),
                        mean_abs_shift=("abs_rank_shift", "mean"),
                    )
                    shift_df = grp[grp["n_stocks"] >= 3].sort_values("mean_abs_shift", ascending=False)

    # 최종 정제 및 수치 변환
    if not shift_df.empty:
        sec_col = next((c for c in ["sector_final", "sector", "raw_sector", "업종"] if c in shift_df.columns), shift_df.columns[0])
        val_col = next((c for c in ["mean_abs_shift", "abs_rank_shift", "rank_shift", "mean_shift"] if c in shift_df.columns), None)
        if val_col is None:
            num_cols = [c for c in shift_df.columns if c != sec_col and pd.api.types.is_numeric_dtype(pd.to_numeric(shift_df[c], errors="coerce"))]
            val_col = num_cols[0] if num_cols else shift_df.columns[-1]

        s_clean = shift_df.copy()
        s_clean["sector_clean"] = clean_sector_series(s_clean[sec_col])
        s_clean["shift_val"] = pd.to_numeric(s_clean[val_col], errors="coerce")
        if "n_stocks" in s_clean.columns:
            s_clean["n_stocks"] = pd.to_numeric(s_clean["n_stocks"], errors="coerce")
            s_clean = s_clean[s_clean["n_stocks"].fillna(3) >= 3]
        s_clean = s_clean.dropna(subset=["sector_clean", "shift_val"]).sort_values("shift_val", ascending=False)
    else:
        s_clean = pd.DataFrame(columns=["sector_clean", "shift_val"])

    if not new_df.empty:
        n_sec_col = next((c for c in ["sector_clean", "sector", "raw_sector", "업종"] if c in new_df.columns), None)
        if n_sec_col:
            n_clean = new_df.copy()
            n_clean["sector_clean"] = clean_sector_series(n_clean[n_sec_col])
            n_clean = n_clean.dropna(subset=["sector_clean"])
            n_counts = (
                n_clean.groupby("sector_clean", as_index=False)
                .size()
                .rename(columns={"size": "cnt"})
                .sort_values("cnt", ascending=False)
            )
        else:
            n_counts = pd.DataFrame(columns=["sector_clean", "cnt"])
    else:
        n_counts = pd.DataFrame(columns=["sector_clean", "cnt"])

    return s_clean.head(10).iloc[::-1], n_counts.head(10).iloc[::-1]


def build_plot(out_path: str = "plots_q1_q4/q1_market_cap_rank_shift.png") -> None:
    setup_korean_font()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shift_plot, new_plot = load_or_compute_q1_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("white")

    # 왼쪽 패널: 업종별 평균 순위 이동 크기
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

    # 오른쪽 패널: 신규 진입 종목 수 (업종별)
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
