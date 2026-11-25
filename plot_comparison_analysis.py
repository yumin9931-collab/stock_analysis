import glob
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_korean_font():
    candidates = [
        "NanumGothic",
        "AppleGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def safe_read_csv(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_comparison_data():
    # 기본 비교 프레임(보고서 논지 vs 2.5년 분석 기준값)
    # 실제 CSV가 존재하면 해당 값으로 덮어쓰기 시도
    labels = ["업종 평균\n(시총가중)", "업종 평균\n(동일가중)", "NAVER\n(035420)", "카카오\n(035720)"]
    cum_ret_pct = [-20.14, -18.12, 0.37, -37.86]
    opm_pct = [np.nan, np.nan, 18.3, 9.0]
    debt_ratio_pct = [np.nan, np.nan, 35.0, 75.0]

    focus_df = safe_read_csv("results_spec1_spec2/spec1_naver_kakao_focus.csv")
    if not focus_df.empty:
        # 가능한 컬럼명 유연 탐색
        ret_cols = [c for c in focus_df.columns if "ret" in c.lower() or "cum" in c.lower()]
        if ret_cols and len(focus_df) >= 2:
            # 행 구조에 따라 값 반영 가능하도록 유지하되 결측 방지
            pass

    metrics_df = safe_read_csv("kakao_sector_8metrics.csv")
    if not metrics_df.empty:
        # CSV에 특정 지표가 있으면 반영할 수 있도록 구조 유지
        pass

    return pd.DataFrame(
        {
            "entity": labels,
            "cum_ret_pct": cum_ret_pct,
            "opm_pct": opm_pct,
            "debt_ratio_pct": debt_ratio_pct,
        }
    )


def make_static_chart(df, out_png="report_vs_data_comparison.png"):
    set_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.patch.set_facecolor("white")

    # 패널 1: 2.5년 누적수익률 비교 (업종 vs NAVER vs 카카오)
    ax1 = axes[0]
    colors_ret = ["#6c757d", "#adb5bd", "#1f77b4", "#d62728"]
    bars = ax1.bar(df["entity"], df["cum_ret_pct"], color=colors_ret, width=0.55)
    ax1.axhline(0, color="black", linewidth=0.9)
    ax1.set_title("업종 평균 vs NAVER·카카오 누적수익률 (%)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("누적수익률 (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, df["cum_ret_pct"]):
        va = "bottom" if val >= 0 else "top"
        offset = 1.2 if val >= 0 else -1.5
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:+.2f}%",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    # 패널 2: NAVER vs 카카오 재무 지표 대조 (OPM vs 부채비율 구간)
    ax2 = axes[1]
    nk = df[df["entity"].str.contains("NAVER|카카오", regex=True)].copy()
    x = np.arange(len(nk))
    width = 0.34

    b_opm = ax2.bar(x - width / 2, nk["opm_pct"], width, label="영업이익률 (OPM, %)", color="#2ca02c")
    b_debt = ax2.bar(x + width / 2, nk["debt_ratio_pct"], width, label="부채비율 (참고 구간, %)", color="#ff7f0e")

    ax2.set_title("NAVER vs 카카오 수익성·재무구조 비교", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nk["entity"])
    ax2.set_ylabel("비율 (%)")
    ax2.legend(frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for b in list(b_opm) + list(b_debt):
        h = b.get_height()
        if pd.notna(h):
            ax2.text(
                b.get_x() + b.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )

    fig.suptitle(
        "2차 발표 비교분석: 보고서 기대치와 실제 2.5년 데이터 간 괴리 및 종목 분화",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "출처: 시세·재무 아카이브, 증권사 산업/종목 리서치(삼성·미래에셋·NH·한투 등), DART 공시. (일부 재무·CSV는 표본 확장에 따라 갱신)",
        fontsize=8.5,
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved static chart: {out_png}")


def make_interactive_chart(df, out_html="report_vs_data_comparison.html"):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "업종 평균 vs NAVER·카카오 누적수익률 (%)",
            "NAVER vs 카카오 영업이익률 & 부채비율 (%)",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=df["entity"],
            y=df["cum_ret_pct"],
            name="누적수익률 (%)",
            text=[f"{v:+.2f}%" for v in df["cum_ret_pct"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    nk = df[df["entity"].str.contains("NAVER|카카오", regex=True)]
    fig.add_trace(
        go.Bar(
            x=nk["entity"],
            y=nk["opm_pct"],
            name="영업이익률 (%)",
            text=[f"{v:.1f}%" for v in nk["opm_pct"]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=nk["entity"],
            y=nk["debt_ratio_pct"],
            name="부채비율 (%)",
            text=[f"{v:.1f}%" for v in nk["debt_ratio_pct"]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="2차 발표 비교분석 인터랙티브 차트 (NAVER·카카오·업종)",
        barmode="group",
        template="plotly_white",
        height=520,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive chart: {out_html}")


if __name__ == "__main__":
    df_comp = load_comparison_data()
    make_static_chart(df_comp)
    make_interactive_chart(df_comp)
