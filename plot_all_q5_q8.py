import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SRC_DIR = "results_all_8q"
OUT_DIR = "plots_all_8q"
os.makedirs(OUT_DIR, exist_ok=True)

for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

# Q5. 업종별 평균 영업이익률 vs 누적수익률
q5_path = os.path.join(SRC_DIR, "q5_opm_vs_return_sector.csv")
if os.path.exists(q5_path):
    q5 = pd.read_csv(q5_path)
    q5["mean_opm"] = pd.to_numeric(q5.get("mean_opm"), errors="coerce")
    q5["mean_cum_ret"] = pd.to_numeric(q5.get("mean_cum_ret"), errors="coerce")
    valid_q5 = q5.dropna(subset=["mean_opm", "mean_cum_ret"])

    if not valid_q5.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        ax.scatter(valid_q5["mean_opm"], valid_q5["mean_cum_ret"], alpha=0.75, color="#2b5c8f")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Q5. 업종별 평균 영업이익률 vs 누적 수익률")
        ax.set_xlabel("평균 영업이익률")
        ax.set_ylabel("평균 누적 수익률")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "q5_opm_vs_return.png"))
        plt.close(fig)
    else:
        ret_q5 = q5.dropna(subset=["mean_cum_ret"]).sort_values("mean_cum_ret")
        top_bot = pd.concat([ret_q5.head(8), ret_q5.tail(8)]).drop_duplicates(subset=["sector"])
        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        colors = ["#c0392b" if v < 0 else "#2b5c8f" for v in top_bot["mean_cum_ret"]]
        ax.barh(top_bot["sector"].astype(str), top_bot["mean_cum_ret"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Q5. 업종별 평균 누적 수익률 (하위 8 · 상위 8, mean_opm 결측)")
        ax.set_xlabel("평균 누적 수익률")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "q5_opm_vs_return.png"))
        plt.close(fig)
# Q6. 실적 발표 전후(-10일 ~ +10일) Beat vs Miss 누적수익률
q6_path = os.path.join(SRC_DIR, "q6_event_car_10d.csv")
if os.path.exists(q6_path):
    q6 = pd.read_csv(q6_path, index_col=0)
    car_cols = [c for c in q6.columns if c.startswith("car_t")]
    if car_cols:
        x_labels = [c.replace("car_t", "t") for c in car_cols]
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        for idx, row in q6.iterrows():
            ax.plot(x_labels, row[car_cols].values, marker="o", label=str(idx))
        ax.axvline(10, color="red", linestyle="--", linewidth=1, label="발표일(t0)")
        ax.set_title("Q6. 실적 발표 전후 10일 누적수익률 (Beat vs Miss)")
        ax.set_ylabel("누적수익률(CAR)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "q6_earnings_car.png"))
        plt.close(fig)

# Q7. 업종별 추정치 오차 절대평균(MAE) 상위 12개
q7_path = os.path.join(SRC_DIR, "q7_forecast_error_bias.csv")
if os.path.exists(q7_path):
    q7 = pd.read_csv(q7_path)
    q7_top = q7.groupby("sector", as_index=False)["mae"].mean().nlargest(12, "mae").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.barh(q7_top["sector"].astype(str), q7_top["mae"], color="#e67e22")
    ax.set_title("Q7. 업종별 평균 절대 추정 오차(MAE) 상위 12개")
    ax.set_xlabel("평균 절대 오차(MAE)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q7_forecast_mae.png"))
    plt.close(fig)

# Q8. 저PBR(<1) 종목 분포 업종 TOP 10 & ROE 양수/음수 수익률 비교
q8_sec_path = os.path.join(SRC_DIR, "q8_low_pbr_by_sector.csv")
q8_roe_path = os.path.join(SRC_DIR, "q8_low_pbr_by_roe.csv")
if os.path.exists(q8_sec_path) and os.path.exists(q8_roe_path):
    s_df = pd.read_csv(q8_sec_path).head(10).iloc[::-1]
    r_df = pd.read_csv(q8_roe_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    axes[0].barh(s_df["sector"].astype(str), s_df["n"], color="#3b9c7c")
    axes[0].set_title("Q8-1. 저PBR(<1) 종목 수 상위 10개 업종")
    axes[0].set_xlabel("종목 수")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    if "mean" in r_df.columns and "roe_pos" in r_df.columns:
        axes[1].bar(r_df["roe_pos"].astype(str), r_df["mean"], color=["#2b5c8f", "#c0392b"])
        axes[1].set_title("Q8-2. 저PBR 종목 ROE 부호별 평균 누적수익률")
        axes[1].set_ylabel("평균 누적수익률")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q8_low_pbr_summary.png"))
    plt.close(fig)

print("Q5~Q8 시각화 저장 완료 ->", OUT_DIR)
