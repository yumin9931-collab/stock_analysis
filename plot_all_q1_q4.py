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

# Q1. 업종별 시가총액 순위 이동 크기 상위 10개
q1_path = os.path.join(SRC_DIR, "q1_sector_rank_shift.csv")
if os.path.exists(q1_path):
    q1 = pd.read_csv(q1_path).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.barh(q1["sector"].astype(str), q1["mean_abs_shift"], color="#2b5c8f")
    ax.set_title("Q1. 업종별 평균 절대 순위 이동 크기 (상위 10개)")
    ax.set_xlabel("평균 절대 순위 변화")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q1_rank_shift.png"))
    plt.close(fig)

# Q2. 업종 간 일별 수익률 분산 추이 & 누적수익률 괴리 상위 업종
q2_disp_path = os.path.join(SRC_DIR, "q2_daily_dispersion.csv")
q2_gap_path = os.path.join(SRC_DIR, "q2_sector_cum_return_gap.csv")
if os.path.exists(q2_disp_path) and os.path.exists(q2_gap_path):
    disp = pd.read_csv(q2_disp_path)
    disp["date"] = pd.to_datetime(disp["date"])
    gap = pd.read_csv(q2_gap_path).head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    axes[0].plot(disp["date"], disp["cross_sec_var"], color="#1f77b4", linewidth=1.3)
    axes[0].set_title("Q2-1. 업종 간 수익률 분산 추이")
    axes[0].set_ylabel("횡단면 분산")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].barh(gap["sector"].astype(str), gap["diff_from_mean"], color="#d95f02")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Q2-2. 업종 평균 대비 누적수익률 괴리 (상위 10개)")
    axes[1].set_xlabel("누적수익률 차이")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q2_sector_dispersion_gap.png"))
    plt.close(fig)

# Q3. 롤링 평균 상관 추이 & 충격일 vs 보통일 비교
q3_roll_path = os.path.join(SRC_DIR, "q3_rolling_corr.csv")
q3_reg_path = os.path.join(SRC_DIR, "q3_shock_vs_normal_corr.csv")
if os.path.exists(q3_roll_path) and os.path.exists(q3_reg_path):
    roll = pd.read_csv(q3_roll_path)
    roll["date"] = pd.to_datetime(roll["date"])
    reg = pd.read_csv(q3_reg_path).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    axes[0].plot(roll["date"], roll["mean_pair_corr"], color="#6a3d9a", linewidth=1.5)
    axes[0].set_title("Q3-1. 업종 쌍별 평균 상관계수 추이")
    axes[0].set_ylabel("평균 상관계수")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="x", rotation=25)

    labels = ["전체", "하락충격(하위10%)", "보통일"]
    vals = [reg["all_days_corr"], reg["shock_bottom10_corr"], reg["normal_days_corr"]]
    axes[1].bar(labels, vals, color=["#7570b3", "#e7298a", "#1b9e77"])
    axes[1].set_title("Q3-2. 시장 충격일 vs 보통일 업종 상관")
    axes[1].set_ylabel("평균 상관계수")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q3_comovement_regime.png"))
    plt.close(fig)

# Q4. 거래량 급증 정의별 5·10·20일 평균 수익률
q4_path = os.path.join(SRC_DIR, "q4_volume_surge_all_defs.csv")
if os.path.exists(q4_path):
    q4 = pd.read_csv(q4_path).set_index("definition")[["fwd_5d_mean", "fwd_10d_mean", "fwd_20d_mean"]]
    ax = q4.plot(kind="bar", figsize=(10, 4.8), dpi=140)
    ax.set_title("Q4. 거래량 급증 정의별 이후 5·10·20일 평균 수익률")
    ax.set_ylabel("Forward 수익률")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "q4_volume_fwd_returns.png"))
    plt.close(fig)

print("Q1~Q4 시각화 저장 완료 ->", OUT_DIR)
