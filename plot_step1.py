import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

in_dir = "results_q1_q4"
out_dir = "plots_q1_q4"
os.makedirs(out_dir, exist_ok=True)

has_kr = False
for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        has_kr = True
        break
plt.rcParams["axes.unicode_minus"] = False

shift_df = pd.read_csv(os.path.join(in_dir, "q1_sector_rank_shift.csv")).dropna(subset=["mean_abs_shift"]).head(10).iloc[::-1]
new_path = os.path.join(in_dir, "q1_top30_new_entries.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

y_col = "sector_final" if "sector_final" in shift_df.columns else shift_df.columns[0]
axes[0].barh(shift_df[y_col].astype(str), shift_df["mean_abs_shift"], color="#2b5c8f")
axes[0].set_title("업종별 평균 순위 이동 크기" if has_kr else "Mean Abs Rank Shift by Sector")
axes[0].set_xlabel("평균 순위 변동 폭" if has_kr else "Mean Abs Shift")
axes[0].grid(axis="x", linestyle="--", alpha=0.4)

if os.path.exists(new_path):
    new_df = pd.read_csv(new_path)
    if not new_df.empty and "sector" in new_df.columns:
        cnt = new_df["sector"].fillna("미분류").value_counts().head(10).iloc[::-1]
        axes[1].barh(cnt.index.astype(str), cnt.values, color="#3b9c7c")
        axes[1].set_title("Top 30 신규 진입 종목 수 (업종별)" if has_kr else "New Top-30 Entrants by Sector")
        axes[1].set_xlabel("종목 수" if has_kr else "Count")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    else:
        axes[1].text(0.5, 0.5, "신규 진입 종목 없음" if has_kr else "No new entrants", ha="center", va="center")
        axes[1].set_axis_off()
else:
    axes[1].set_axis_off()

plt.tight_layout()
out_file = os.path.join(out_dir, "q1_market_cap_rank_shift.png")
plt.savefig(out_file)
plt.close(fig)
print("저장 완료:", out_file)
