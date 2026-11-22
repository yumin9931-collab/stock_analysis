import os
import numpy as np
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

disp_df = pd.read_csv(os.path.join(in_dir, "q2_monthly_dispersion.csv"))
date_col = disp_df.columns[0]
disp_df[date_col] = pd.to_datetime(disp_df[date_col], errors="coerce")
disp_df = disp_df.dropna(subset=[date_col]).sort_values(date_col)

drv_df = pd.read_csv(os.path.join(in_dir, "q2_sector_dispersion_drivers.csv")).head(10).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=150)

mean_col = "mean" if "mean" in disp_df.columns else disp_df.columns[1]
axes[0].plot(disp_df[date_col], disp_df[mean_col], marker="o", markersize=3, color="#c0392b", linewidth=1.8)
axes[0].set_title("월별 업종 간 수익률 분산 추이" if has_kr else "Monthly Sector Return Variance")
axes[0].set_ylabel("일별 분산의 월평균" if has_kr else "Monthly Mean Variance")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].tick_params(axis="x", rotation=25)

colors = np.where(drv_df["diff_from_mean"] >= 0, "#2980b9", "#e67e22")
axes[1].barh(drv_df["sector"].astype(str), drv_df["diff_from_mean"] * 100.0, color=colors)
axes[1].axvline(0, color="black", linewidth=0.9)
axes[1].set_title("업종 평균 대비 누적수익률 차이 (%p)" if has_kr else "Cum. Return Diff vs. Mean (%p)")
axes[1].set_xlabel("차이 (%p)" if has_kr else "Difference (%p)")
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
out_file = os.path.join(out_dir, "q2_sector_dispersion.png")
plt.savefig(out_file)
plt.close(fig)
print("저장 완료:", out_file)
