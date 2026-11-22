import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

in_dir, out_dir = "results_q1_q4", "plots_q1_q4"
os.makedirs(out_dir, exist_ok=True)

for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

roll_path = os.path.join(in_dir, "q3_rolling_60d_corr.csv")
mat_path = os.path.join(in_dir, "q3_sector_corr_matrix.csv")

roll_df = pd.DataFrame()
if os.path.exists(roll_path) and os.path.getsize(roll_path) > 10:
    roll_df = pd.read_csv(roll_path)
    if "date" in roll_df.columns:
        roll_df["date"] = pd.to_datetime(roll_df["date"], errors="coerce")
        roll_df = roll_df.dropna(subset=["date"]).sort_values("date")

corr_mat = pd.read_csv(mat_path, index_col=0)
if corr_mat.shape[0] > 20:
    corr_mat = corr_mat.iloc[:20, :20]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

if not roll_df.empty and "rolling_60d_mean_corr" in roll_df.columns:
    axes[0].plot(roll_df["date"], roll_df["rolling_60d_mean_corr"], color="#8e44ad", linewidth=1.8)
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="x", rotation=25)
else:
    axes[0].text(0.5, 0.5, "60d rolling data not available\n(fewer than 60 trading days)", ha="center", va="center")
    axes[0].set_axis_off()
axes[0].set_title("Rolling 60D Mean Correlation")

im = axes[1].imshow(corr_mat.values, cmap="coolwarm", vmin=-0.2, vmax=1.0, aspect="auto")
axes[1].set_xticks(range(len(corr_mat.columns)))
axes[1].set_yticks(range(len(corr_mat.index)))
axes[1].set_xticklabels(corr_mat.columns.astype(str), rotation=90, fontsize=7)
axes[1].set_yticklabels(corr_mat.index.astype(str), fontsize=7)
axes[1].set_title("Sector Return Correlation Matrix")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
out_file = os.path.join(out_dir, "q3_sector_comovement.png")
plt.savefig(out_file)
plt.close(fig)
print("Saved:", out_file)
