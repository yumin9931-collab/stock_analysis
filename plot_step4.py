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

q4_path = os.path.join(in_dir, "q4_volume_surge_sensitivity.csv")
if not os.path.exists(q4_path) or os.path.getsize(q4_path) == 0:
    raise ValueError(f"'{q4_path}' 파일이 없거나 비어 있습니다. analyze_4_questions.py를 먼저 실행하세요.")

df = pd.read_csv(q4_path)
horizons = [5, 10, 20]
x = np.arange(len(df))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=150)

for idx, h in enumerate(horizons):
    col = f"fwd_{h}d_mean"
    vals = df[col].values * 100.0
    axes[0].bar(x + (idx - 1) * width, vals, width=width, label=f"{h}일 후" if has_kr else f"{h}D fwd")
axes[0].axhline(0, color="black", linewidth=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(df["definition"].astype(str), rotation=18, ha="right", fontsize=9)
axes[0].set_ylabel("평균 수익률 (%)" if has_kr else "Mean Forward Return (%)")
axes[0].set_title("거래량 급증 정의별 이후 수익률" if has_kr else "Forward Return by Surge Definition")
axes[0].legend()
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

for idx, h in enumerate(horizons):
    col = f"fwd_{h}d_excess_vs_uncond"
    vals = df[col].values * 100.0
    axes[1].bar(x + (idx - 1) * width, vals, width=width, label=f"{h}일 초과" if has_kr else f"{h}D excess")
axes[1].axhline(0, color="black", linewidth=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(df["definition"].astype(str), rotation=18, ha="right", fontsize=9)
axes[1].set_ylabel("무조건부 대비 초과수익률 (%p)" if has_kr else "Excess vs. Unconditional (%p)")
axes[1].set_title("급증 정의 변경에 따른 초과수익률 민감도" if has_kr else "Excess Return Sensitivity")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out_file = os.path.join(out_dir, "q4_volume_lead_lag.png")
plt.savefig(out_file)
plt.close(fig)
print("저장 완료:", out_file)
