import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIN_DIR = "fin_parquet"
OUT_DIR = "plots_fin_5q"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["axes.unicode_minus"] = False

files = sorted(glob.glob(os.path.join(FIN_DIR, "*.parquet")))
if not files:
    print("fin_parquet 파일이 아직 없습니다. parse_financials.py를 먼저 실행하세요.")
    raise SystemExit

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# 좌측: 일자별 파싱된 레코드 수
daily_cnt = df.groupby("date").size().sort_index()
axes[0].bar(daily_cnt.index.astype(str), daily_cnt.values, color="#2b5c8f")
axes[0].set_title("Parsed Financial Rows by Date")
axes[0].set_ylabel("Rows")
axes[0].tick_params(axis="x", rotation=25)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

# 우측: 항목(metric)별 데이터 건수
m_cnt = df["metric"].value_counts().iloc[::-1]
axes[1].barh(m_cnt.index.astype(str), m_cnt.values, color="#3b9c7c")
axes[1].set_title("Records by Financial Metric")
axes[1].set_xlabel("Count")
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fin_step1_parsing_overview.png")
plt.savefig(out_path)
plt.close(fig)
print("저장 완료:", out_path)
