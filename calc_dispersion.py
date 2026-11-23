import glob
import re
import pandas as pd

files = sorted(glob.glob("plots_fin_5q/parquet/*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

if "mktcap_100mn" not in df.columns and "market_cap" in df.columns:
    df = df.rename(columns={"market_cap": "mktcap_100mn"})

df["date"] = pd.to_datetime(df["date"])
df["sector"] = df["sector"].astype(str).str.strip()

# 날짜(02/11), 각주([1]), 출처/설명 문구 등 오염 라벨 제외
noise_mask = (
    df["sector"].str.match(r"^\d{1,2}/\d{1,2}$")
    | df["sector"].str.match(r"^\[?\d+\]?$")
    | df["sector"].str.contains("출처|경쟁사|주 업종|함\.", na=False)
    | df["sector"].isin(["", "nan", "None"])
)
df = df[~noise_mask].dropna(subset=["code", "date", "sector", "close", "mktcap_100mn"])
df = df.sort_values(["code", "date"])

# 일별 수익률 및 전일 시가총액 계산 (액면분할/감자 등 ±50% 밖 극단치 제외)
df["ret"] = df.groupby("code")["close"].pct_change()
df = df[df["ret"].between(-0.5, 0.5)]
df["w"] = df.groupby("code")["mktcap_100mn"].shift(1)

valid = df.dropna(subset=["ret", "w"]).copy()
valid["w_ret"] = valid["ret"] * valid["w"]
agg_cap = valid.groupby(["date", "sector"], as_index=False)[["w_ret", "w"]].sum()
agg_cap["sec_ret_cap"] = agg_cap["w_ret"] / agg_cap["w"]

# 날짜 × 업종 일별·누적 수익률
sector_ret = agg_cap.pivot(index="date", columns="sector", values="sec_ret_cap").sort_index()
sector_cum = (1 + sector_ret.fillna(0)).cumprod() - 1

# (a) 월별 수익률의 횡단면 표준편차
monthly = sector_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
disp = monthly.std(axis=1)

# (b) 누적 수익률의 표준편차 및 최고-최저 범위
spread = sector_cum.std(axis=1)
spread_range = sector_cum.max(axis=1) - sector_cum.min(axis=1)

summary = pd.DataFrame({
    "spread_std": spread,
    "spread_range": spread_range,
})

print("기간:", sector_ret.index.min().date(), "~", sector_ret.index.max().date(), f"| 정제 업종 수: {sector_ret.shape[1]}개")
print("\n[월별 업종 간 수익률 분산 (disp = monthly.std(axis=1))]")
print(disp.to_string())

print("\n[누적 수익률 퍼짐 (초기 / 중간 / 말기)]")
sample_idx = sorted(set([0, len(summary) // 2, len(summary) - 1]))
print(summary.iloc[sample_idx].to_string())
