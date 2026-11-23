import glob
import pandas as pd
files = sorted(glob.glob("plots_fin_5q/parquet/*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

if "mktcap_100mn" not in df.columns and "market_cap" in df.columns:
    df = df.rename(columns={"market_cap": "mktcap_100mn"})

df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["code", "date", "sector", "close", "mktcap_100mn"])
df = df.sort_values(["code", "date"])

df["ret"] = df.groupby("code")["close"].pct_change()
df = df[df["ret"].between(-0.5, 0.5)]
df["w"] = df.groupby("code")["mktcap_100mn"].shift(1)

valid = df.dropna(subset=["ret", "w"]).copy()
valid["w_ret"] = valid["ret"] * valid["w"]

agg_cap = valid.groupby(["date", "sector"], as_index=False)[["w_ret", "w"]].sum()
agg_cap["sec_ret_cap"] = agg_cap["w_ret"] / agg_cap["w"]
sector_ret_cap = agg_cap.pivot(index="date", columns="sector", values="sec_ret_cap")
sector_cum_cap = (1 + sector_ret_cap.fillna(0)).cumprod() - 1

sector_ret_eq = valid.groupby(["date", "sector"])["ret"].mean().unstack("sector")
sector_cum_eq = (1 + sector_ret_eq.fillna(0)).cumprod() - 1

out = pd.DataFrame({
    "cap_weighted_cum": sector_cum_cap.iloc[-1],
    "equal_weighted_cum": sector_cum_eq.iloc[-1],
})
out["diff_cap_minus_eq"] = out["cap_weighted_cum"] - out["equal_weighted_cum"]

print("기간:", sector_ret_cap.index.min().date(), "~", sector_ret_cap.index.max().date())
print("\n[상위 10개 업종 - 시총가중 누적수익률]")
print(out.sort_values("cap_weighted_cum", ascending=False).head(10).to_string())
print("\n[하위 10개 업종 - 시총가중 누적수익률]")
print(out.sort_values("cap_weighted_cum", ascending=True).head(10).to_string())
