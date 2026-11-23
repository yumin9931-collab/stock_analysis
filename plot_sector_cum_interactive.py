import glob
import pandas as pd
import plotly.graph_objects as go

# 1. Parquet 로드 및 기본 정제
files = sorted(glob.glob("plots_fin_5q/parquet/*.parquet"))
if not files:
    raise FileNotFoundError("plots_fin_5q/parquet/*.parquet 파일을 찾지 못했습니다.")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
if "mktcap_100mn" not in df.columns and "market_cap" in df.columns:
    df = df.rename(columns={"market_cap": "mktcap_100mn"})

df["date"] = pd.to_datetime(df["date"])
df["sector"] = df["sector"].astype(str).str.strip()

# 날짜(02/11), 각주([1]), 출처/기업설명 문구 등 오염 라벨 제외
noise_mask = (
    df["sector"].str.match(r"^\d{1,2}/\d{1,2}$")
    | df["sector"].str.match(r"^\[?\d+\]?$")
    | df["sector"].str.contains(r"출처|경쟁사|주 업종|함\.", na=False)
    | df["sector"].isin(["", "nan", "None"])
)
df = df[~noise_mask].dropna(subset=["code", "date", "sector", "close", "mktcap_100mn"])
df = df.sort_values(["code", "date"])

# 2. 일별 수익률 및 전일 시가총액 가중치 (±50% 극단치 제외)
df["ret"] = df.groupby("code")["close"].pct_change()
df = df[df["ret"].between(-0.5, 0.5)]
df["w"] = df.groupby("code")["mktcap_100mn"].shift(1)

valid = df.dropna(subset=["ret", "w"]).copy()
valid["w_ret"] = valid["ret"] * valid["w"]

# (a) 시가총액 가중 일별·누적 수익률
agg_cap = valid.groupby(["date", "sector"], as_index=False)[["w_ret", "w"]].sum()
agg_cap["sec_ret_cap"] = agg_cap["w_ret"] / agg_cap["w"]
ret_cap = agg_cap.pivot(index="date", columns="sector", values="sec_ret_cap").sort_index()
cum_cap = (1 + ret_cap.fillna(0)).cumprod() - 1

# (b) 동일 가중 일별·누적 수익률
ret_eq = valid.groupby(["date", "sector"])["ret"].mean().unstack("sector").sort_index()
ret_eq = ret_eq.reindex(index=ret_cap.index, columns=ret_cap.columns)
cum_eq = (1 + ret_eq.fillna(0)).cumprod() - 1

# 시총-동일 누적 격차
cum_diff = cum_cap - cum_eq

# 발표 화면 가독성을 위해 시총가중 최종 누적수익률 기준 상위 6개 + 하위 6개 업종 선정
final_cap = cum_cap.iloc[-1].dropna().sort_values(ascending=False)
selected_sectors = list(dict.fromkeys(list(final_cap.head(6).index) + list(final_cap.tail(6).index)))

# 3. Plotly 그래프 생성 (기본은 시가총액 가중)
fig = go.Figure()

for sec in selected_sectors:
    fig.add_trace(
        go.Scatter(
            x=cum_cap.index,
            y=cum_cap[sec] * 100,
            mode="lines",
            name=sec,
            hovertemplate=(
                f"<b>{sec}</b><br>"
                "날짜: %{x|%Y-%m-%d}<br>"
                "누적수익률: %{y:.2f}%<extra></extra>"
            ),
        )
    )

# 버튼으로 시가총액 가중 / 동일 가중 / 가중치 차이(시총-동일) 전환
cap_y_list = [cum_cap[sec] * 100 for sec in selected_sectors]
eq_y_list = [cum_eq[sec] * 100 for sec in selected_sectors]
diff_y_list = [cum_diff[sec] * 100 for sec in selected_sectors]

fig.update_layout(
    title="업종별 누적수익률 곡선 — 시가총액 가중 vs 동일 가중 (상위 6 · 하위 6)",
    xaxis_title="날짜",
    yaxis_title="누적 수익률 (%)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            y=1.13,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="시가총액 가중 (기본)",
                    method="update",
                    args=[
                        {"y": cap_y_list},
                        {
                            "title": "업종별 누적수익률 곡선 — 전일 시가총액 가중 (상위 6 · 하위 6)",
                            "yaxis.title.text": "누적 수익률 (%)",
                        },
                    ],
                ),
                dict(
                    label="동일 가중 (w=1)",
                    method="update",
                    args=[
                        {"y": eq_y_list},
                        {
                            "title": "업종별 누적수익률 곡선 — 동일 가중 (상위 6 · 하위 6)",
                            "yaxis.title.text": "누적 수익률 (%)",
                        },
                    ],
                ),
                dict(
                    label="차이 (시총가중 - 동일가중)",
                    method="update",
                    args=[
                        {"y": diff_y_list},
                        {
                            "title": "누적수익률 격차 (시가총액 가중 - 동일 가중, %p)",
                            "yaxis.title.text": "누적수익률 차이 (%p)",
                        },
                    ],
                ),
            ],
        )
    ],
)

out_html = "sector_cum_toggle.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print("저장 완료:", out_html)
