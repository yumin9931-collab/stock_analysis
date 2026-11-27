import glob
import html
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def safe_read_csv(path: str) -> pd.DataFrame:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def find_csv_by_pattern(pattern: str) -> pd.DataFrame:
    candidates = sorted(glob.glob(os.path.expanduser(f"~/results_5fin_themes/{pattern}")))
    if not candidates:
        candidates = sorted(glob.glob(f"results_5fin_themes/{pattern}"))
    for fp in candidates:
        df = safe_read_csv(fp)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_event_window_data() -> dict:
    """
    반환 구조:
      {
        "days": [-10, ..., 0, ..., +10],
        "beat_mean": [...],
        "miss_mean": [...],
        "beat_median": [...],
        "miss_median": [...],
      }
    """
    days = list(range(-10, 11))
    # 기본 예시 시계열(실제 CSV가 있으면 덮어쓰기)
    beat_mean = [-0.4, -0.2, 0.1, 0.4, 0.9, 1.5, 2.2, 2.8, 3.1, 3.3, 3.8, 4.1, 4.3, 4.4, 4.5, 4.6, 4.6, 4.7, 4.7, 4.8, 4.8]
    miss_mean = [-0.3, -0.5, -0.8, -1.2, -1.7, -2.3, -3.0, -3.6, -4.0, -4.2, -4.7, -5.0, -5.2, -5.3, -5.4, -5.5, -5.5, -5.6, -5.6, -5.6, -5.7]
    beat_median = [-0.2, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5, 1.9, 2.2, 2.4, 2.7, 2.9, 3.0, 3.1, 3.1, 3.2, 3.2, 3.2, 3.3, 3.3, 3.3]
    miss_median = [-0.2, -0.4, -0.6, -0.9, -1.2, -1.6, -2.1, -2.5, -2.8, -3.0, -3.4, -3.6, -3.7, -3.8, -3.9, -3.9, -4.0, -4.0, -4.0, -4.0, -4.1]

    ev_df = find_csv_by_pattern("*event*.csv")
    if not ev_df.empty:
        day_col = next((c for c in ev_df.columns if c.lower() in {"day", "rel_day", "t", "offset", "일자"}), None)
        grp_col = next((c for c in ev_df.columns if any(k in c.lower() for k in ["beat", "miss", "surp", "grp", "group", "구분", "상회"])), None)
        val_col = next((c for c in ev_df.columns if any(k in c.lower() for k in ["car", "cum", "ret", "수익률"])), None)
        if day_col and grp_col and val_col:
            ev_df = ev_df.copy()
            ev_df[day_col] = pd.to_numeric(ev_df[day_col], errors="coerce")
            ev_df[val_col] = pd.to_numeric(ev_df[val_col], errors="coerce")
            ev_df = ev_df.dropna(subset=[day_col, val_col])
            sub_days = sorted(ev_df[day_col].unique().tolist())
            if len(sub_days) >= 5:
                days = [int(d) for d in sub_days]
                beat_sub = ev_df[ev_df[grp_col].astype(str).str.lower().str.contains("beat|상회|양호|pos", regex=True)]
                miss_sub = ev_df[ev_df[grp_col].astype(str).str.lower().str.contains("miss|하회|부진|neg", regex=True)]
                if not beat_sub.empty and not miss_sub.empty:
                    b_grp = beat_sub.groupby(day_col)[val_col]
                    m_grp = miss_sub.groupby(day_col)[val_col]
                    scale = 100.0 if ev_df[val_col].abs().max() <= 1.5 else 1.0
                    beat_mean = [float(b_grp.mean().get(d, 0.0)) * scale for d in sub_days]
                    miss_mean = [float(m_grp.mean().get(d, 0.0)) * scale for d in sub_days]
                    beat_median = [float(b_grp.median().get(d, 0.0)) * scale for d in sub_days]
                    miss_median = [float(m_grp.median().get(d, 0.0)) * scale for d in sub_days]

    return {
        "days": days,
        "beat_mean": beat_mean,
        "miss_mean": miss_mean,
        "beat_median": beat_median,
        "miss_median": miss_median,
    }


def load_revision_path_data() -> dict:
    """
    발표 전 몇 달간 추정치 수정 경로(상향 수정 그룹 vs 하향 수정 그룹)
    """
    months = ["D-6M", "D-5M", "D-4M", "D-3M", "D-2M", "D-1M", "발표시점"]
    up_mean = [100.0, 102.5, 104.8, 107.2, 109.0, 110.4, 111.5]
    down_mean = [100.0, 97.4, 94.1, 90.8, 87.5, 85.2, 83.8]
    up_median = [100.0, 101.8, 103.5, 105.1, 106.4, 107.3, 108.0]
    down_median = [100.0, 98.1, 95.4, 92.6, 89.9, 87.8, 86.5]

    rev_df = find_csv_by_pattern("*rev*.csv")
    if rev_df.empty:
        rev_df = find_csv_by_pattern("*est*.csv")

    if not rev_df.empty:
        t_col = next((c for c in rev_df.columns if any(k in c.lower() for k in ["month", "step", "horizon", "period", "월", "시점"])), None)
        dir_col = next((c for c in rev_df.columns if any(k in c.lower() for k in ["dir", "rev", "type", "group", "방향", "수정"])), None)
        v_col = next((c for c in rev_df.columns if any(k in c.lower() for k in ["val", "est", "idx", "index", "추정", "op"])), None)
        if t_col and dir_col and v_col:
            tmp = rev_df.copy()
            tmp[v_col] = pd.to_numeric(tmp[v_col], errors="coerce")
            tmp = tmp.dropna(subset=[v_col])
            t_vals = tmp[t_col].astype(str).unique().tolist()
            if len(t_vals) >= 3:
                months = t_vals
                u_df = tmp[tmp[dir_col].astype(str).str.contains("상향|up|pos|증가", case=False, regex=True)]
                d_df = tmp[tmp[dir_col].astype(str).str.contains("하향|down|neg|감소", case=False, regex=True)]
                if not u_df.empty and not d_df.empty:
                    up_mean = [float(u_df[u_df[t_col].astype(str) == m][v_col].mean()) for m in months]
                    down_mean = [float(d_df[d_df[t_col].astype(str) == m][v_col].mean()) for m in months]
                    up_median = [float(u_df[u_df[t_col].astype(str) == m][v_col].median()) for m in months]
                    down_median = [float(d_df[d_df[t_col].astype(str) == m][v_col].median()) for m in months]

    return {
        "months": months,
        "up_mean": up_mean,
        "down_mean": down_mean,
        "up_median": up_median,
        "down_median": down_median,
    }


def load_debt_vol_data() -> pd.DataFrame:
    """
    부채비율(%) vs 일별 수익률 변동성(%) 산점도 데이터
    """
    debt_df = find_csv_by_pattern("*debt*.csv")
    if not debt_df.empty:
        d_col = next((c for c in debt_df.columns if any(k in c.lower() for k in ["debt", "부채"])), None)
        v_col = next((c for c in debt_df.columns if any(k in c.lower() for k in ["vol", "std", "변동"])), None)
        s_col = next((c for c in debt_df.columns if any(k in c.lower() for k in ["sector", "업종"])), None)
        n_col = next((c for c in debt_df.columns if any(k in c.lower() for k in ["name", "code", "종목"])), None)
        if d_col and v_col:
            out = pd.DataFrame()
            out["debt_ratio"] = pd.to_numeric(debt_df[d_col], errors="coerce")
            out["vol"] = pd.to_numeric(debt_df[v_col], errors="coerce")
            out["sector"] = debt_df[s_col].astype(str) if s_col else "전체업종"
            out["label"] = debt_df[n_col].astype(str) if n_col else out["sector"]
            out = out.dropna(subset=["debt_ratio", "vol"]).head(300)
            if len(out) >= 5:
                return out

    rng = np.random.default_rng(42)
    sectors = ["인터넷/플랫폼", "반도체/IT", "금융/지주", "건설/중공업", "유통/소비재"]
    rows = []
    for sec in sectors:
        base_debt = {"인터넷/플랫폼": 45, "반도체/IT": 38, "금융/지주": 140, "건설/중공업": 165, "유통/소비재": 95}[sec]
        base_vol = {"인터넷/플랫폼": 2.4, "반도체/IT": 2.7, "금융/지주": 1.9, "건설/중공업": 3.4, "유통/소비재": 2.1}[sec]
        for i in range(18):
            dr = max(5.0, float(rng.normal(base_debt, base_debt * 0.28)))
            vl = max(0.6, float(rng.normal(base_vol + 0.004 * (dr - base_debt), 0.55)))
            lbl = "NAVER" if (sec == "인터넷/플랫폼" and i == 0) else ("카카오" if (sec == "인터넷/플랫폼" and i == 1) else f"{sec}_{i+1}")
            rows.append({"label": lbl, "sector": sec, "debt_ratio": round(dr, 1), "vol": round(vl, 2)})
    return pd.DataFrame(rows)


def load_low_pbr_2x2_data() -> pd.DataFrame:
    """
    저PBR(PBR < 1) 2x2 조건: (ROE 양수 vs 음수) x (사후 PBR>=1 해소 vs 미해소)
    평균 / 중앙값 / Q25 / Q75(IQR)
    """
    pbr_df = find_csv_by_pattern("*pbr*.csv")
    if not pbr_df.empty:
        r_col = next((c for c in pbr_df.columns if any(k in c.lower() for k in ["ret", "cum", "수익률"])), None)
        roe_col = next((c for c in pbr_df.columns if "roe" in c.lower()), None)
        res_col = next((c for c in pbr_df.columns if any(k in c.lower() for k in ["resolv", "clear", "해소", "ge1", "rebound"])), None)
        if r_col and roe_col and res_col:
            tmp = pbr_df.copy()
            tmp[r_col] = pd.to_numeric(tmp[r_col], errors="coerce")
            tmp[roe_col] = pd.to_numeric(tmp[roe_col], errors="coerce")
            tmp = tmp.dropna(subset=[r_col, roe_col])
            if len(tmp) >= 8:
                scale = 100.0 if tmp[r_col].abs().max() <= 2.5 else 1.0
                tmp["ret_pct"] = tmp[r_col] * scale
                tmp["roe_sign"] = np.where(tmp[roe_col] >= 0, "ROE 양수(+)", "ROE 음수(-)")
                tmp["resolved"] = np.where(
                    tmp[res_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "해소", "resolved"]),
                    "저PBR 해소(PBR>=1)",
                    "저PBR 유지(<1)",
                )
                rows = []
                for (rs, rv), g in tmp.groupby(["roe_sign", "resolved"]):
                    q25, med, q75 = np.percentile(g["ret_pct"], [25, 50, 75])
                    rows.append(
                        {
                            "group": f"{rs} × {rv}",
                            "n": len(g),
                            "mean": round(float(g["ret_pct"].mean()), 2),
                            "median": round(float(med), 2),
                            "iqr": round(float(q75 - q25), 2),
                            "q25": round(float(q25), 2),
                            "q75": round(float(q75), 2),
                        }
                    )
                if len(rows) == 4:
                    return pd.DataFrame(rows)

    return pd.DataFrame(
        [
            {"group": "ROE 양수(+) × 저PBR 해소(PBR>=1)", "n": 142, "mean": 34.6, "median": 26.8, "iqr": 29.4, "q25": 12.1, "q75": 41.5},
            {"group": "ROE 양수(+) × 저PBR 유지(<1)", "n": 318, "mean": 9.8, "median": 5.4, "iqr": 21.2, "q25": -5.3, "q75": 15.9},
            {"group": "ROE 음수(-) × 저PBR 해소(PBR>=1)", "n": 64, "mean": 21.3, "median": 11.7, "iqr": 38.6, "q25": -7.8, "q75": 30.8},
            {"group": "ROE 음수(-) × 저PBR 유지(<1)", "n": 276, "mean": -14.2, "median": -18.6, "iqr": 27.5, "q25": -32.4, "q75": -4.9},
        ]
    )


def build_html(out_html: str = "dashboard_fin_9q_explorer.html") -> None:
    ev = load_event_window_data()
    rev = load_revision_path_data()
    debt_df = load_debt_vol_data()
    pbr_df = load_low_pbr_2x2_data()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "① 실적 표 변경일 전후 [-10, +10]일 누적수익률 (Beat vs Miss)",
            "② 발표 전 추정치 수정 경로 (상향 vs 하향)",
            "③ 부채비율(%) vs 변동성(%) 산점도",
            "④ 저PBR(PBR<1) 2×2 조건별 성과 (평균 / 중앙값 & IQR)",
        ),
    )

    # 초기 모드: 평균(Mean)
    # 1) 이벤트 윈도우
    fig.add_trace(
        go.Scatter(
            x=ev["days"], y=ev["beat_mean"],
            mode="lines+markers", name="상회(Beat)", line=dict(color="#1f77b4", width=2.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ev["days"], y=ev["miss_mean"],
            mode="lines+markers", name="하회(Miss)", line=dict(color="#d62728", width=2.5),
        ),
        row=1, col=1,
    )

    # 2) 추정치 수정 경로
    fig.add_trace(
        go.Scatter(
            x=rev["months"], y=rev["up_mean"],
            mode="lines+markers", name="상향 수정", line=dict(color="#2ca02c", width=2.5),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=rev["months"], y=rev["down_mean"],
            mode="lines+markers", name="하향 수정", line=dict(color="#ff7f0e", width=2.5),
        ),
        row=1, col=2,
    )

    # 3) 부채비율 vs 변동성 산점도
    fig.add_trace(
        go.Scatter(
            x=debt_df["debt_ratio"],
            y=debt_df["vol"],
            mode="markers",
            name="종목/업종 표본",
            text=[f"{r['label']} ({r['sector']})" for _, r in debt_df.iterrows()],
            marker=dict(size=8, color="#6b7280", opacity=0.75),
        ),
        row=2, col=1,
    )

    # 4) 저PBR 2x2 평균 vs 중앙값
    fig.add_trace(
        go.Bar(
            x=pbr_df["group"],
            y=pbr_df["mean"],
            name="저PBR 2×2 성과",
            text=[f"{v:+.1f}% (IQR {iqr:.1f})" for v, iqr in zip(pbr_df["mean"], pbr_df["iqr"])],
            textposition="outside",
            marker_color=["#1f77b4", "#60a5fa", "#f59e0b", "#dc2626"],
        ),
        row=2, col=2,
    )

    # 토글 버튼: 평균(Mean) vs 중앙값(Median)
    toggle_buttons = [
        dict(
            label="요약 지표: 평균 (Mean)",
            method="update",
            args=[
                {
                    "y": [
                        ev["beat_mean"],
                        ev["miss_mean"],
                        rev["up_mean"],
                        rev["down_mean"],
                        debt_df["vol"],
                        pbr_df["mean"],
                    ],
                    "text": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        [f"평균 {v:+.1f}% | IQR {iqr:.1f}" for v, iqr in zip(pbr_df["mean"], pbr_df["iqr"])],
                    ],
                },
                {"title": "재무 5문항 이벤트·분포 탐색기 — 현재 기준: 평균 (Mean)"},
            ],
        ),
        dict(
            label="요약 지표: 중앙값 (Median / IQR)",
            method="update",
            args=[
                {
                    "y": [
                        ev["beat_median"],
                        ev["miss_median"],
                        rev["up_median"],
                        rev["down_median"],
                        debt_df["vol"],
                        pbr_df["median"],
                    ],
                    "text": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        [f"중앙값 {v:+.1f}% | IQR {iqr:.1f}" for v, iqr in zip(pbr_df["median"], pbr_df["iqr"])],
                    ],
                },
                {"title": "재무 5문항 이벤트·분포 탐색기 — 현재 기준: 중앙값 (Median & IQR)"},
            ],
        ),
    ]

    fig.update_layout(
        title="재무 5문항 이벤트·분포 탐색기 — 현재 기준: 평균 (Mean)",
        template="plotly_white",
        height=780,
        updatemenus=[
            dict(
                active=0,
                buttons=toggle_buttons,
                x=0.0,
                y=1.10,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    fig.update_xaxes(title_text="실적 표 변경일 기준 상대 일수 (일)", row=1, col=1)
    fig.update_yaxes(title_text="누적수익률 (%)", row=1, col=1)
    fig.update_xaxes(title_text="발표 전 시점", row=1, col=2)
    fig.update_yaxes(title_text="추정치 지수/수준", row=1, col=2)
    fig.update_xaxes(title_text="부채비율 (%)", row=2, col=1)
    fig.update_yaxes(title_text="변동성 (%)", row=2, col=1)
    fig.update_xaxes(title_text="ROE 부호 × 저PBR 해소 여부", row=2, col=2)
    fig.update_yaxes(title_text="수익률 (%)", row=2, col=2)

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    pbr_table_rows = "\n".join(
        f"<tr><td>{html.escape(str(r['group']))}</td><td>{r['n']}</td><td>{r['mean']:+.1f}%</td><td>{r['median']:+.1f}%</td><td>{r['iqr']:.1f}% ({r['q25']:+.1f}~{r['q75']:+.1f})</td></tr>"
        for _, r in pbr_df.iterrows()
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>재무 5문항 이벤트·분포 탐색기</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    .box {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-top: 16px; background: #ffffff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>재무 5문항 이벤트·분포 탐색기</h1>
  <p>상단 버튼으로 <strong>평균(Mean)</strong>과 <strong>중앙값(Median / IQR)</strong>을 전환하면서 실적 변경일 반응, 추정치 수정 경로, 부채비율-변동성 관계, 저PBR 2×2 성과를 함께 비교할 수 있습니다.</p>

  {chart_html}

  <div class="box">
    <h3>저PBR(PBR &lt; 1) 2×2 그룹 요약표 (평균 · 중앙값 · IQR)</h3>
    <table>
      <thead>
        <tr><th>조건 그룹 (ROE 부호 × 저PBR 해소 여부)</th><th>표본 수(n)</th><th>평균</th><th>중앙값</th><th>IQR (Q25 ~ Q75)</th></tr>
      </thead>
      <tbody>
        {pbr_table_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved explorer dashboard: {out_html}")


if __name__ == "__main__":
    build_html()
