from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RES_DIR = "/home/s202423331/results_spec1_spec2"
OUT_HTML = "/home/s202423331/dashboard_spec1_spec2.html"


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RES_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_spec1_spec2_dashboard() -> str:
    nk_df = _load_csv("spec1_naver_kakao_focus")
    comp_df = _load_csv("spec1_sector_comparison")
    disp_m = _load_csv("spec2_dispersion_monthly")
    spread_d = _load_csv("spec2_spread_daily")
    tb_df = _load_csv("spec2_top_bottom_sectors")

    panels: list[tuple[str, str, go.Figure]] = []

    # 1. NAVER & 카카오 vs 소속 업종 누적수익률
    if not nk_df.empty and {"name", "stock_cum_ret"}.issubset(nk_df.columns):
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=nk_df["name"],
                y=nk_df["stock_cum_ret"] * 100.0,
                name="종목 누적수익률 (%)",
                text=[f"{v*100:.2f}%" for v in nk_df["stock_cum_ret"]],
                textposition="outside",
                marker_color="#1d4ed8",
            )
        )
        if "sector_cum_ret_mcap_w" in nk_df.columns:
            fig1.add_trace(
                go.Bar(
                    x=nk_df["name"],
                    y=nk_df["sector_cum_ret_mcap_w"] * 100.0,
                    name="업종 누적수익률 (시총가중, %)",
                    text=[f"{v*100:.2f}%" if pd.notna(v) else "N/A" for v in nk_df["sector_cum_ret_mcap_w"]],
                    textposition="outside",
                    marker_color="#10b981",
                )
            )
        if "sector_cum_ret_equal_w" in nk_df.columns:
            fig1.add_trace(
                go.Bar(
                    x=nk_df["name"],
                    y=nk_df["sector_cum_ret_equal_w"] * 100.0,
                    name="업종 누적수익률 (동일가중, %)",
                    text=[f"{v*100:.2f}%" if pd.notna(v) else "N/A" for v in nk_df["sector_cum_ret_equal_w"]],
                    textposition="outside",
                    marker_color="#f59e0b",
                )
            )
        fig1.update_layout(
            title="1. NAVER·카카오 및 소속 업종 누적수익률 비교",
            yaxis_title="누적수익률 (%)",
            barmode="group",
            hovermode="x unified",
        )
        panels.append(("1_nk_vs_sector", "NAVER·카카오 vs 업종", fig1))

    # 2. 시총가중 vs 동일가중 괴리 (상위/하위 12개 업종)
    if not comp_df.empty and {"sector", "diff_mcap_minus_ew"}.issubset(comp_df.columns):
        d2 = comp_df.dropna(subset=["diff_mcap_minus_ew"]).sort_values("diff_mcap_minus_ew", ascending=False)
        d2_sub = pd.concat([d2.head(8), d2.tail(8)]).drop_duplicates("sector")
        fig2 = go.Figure(
            go.Bar(
                x=d2_sub["sector"],
                y=d2_sub["diff_mcap_minus_ew"] * 100.0,
                name="시총가중 - 동일가중 (%p)",
                marker_color=np.where(d2_sub["diff_mcap_minus_ew"] >= 0, "#2563eb", "#ef4444"),
                hovertemplate="업종: %{x}<br>괴리(시총가중-동일가중): %{y:.2f}%p<extra></extra>",
            )
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig2.update_layout(
            title="2. 업종별 시가총액 가중 vs 동일 가중 누적수익률 차이 (대형주 편중 확인)",
            xaxis_title="업종",
            yaxis_title="차이 (%p)",
            hovermode="x unified",
        )
        panels.append(("2_mcap_vs_ew", "시총가중 vs 동일가중", fig2))

    # 3. 월별 업종 간 분산 및 민감도(STD, IQR, MAD, 초과수익률 STD)
    if not disp_m.empty and {"month", "std_mcap_w"}.issubset(disp_m.columns):
        fig3 = go.Figure()
        series_map = [
            ("std_mcap_w", "월별 표준편차 (시총가중)", "#1d4ed8", "solid"),
            ("std_excess_vs_mkt", "월별 표준편차 (시장 초과수익률)", "#16a34a", "dash"),
            ("iqr_mcap_w", "월별 사분위범위 (IQR)", "#f59e0b", "dot"),
            ("mad_mcap_w", "월별 중앙절대편차 (MAD)", "#9333ea", "dashdot"),
        ]
        for col, label, color, dash_style in series_map:
            if col in disp_m.columns:
                fig3.add_trace(
                    go.Scatter(
                        x=disp_m["month"],
                        y=disp_m[col] * 100.0,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2, dash=dash_style),
                        hovertemplate="%{x}<br>" + label + ": %{y:.2f}%p<extra></extra>",
                    )
                )
        fig3.update_layout(
            title="3. 월별 업종 간 수익률 산포 및 민감도 비교",
            xaxis_title="월 (YYYY-MM)",
            yaxis_title="산포 척도 (%p)",
            hovermode="x unified",
        )
        panels.append(("3_monthly_disp", "월별 분산·민감도", fig3))

    # 4. 누적수익률 퍼짐 추이 (표준편차 및 최고-최저 범위)
    if not spread_d.empty and {"date", "cum_std_mcap_w"}.issubset(spread_d.columns):
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(
            go.Scatter(
                x=spread_d["date"],
                y=spread_d["cum_std_mcap_w"] * 100.0,
                mode="lines",
                name="누적수익률 횡단면 표준편차 (%p)",
                line=dict(color="#dc2626", width=2.5),
                hovertemplate="날짜: %{x}<br>표준편차: %{y:.2f}%p<extra></extra>",
            ),
            secondary_y=False,
        )
        if "cum_range_mcap_w" in spread_d.columns:
            fig4.add_trace(
                go.Scatter(
                    x=spread_d["date"],
                    y=spread_d["cum_range_mcap_w"] * 100.0,
                    mode="lines",
                    name="최고-최저 범위 (%p)",
                    line=dict(color="#0ea5e9", width=1.8, dash="dot"),
                    hovertemplate="날짜: %{x}<br>최고-최저 범위: %{y:.2f}%p<extra></extra>",
                ),
                secondary_y=True,
            )
        fig4.update_yaxes(title_text="누적수익률 표준편차 (%p)", secondary_y=False)
        fig4.update_yaxes(title_text="최고-최저 범위 (%p)", secondary_y=True)
        fig4.update_layout(
            title="4. 업종별 누적수익률 퍼짐 추이 (벌어졌는가, 좁혀졌는가)",
            xaxis_title="날짜",
            hovermode="x unified",
        )
        panels.append(("4_cum_spread", "누적수익률 퍼짐 추이", fig4))

    # 5. 격차를 만든 상위/하위 업종
    if not tb_df.empty and {"sector", "diff_from_sector_mean"}.issubset(tb_df.columns):
        d5 = pd.concat([tb_df.head(7), tb_df.tail(7)]).drop_duplicates("sector")
        fig5 = go.Figure(
            go.Bar(
                x=d5["sector"],
                y=d5["diff_from_sector_mean"] * 100.0,
                name="업종 평균 대비 누적수익률 차이 (%p)",
                marker_color=np.where(d5["diff_from_sector_mean"] >= 0, "#059669", "#e11d48"),
                hovertemplate="업종: %{x}<br>업종평균 대비: %{y:.2f}%p<extra></extra>",
            )
        )
        fig5.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig5.update_layout(
            title="5. 누적수익률 격차를 만든 상위·하위 업종 (업종 평균 대비)",
            xaxis_title="업종",
            yaxis_title="평균 대비 차이 (%p)",
            hovermode="x unified",
        )
        panels.append(("5_top_bottom", "격차 상위·하위 업종", fig5))

    buttons_html = []
    divs_html = []
    for idx, (tab_id, tab_title, fig) in enumerate(panels):
        active_cls = "active" if idx == 0 else ""
        disp_style = "block" if idx == 0 else "none"
        buttons_html.append(
            f'<button class="tab-btn {active_cls}" onclick="showTab(\'{tab_id}\', this)">{tab_title}</button>'
        )
        chart_div = fig.to_html(full_html=False, include_plotlyjs=("cdn" if idx == 0 else False))
        divs_html.append(
            f'<div id="{tab_id}" class="tab-panel" style="display:{disp_style};">{chart_div}</div>'
        )

    if not panels:
        divs_html.append(
            "<p style='padding:20px;font-family:sans-serif;'>표시할 CSV 데이터가 아직 없습니다. "
            "먼저 <code>python3 run_spec1_spec2_naver_kakao.py</code>를 실행해 주세요.</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>견본 1·2 업종 수익률 및 분산 인터랙티브 대시보드</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif; margin: 20px; background: #f8fafc; color: #0f172a; }}
  h1 {{ font-size: 20px; margin-bottom: 12px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
  .tab-btn {{ padding: 8px 14px; border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; cursor: pointer; font-size: 13px; }}
  .tab-btn.active {{ background: #1d4ed8; color: #ffffff; border-color: #1d4ed8; font-weight: 600; }}
  .tab-panel {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
</style>
</head>
<body>
  <h1>견본 1(업종 수익률) · 견본 2(업종 간 분산) 인터랙티브 대시보드</h1>
  <div class="tabs">{''.join(buttons_html)}</div>
  {''.join(divs_html)}
  <script>
    function showTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-panel').forEach(el => el.style.display = 'none');
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      const target = document.getElementById(tabId);
      if (target) {{
        target.style.display = 'block';
        target.querySelectorAll('.js-plotly-plot').forEach(p => {{
          if (window.Plotly) Plotly.Plots.resize(p);
        }});
      }}
      if (btnEl) btnEl.classList.add('active');
    }}
  </script>
</body>
</html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    return OUT_HTML


if __name__ == "__main__":
    out_file = build_spec1_spec2_dashboard()
    print(f"인터랙티브 시각화 생성 완료: {out_file}")
