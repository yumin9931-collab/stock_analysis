from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RES_DIR = "/home/s202423331/results_5fin_themes"
OUT_HTML = "/home/s202423331/dashboard_5fin_themes.html"


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RES_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_interactive_dashboard() -> str:
    q1_sec = _load_csv("q1_sector_opm_ret")
    q1_quad = _load_csv("q1_quadrant_2x2")
    q2_win = _load_csv("q2_event_window")
    q3_err = _load_csv("q3_sector_error")
    q3_rev = _load_csv("q3_revision_path")
    q4_sec = _load_csv("q4_sector_debt_vol")
    q4_corr = _load_csv("q4_corr_summary")
    q5_roe = _load_csv("q5_roe_pos_vs_neg")
    q5_res = _load_csv("q5_resolved_vs_unresolved")
    q5_conc = _load_csv("q5_sector_conc")

    panels: list[tuple[str, str, go.Figure]] = []

    # 1. 영업이익률 vs 누적 수익률 (상위 15개 업종)
    if not q1_sec.empty and {"sector", "sec_opm_mean", "sec_cum_ret"}.issubset(q1_sec.columns):
        d1 = q1_sec.head(15).copy()
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(
            go.Bar(
                x=d1["sector"],
                y=d1["sec_opm_mean"],
                name="평균 영업이익률 (%)",
                marker_color="#2563eb",
                hovertemplate="업종: %{x}<br>평균 OPM: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(
                x=d1["sector"],
                y=d1["sec_cum_ret"] * 100.0,
                name="누적 수익률 (%)",
                mode="lines+markers",
                marker=dict(size=8, color="#dc2626"),
                line=dict(color="#dc2626", width=2.5),
                hovertemplate="업종: %{x}<br>누적 수익률: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        fig1.update_yaxes(title_text="평균 영업이익률 (%)", secondary_y=False)
        fig1.update_yaxes(title_text="누적 수익률 (%)", secondary_y=True)
        fig1.update_layout(title="1. 업종별 영업이익률 vs 누적 수익률 (상위 15개)", hovermode="x unified")
        panels.append(("1_OPM_vs_수익률", "영업이익률과 누적 수익률 관계", fig1))

    # 2. 실적 발표 전후 10일 누적 수익률 (Beat vs Miss)
    if not q2_win.empty and {"beat", "tau", "mean_cum_ret_from_m1"}.issubset(q2_win.columns):
        fig2 = go.Figure()
        for beat_val, grp in q2_win.groupby("beat"):
            g = grp.sort_values("tau")
            label = "실적 서프라이즈 (Beat)" if str(beat_val).lower() == "true" else "실적 하회 (Miss)"
            fig2.add_trace(
                go.Scatter(
                    x=g["tau"],
                    y=g["mean_cum_ret_from_m1"] * 100.0,
                    mode="lines+markers",
                    name=label,
                    hovertemplate="D%{x:+d}일<br>누적수익률: %{y:.2f}%<extra></extra>",
                )
            )
        fig2.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="발표일(D0)")
        fig2.update_layout(
            title="2. 실적 발표 전후 [-10, +10]일 평균 누적 수익률",
            xaxis_title="발표일 기준 상대 거래일 (tau)",
            yaxis_title="누적 수익률 (%)",
            hovermode="x unified",
        )
        panels.append(("2_실적반응_10일", "실적 발표 전후 주가 반응", fig2))

    # 3. 업종별 추정 오차 및 발표 전 수정 경로
    if not q3_err.empty and {"sector", "mean_abs_error", "mean_signed_error"}.issubset(q3_err.columns):
        d3 = q3_err.head(15).copy()
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=d3["sector"],
                y=d3["mean_abs_error"] * 100.0,
                name="평균 절대오차 (|실적-추정|/|실적|, %)",
                marker_color="#f59e0b",
                hovertemplate="업종: %{x}<br>평균 절대오차: %{y:.2f}%<extra></extra>",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=d3["sector"],
                y=d3["mean_signed_error"] * 100.0,
                name="부호 오차 ((실적-추정)/|실적|, %)",
                mode="lines+markers",
                marker=dict(size=8, color="#10b981"),
                hovertemplate="업종: %{x}<br>부호 오차: %{y:.2f}%<extra></extra>",
            )
        )
        fig3.update_layout(
            title="3. 업종별 영업이익 추정 오차 분포",
            xaxis_title="업종",
            yaxis_title="오차율 (%)",
            hovermode="x unified",
        )
        panels.append(("3_추정오차", "업종별 추정 오차 및 방향성", fig3))

    if not q3_rev.empty and {"months_before", "mean_est_bias_pct"}.issubset(q3_rev.columns):
        r3 = q3_rev.sort_values("months_before", ascending=False)
        fig3b = go.Figure(
            go.Scatter(
                x=r3["months_before"],
                y=r3["mean_est_bias_pct"] * 100.0,
                mode="lines+markers",
                name="추정치 편향 ((추정-실적)/|실적|, %)",
                line=dict(color="#6366f1", width=2.5),
                hovertemplate="발표 %{x}개월 전<br>추정치 편향: %{y:.2f}%<extra></extra>",
            )
        )
        fig3b.update_layout(
            title="3-B. 발표 전 개월수별 추정치 수정 경로",
            xaxis_title="발표 전 개월수 (months_before)",
            yaxis_title="추정치 편향 (%)",
            hovermode="x unified",
        )
        panels.append(("3_추정수정경로", "발표 전 추정치 수정 경로", fig3b))

    # 4. 부채비율과 변동성 (전체 / 급락 구간 / 보통 구간)
    if not q4_sec.empty and {"sector", "sec_debt_ratio", "sec_ann_vol"}.issubset(q4_sec.columns):
        d4 = q4_sec.head(15).copy()
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(
            go.Bar(
                x=d4["sector"],
                y=d4["sec_debt_ratio"],
                name="평균 부채비율 (%)",
                marker_color="#64748b",
                hovertemplate="업종: %{x}<br>부채비율: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=False,
        )
        fig4.add_trace(
            go.Scatter(
                x=d4["sector"],
                y=d4["sec_ann_vol"] * 100.0,
                name="전체 연환산 변동성 (%)",
                mode="lines+markers",
                marker=dict(size=8, color="#e11d48"),
                hovertemplate="업종: %{x}<br>전체 변동성: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        if "sec_vol_crash" in d4.columns:
            fig4.add_trace(
                go.Scatter(
                    x=d4["sector"],
                    y=d4["sec_vol_crash"] * 100.0,
                    name="급락일 연환산 변동성 (%)",
                    mode="markers",
                    marker=dict(size=9, symbol="diamond", color="#111827"),
                    hovertemplate="업종: %{x}<br>급락일 변동성: %{y:.2f}%<extra></extra>",
                ),
                secondary_y=True,
            )
        fig4.update_yaxes(title_text="부채비율 (%)", secondary_y=False)
        fig4.update_yaxes(title_text="연환산 변동성 (%)", secondary_y=True)
        fig4.update_layout(title="4. 업종별 부채비율 및 변동성 비교", hovermode="x unified")
        panels.append(("4_부채_변동성", "부채비율과 변동성 관계", fig4))

    # 5. 저PBR(PBR < 1) 종목 분석 (ROE 양/음수 및 해소 여부)
    if not q5_roe.empty and {"roe_positive", "mean_cum_ret"}.issubset(q5_roe.columns):
        r5 = q5_roe.copy()
        r5["label"] = r5["roe_positive"].map({True: "ROE > 0 (양수)", False: "ROE <= 0 (음수/0)"})
        fig5a = go.Figure(
            go.Bar(
                x=r5["label"],
                y=r5["mean_cum_ret"] * 100.0,
                text=[f"{v*100:.2f}%" for v in r5["mean_cum_ret"]],
                textposition="outside",
                marker_color=["#0ea5e9", "#f97316"],
            )
        )
        fig5a.update_layout(
            title="5-A. 저PBR(PBR < 1) 종목: ROE 양수 vs 음수 평균 누적수익률",
            yaxis_title="평균 누적수익률 (%)",
        )
        panels.append(("5_저PBR_ROE", "저PBR 종목 ROE 양/음수 비교", fig5a))

    if not q5_res.empty and {"pbr_resolved", "mean_cum_ret"}.issubset(q5_res.columns):
        s5 = q5_res.copy()
        s5["label"] = s5["pbr_resolved"].map({True: "저PBR 해소 (종료 PBR >= 1)", False: "저PBR 미해소 (종료 PBR < 1)"})
        fig5b = go.Figure(
            go.Bar(
                x=s5["label"],
                y=s5["mean_cum_ret"] * 100.0,
                text=[f"{v*100:.2f}%" for v in s5["mean_cum_ret"]],
                textposition="outside",
                marker_color=["#16a34a", "#94a3b8"],
            )
        )
        fig5b.update_layout(
            title="5-B. 저PBR 해소 여부별 평균 누적수익률",
            yaxis_title="평균 누적수익률 (%)",
        )
        panels.append(("5_저PBR_해소", "저PBR 해소 vs 미해소 비교", fig5b))

    if not q5_conc.empty and {"sector", "low_pbr_count"}.issubset(q5_conc.columns):
        c5 = q5_conc.head(15).copy()
        fig5c = go.Figure(
            go.Bar(
                x=c5["sector"],
                y=c5["low_pbr_count"],
                name="초기 저PBR(PBR < 1) 종목수",
                marker_color="#8b5cf6",
            )
        )
        fig5c.update_layout(
            title="5-C. 업종별 초기 저PBR(PBR < 1) 종목 분포",
            xaxis_title="업종",
            yaxis_title="종목수",
        )
        panels.append(("5_저PBR_업종", "저PBR 종목 업종 분포", fig5c))

    # 탭형 인터랙티브 HTML 조립
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
            "<p style='padding:20px;font-family:sans-serif;'>시각화할 데이터가 아직 충분하지 않습니다. "
            "~/fin_parquet에 대상 종목/기간 데이터가 채워진 뒤 다시 실행해 주세요.</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>5대 재무·수익률 테마 인터랙티브 대시보드</title>
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
  <h1>재무·수익률·변동성·저PBR 인터랙티브 대시보드</h1>
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
    out_path = build_interactive_dashboard()
    print(f"인터랙티브 대시보드 생성 완료: {out_path}")
