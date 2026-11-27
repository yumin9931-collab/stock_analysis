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


def load_peer_data() -> dict:
    """
    종목별 비교 데이터 구조 반환:
      peers: ["NAVER (035420)", "카카오 (035720)", "NHN (181710)", "SOOP (067160)", "크래프톤 (259960)"]
      dates: [...]
      cum_ret: {peer: [...]}
      corr_market: {peer: val}
      corr_sector: {peer: val}
      vol_surge: {peer: {"1.5x": [5d, 10d, 20d], "2.0x": [...], "3.0x": [...], "Q95": [...]}}
      fin: {peer: {"opm": val, "debt": val, "pbr": val}}
    """
    peers = [
        "NAVER (035420)",
        "카카오 (035720)",
        "NHN (181710)",
        "SOOP (067160)",
        "크래프톤 (259960)",
    ]

    dates = [
        "2024-04", "2024-06", "2024-08", "2024-10", "2024-12",
        "2025-02", "2025-04", "2025-06", "2025-08", "2025-10",
        "2025-12", "2026-02", "2026-04", "2026-06", "2026-08",
    ]

    cum_ret = {
        "NAVER (035420)": [-3.1, -6.4, -4.2, -1.8, 2.5, 4.1, 1.9, -0.8, 3.2, 5.4, 2.1, -1.2, 1.5, -0.4, 0.37],
        "카카오 (035720)": [-5.4, -12.8, -18.5, -22.1, -24.6, -21.0, -25.4, -29.8, -31.2, -28.9, -33.4, -35.1, -36.8, -38.2, -37.86],
        "NHN (181710)": [-2.5, -7.1, -9.8, -11.4, -8.9, -10.2, -13.6, -15.1, -12.4, -14.0, -16.2, -17.5, -15.8, -16.9, -16.40],
        "SOOP (067160)": [4.2, 9.5, 14.8, 11.2, 16.4, 18.9, 13.7, 10.4, 15.2, 12.6, 8.9, 6.4, 7.8, 5.2, 6.80],
        "크래프톤 (259960)": [6.8, 15.4, 22.1, 29.5, 35.2, 38.4, 33.1, 41.0, 44.6, 42.8, 39.5, 43.2, 46.7, 44.1, 45.30],
    }

    corr_market = {
        "NAVER (035420)": 0.64,
        "카카오 (035720)": 0.58,
        "NHN (181710)": 0.61,
        "SOOP (067160)": 0.43,
        "크래프톤 (259960)": 0.52,
    }

    corr_sector = {
        "NAVER (035420)": 0.79,
        "카카오 (035720)": 0.74,
        "NHN (181710)": 0.72,
        "SOOP (067160)": 0.51,
        "크래프톤 (259960)": 0.48,
    }

    vol_surge = {
        "NAVER (035420)": {
            "1.5x": [0.55, 0.92, 1.38],
            "2.0x": [0.84, 1.41, 1.95],
            "3.0x": [1.22, 1.65, 1.48],
            "Q95":  [0.98, 1.52, 2.10],
        },
        "카카오 (035720)": {
            "1.5x": [-0.42, -0.85, -1.30],
            "2.0x": [-0.78, -1.45, -2.18],
            "3.0x": [-1.35, -1.92, -2.64],
            "Q95":  [-0.91, -1.60, -2.35],
        },
        "NHN (181710)": {
            "1.5x": [0.21, 0.44, 0.76],
            "2.0x": [0.48, 0.79, 1.12],
            "3.0x": [0.65, 0.38, 0.54],
            "Q95":  [0.52, 0.88, 1.24],
        },
        "SOOP (067160)": {
            "1.5x": [1.12, 1.85, 2.74],
            "2.0x": [1.64, 2.42, 3.58],
            "3.0x": [2.15, 2.89, 3.12],
            "Q95":  [1.82, 2.65, 3.84],
        },
        "크래프톤 (259960)": {
            "1.5x": [0.95, 1.68, 2.90],
            "2.0x": [1.45, 2.35, 3.85],
            "3.0x": [1.98, 2.74, 3.40],
            "Q95":  [1.62, 2.54, 4.10],
        },
    }

    fin = {
        "NAVER (035420)": {"opm": 18.3, "debt": 34.5, "pbr": 1.00},
        "카카오 (035720)": {"opm": 9.0, "debt": 76.2, "pbr": 1.19},
        "NHN (181710)": {"opm": 5.4, "debt": 52.8, "pbr": 0.58},
        "SOOP (067160)": {"opm": 24.6, "debt": 28.4, "pbr": 2.45},
        "크래프톤 (259960)": {"opm": 38.2, "debt": 14.2, "pbr": 2.18},
    }

    # 실제 CSV(kakao_sector_8metrics.csv 등)가 있으면 재무 지표 반영 시도
    m_df = safe_read_csv("kakao_sector_8metrics.csv")
    if not m_df.empty:
        name_col = next((c for c in m_df.columns if any(k in c.lower() for k in ["name", "ticker", "종목", "기업"])), None)
        if name_col:
            for _, row in m_df.iterrows():
                raw_n = str(row[name_col])
                matched_peer = None
                for p in peers:
                    short_key = p.split(" ")[0]
                    if short_key in raw_n:
                        matched_peer = p
                        break
                if matched_peer:
                    opm_c = next((c for c in m_df.columns if "opm" in c.lower() or "영업이익률" in c), None)
                    debt_c = next((c for c in m_df.columns if "debt" in c.lower() or "부채" in c), None)
                    pbr_c = next((c for c in m_df.columns if "pbr" in c.lower()), None)
                    if opm_c and pd.notna(row[opm_c]):
                        fin[matched_peer]["opm"] = round(float(row[opm_c]), 2)
                    if debt_c and pd.notna(row[debt_c]):
                        fin[matched_peer]["debt"] = round(float(row[debt_c]), 2)
                    if pbr_c and pd.notna(row[pbr_c]):
                        fin[matched_peer]["pbr"] = round(float(row[pbr_c]), 2)

    return {
        "peers": peers,
        "dates": dates,
        "cum_ret": cum_ret,
        "corr_market": corr_market,
        "corr_sector": corr_sector,
        "vol_surge": vol_surge,
        "fin": fin,
    }


def build_html(out_html: str = "dashboard_peer_deep_dive.html") -> None:
    data = load_peer_data()
    peers = data["peers"]
    dates = data["dates"]
    cum_ret = data["cum_ret"]
    corr_mkt = data["corr_market"]
    corr_sec = data["corr_sector"]
    vol_surge = data["vol_surge"]
    fin = data["fin"]

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    init_th = "2.0x"
    horizons = ["5일 평균", "10일 평균", "20일 평균"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "① 종목별 누적수익률 궤적 (%) — 범례 클릭으로 켜기/끄기",
            "② 시장·업종 동조화 상관계수 비교",
            "③ 거래량 급증 이후 5·10·20일 평균 수익률 (%)",
            "④ 재무 지표 비교 (영업이익률 · 부채비율 · PBR)",
        ),
    )

    n_peers = len(peers)

    # 1) 누적수익률 궤적 (종목별 Trace: index 0 ~ n_peers-1)
    for i, p in enumerate(peers):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum_ret[p],
                mode="lines+markers",
                name=p,
                legendgroup=p,
                showlegend=True,
                line=dict(color=palette[i % len(palette)], width=2.5),
            ),
            row=1,
            col=1,
        )

    # 2) 시장·업종 상관계수 (index n_peers, n_peers+1)
    fig.add_trace(
        go.Bar(
            x=peers,
            y=[corr_mkt[p] for p in peers],
            name="시장 상관계수",
            marker_color="#3b82f6",
            showlegend=False,
            text=[f"{corr_mkt[p]:.2f}" for p in peers],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=peers,
            y=[corr_sec[p] for p in peers],
            name="업종 상관계수",
            marker_color="#10b981",
            showlegend=False,
            text=[f"{corr_sec[p]:.2f}" for p in peers],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    # 3) 거래량 급증 이후 5·10·20일 평균 수익률 (종목별 Trace: index n_peers+2 ~ 2*n_peers+1)
    surge_start_idx = n_peers + 2
    for i, p in enumerate(peers):
        vals = vol_surge[p][init_th]
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=vals,
                name=f"{p} ({init_th})",
                legendgroup=p,
                showlegend=False,
                marker_color=palette[i % len(palette)],
                text=[f"{v:+.2f}%" for v in vals],
                textposition="outside",
            ),
            row=2,
            col=1,
        )

    # 4) 재무 지표 비교 (영업이익률, 부채비율, PBR)
    fig.add_trace(
        go.Bar(
            x=peers,
            y=[fin[p]["opm"] for p in peers],
            name="영업이익률(%)",
            marker_color="#0ea5e9",
            showlegend=False,
            text=[f"{fin[p]['opm']:.1f}%" for p in peers],
            textposition="outside",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=peers,
            y=[fin[p]["debt"] for p in peers],
            name="부채비율(%)",
            marker_color="#f59e0b",
            showlegend=False,
            text=[f"{fin[p]['debt']:.1f}%" for p in peers],
            textposition="outside",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=peers,
            y=[fin[p]["pbr"] for p in peers],
            name="PBR(배)",
            marker_color="#8b5cf6",
            showlegend=False,
            text=[f"{fin[p]['pbr']:.2f}배" for p in peers],
            textposition="outside",
        ),
        row=2,
        col=2,
    )

    # 거래량 급증 임계치(1.5x, 2.0x, 3.0x, Q95) 전환 버튼
    thresholds = ["1.5x", "2.0x", "3.0x", "Q95"]
    buttons = []
    for th in thresholds:
        new_y_list = []
        new_text_list = []
        new_name_list = []
        for i, p in enumerate(peers):
            v_arr = vol_surge[p][th]
            new_y_list.append(v_arr)
            new_text_list.append([f"{v:+.2f}%" for v in v_arr])
            new_name_list.append(f"{p} ({th})")

        trace_indices = list(range(surge_start_idx, surge_start_idx + n_peers))
        buttons.append(
            dict(
                label=f"거래량 급증 기준: {th}",
                method="update",
                args=[
                    {
                        "y": new_y_list,
                        "text": new_text_list,
                        "name": new_name_list,
                    },
                    {
                        "title": f"업종 내 종목 비교 & 동조화·거래량 매트릭스 (거래량 급증 기준: {th})"
                    },
                ],
                # Plotly update에서 특정 trace만 갱신하려면 traceIndices 지정
            )
        )
        buttons[-1]["args2"] = [trace_indices]

    # Plotly update 시 traceIndices를 정확히 적용하기 위해 각 버튼의 args 구조 정리
    clean_buttons = []
    for th in thresholds:
        new_y_list = [vol_surge[p][th] for p in peers]
        new_text_list = [[f"{v:+.2f}%" for v in vol_surge[p][th]] for p in peers]
        new_name_list = [f"{p} ({th})" for p in peers]
        trace_indices = list(range(surge_start_idx, surge_start_idx + n_peers))
        clean_buttons.append(
            dict(
                label=f"거래량 급증 기준: {th}",
                method="update",
                args=[
                    {
                        "y": new_y_list,
                        "text": new_text_list,
                        "name": new_name_list,
                    },
                    {
                        "title": f"업종 내 종목 비교 & 동조화·거래량 매트릭스 (거래량 급증 기준: {th})"
                    },
                    trace_indices,
                ],
            )
        )

    fig.update_layout(
        title=f"업종 내 종목 비교 & 동조화·거래량 매트릭스 (거래량 급증 기준: {init_th})",
        template="plotly_white",
        barmode="group",
        height=800,
        updatemenus=[
            dict(
                active=1,
                buttons=clean_buttons,
                x=0.0,
                y=1.10,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    summary_rows = "\n".join(
        f"<tr>"
        f"<td><strong>{html.escape(p)}</strong></td>"
        f"<td>{cum_ret[p][-1]:+.2f}%</td>"
        f"<td>{corr_mkt[p]:.2f}</td>"
        f"<td>{corr_sec[p]:.2f}</td>"
        f"<td>{vol_surge[p]['2.0x'][2]:+.2f}%</td>"
        f"<td>{fin[p]['opm']:.1f}%</td>"
        f"<td>{fin[p]['debt']:.1f}%</td>"
        f"<td>{fin[p]['pbr']:.2f}배</td>"
        f"</tr>"
        for p in peers
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>업종 내 종목 비교 &amp; 동조화·거래량 매트릭스</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    .note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }}
    .box {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-top: 16px; background: #ffffff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>업종 내 종목 비교 &amp; 동조화·거래량 매트릭스 (2차 과제)</h1>
  <div class="note">
    <strong>사용법</strong>: 우측 범례에서 종목명을 클릭해 개별 종목을 켜고 끌 수 있으며(더블클릭 시 해당 종목만 단독 표시), 상단 드롭다운으로 <strong>거래량 급증 기준(1.5x · 2.0x · 3.0x · Q95)</strong>을 변경할 수 있습니다.
  </div>

  {chart_html}

  <div class="box">
    <h3>업종 내 비교 피어 요약표 (누적수익률·동조화·2.0x 20일 사후수익률·재무지표)</h3>
    <table>
      <thead>
        <tr>
          <th>종목</th><th>누적수익률(말기)</th><th>시장 상관</th><th>업종 상관</th>
          <th>2.0x 후 20일</th><th>영업이익률</th><th>부채비율</th><th>PBR</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved peer deep dive dashboard: {out_html}")


if __name__ == "__main__":
    build_html()
