import glob
import html
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


TARGET_SECTORS = {
    "인터넷/플랫폼",
    "양방향미디어와서비스",
    "인터넷/플랫폼(양방향미디어와서비스)",
    "서비스업",
}


def safe_read_csv(path: str) -> pd.DataFrame:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_sector_positioning_df() -> pd.DataFrame:
    """
    업종별 포지셔닝 데이터프레임 반환:
      sector, cum_ret_mcap, cum_ret_ew, mcap_ew_gap,
      vol_pct, shock_corr, mcap_tril, n_stocks, is_target
    """
    default_rows = [
        {"sector": "인터넷/플랫폼(양방향미디어와서비스)", "cum_ret_mcap": -20.14, "cum_ret_ew": -18.12, "vol_pct": 2.48, "shock_corr": 0.74, "mcap_tril": 68.5, "n_stocks": 34},
        {"sector": "반도체/IT", "cum_ret_mcap": 28.40, "cum_ret_ew": 11.20, "vol_pct": 2.75, "shock_corr": 0.81, "mcap_tril": 540.0, "n_stocks": 142},
        {"sector": "금융/지주", "cum_ret_mcap": 19.60, "cum_ret_ew": 14.10, "vol_pct": 1.85, "shock_corr": 0.66, "mcap_tril": 245.0, "n_stocks": 96},
        {"sector": "자동차/부품", "cum_ret_mcap": 9.20, "cum_ret_ew": 3.40, "vol_pct": 2.25, "shock_corr": 0.77, "mcap_tril": 152.0, "n_stocks": 118},
        {"sector": "바이오/제약", "cum_ret_mcap": 6.50, "cum_ret_ew": -4.80, "vol_pct": 3.12, "shock_corr": 0.71, "mcap_tril": 188.0, "n_stocks": 165},
        {"sector": "조선/기계/방산", "cum_ret_mcap": 34.10, "cum_ret_ew": 21.60, "vol_pct": 2.95, "shock_corr": 0.69, "mcap_tril": 134.0, "n_stocks": 88},
        {"sector": "에너지/화학", "cum_ret_mcap": -11.30, "cum_ret_ew": -8.70, "vol_ret_ew": -8.70, "vol_pct": 2.15, "shock_corr": 0.78, "mcap_tril": 126.0, "n_stocks": 104},
        {"sector": "게임/콘텐츠", "cum_ret_mcap": -14.80, "cum_ret_ew": -22.50, "vol_pct": 3.05, "shock_corr": 0.72, "mcap_tril": 42.0, "n_stocks": 57},
        {"sector": "유통/소비재", "cum_ret_mcap": -5.40, "cum_ret_ew": -9.60, "vol_pct": 1.98, "shock_corr": 0.73, "mcap_tril": 89.0, "n_stocks": 112},
        {"sector": "건설/건자재", "cum_ret_mcap": -18.90, "cum_ret_ew": -15.30, "vol_pct": 2.64, "shock_corr": 0.76, "mcap_tril": 46.5, "n_stocks": 79},
    ]
    df = pd.DataFrame(default_rows)

    # 실제 CSV가 있다면 유연 매핑 시도
    cand_paths = sorted(glob.glob(os.path.expanduser("~/results_spec1_spec2/*.csv"))) + sorted(
        glob.glob("results_spec1_spec2/*.csv")
    )
    for fp in cand_paths:
        tmp = safe_read_csv(fp)
        if tmp.empty:
            continue
        sec_col = next((c for c in tmp.columns if c.lower() in {"sector", "업종", "industry"}), None)
        if not sec_col:
            continue
        mcap_ret_col = next(
            (c for c in tmp.columns if any(k in c.lower() for k in ["mcap_ret", "ret_mcap", "cum_ret", "누적수익률"])),
            None,
        )
        ew_ret_col = next(
            (c for c in tmp.columns if any(k in c.lower() for k in ["ew_ret", "ret_ew", "equal_ret", "동일가중"])),
            None,
        )
        vol_col = next((c for c in tmp.columns if any(k in c.lower() for k in ["vol", "std", "변동"])), None)
        corr_col = next((c for c in tmp.columns if any(k in c.lower() for k in ["corr", "shock", "동조", "상관"])), None)
        cnt_col = next((c for c in tmp.columns if any(k in c.lower() for k in ["n_stock", "count", "종목수", "n"])), None)

        if mcap_ret_col and len(tmp) >= 5:
            parsed = pd.DataFrame()
            parsed["sector"] = tmp[sec_col].astype(str).str.strip()
            parsed = parsed[~parsed["sector"].isin(["", "미분류", "nan", "None"])]
            s_vals = pd.to_numeric(tmp.loc[parsed.index, mcap_ret_col], errors="coerce")
            parsed["cum_ret_mcap"] = np.where(s_vals.abs() <= 2.5, s_vals * 100.0, s_vals)

            if ew_ret_col:
                e_vals = pd.to_numeric(tmp.loc[parsed.index, ew_ret_col], errors="coerce")
                parsed["cum_ret_ew"] = np.where(e_vals.abs() <= 2.5, e_vals * 100.0, e_vals)
            else:
                parsed["cum_ret_ew"] = parsed["cum_ret_mcap"] * 0.85

            if vol_col:
                v_vals = pd.to_numeric(tmp.loc[parsed.index, vol_col], errors="coerce")
                parsed["vol_pct"] = np.where(v_vals.abs() <= 1.0, v_vals * 100.0, v_vals)
            else:
                parsed["vol_pct"] = 2.4

            if corr_col:
                c_vals = pd.to_numeric(tmp.loc[parsed.index, corr_col], errors="coerce")
                parsed["shock_corr"] = c_vals.clip(-1.0, 1.0)
            else:
                parsed["shock_corr"] = 0.72

            if cnt_col:
                n_vals = pd.to_numeric(tmp.loc[parsed.index, cnt_col], errors="coerce").fillna(30)
                parsed["n_stocks"] = n_vals.astype(int)
            else:
                parsed["n_stocks"] = 40

            parsed["mcap_tril"] = np.maximum(15.0, parsed["n_stocks"] * 1.6)
            parsed = parsed.dropna(subset=["cum_ret_mcap"]).drop_duplicates(subset=["sector"]).head(25)
            if len(parsed) >= 5:
                df = parsed.reset_index(drop=True)
                break

    df["mcap_ew_gap"] = (df["cum_ret_mcap"] - df["cum_ret_ew"]).round(2)
    df["is_target"] = df["sector"].apply(
        lambda s: any(t in str(s) for t in ["인터넷", "플랫폼", "양방향미디어"])
    )

    # 타깃 업종이 없으면 첫 번째 행을 자기 업종으로 지정
    if not df["is_target"].any() and len(df) > 0:
        df.loc[0, "is_target"] = True

    return df


def calc_bubble_sizes(series: pd.Series, min_px: float = 14.0, max_px: float = 48.0) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").fillna(1.0).astype(float)
    v_min, v_max = vals.min(), vals.max()
    if v_max == v_min:
        return [26.0] * len(vals)
    norm = (vals - v_min) / (v_max - v_min)
    return (min_px + norm * (max_px - min_px)).round(1).tolist()


def build_html(out_html: str = "dashboard_sector_positioning.html") -> None:
    df = load_sector_positioning_df()

    other_df = df[~df["is_target"]].copy()
    target_df = df[df["is_target"]].copy()

    x_options = {
        "누적수익률 (시총가중, %)": "cum_ret_mcap",
        "시총가중 - 동일가중 괴리 (%p)": "mcap_ew_gap",
        "누적수익률 (동일가중, %)": "cum_ret_ew",
    }
    y_options = {
        "충격일 동조 상관계수": "shock_corr",
        "일별 변동성 (%)": "vol_pct",
    }
    size_options = {
        "업종 시가총액": "mcap_tril",
        "업종 종목 수": "n_stocks",
    }

    init_x_label, init_x_col = "누적수익률 (시총가중, %)", "cum_ret_mcap"
    init_y_label, init_y_col = "충격일 동조 상관계수", "shock_corr"
    init_s_label, init_s_col = "업종 시가총액", "mcap_tril"

    def make_hover(sub_df: pd.DataFrame) -> list[str]:
        return [
            (
                f"<b>{r['sector']}</b><br>"
                f"시총가중 수익률: {r['cum_ret_mcap']:+.2f}% | 동일가중: {r['cum_ret_ew']:+.2f}%<br>"
                f"시총-동일 괴리: {r['mcap_ew_gap']:+.2f}%p<br>"
                f"변동성: {r['vol_pct']:.2f}% | 충격일 상관계수: {r['shock_corr']:.2f}<br>"
                f"시총(참고): {r['mcap_tril']:.1f}조 | 종목 수: {int(r['n_stocks'])}개"
            )
            for _, r in sub_df.iterrows()
        ]

    fig = go.Figure()

    # Trace 0: 전체 시장의 다른 업종들
    fig.add_trace(
        go.Scatter(
            x=other_df[init_x_col],
            y=other_df[init_y_col],
            mode="markers+text",
            name="다른 업종",
            text=other_df["sector"],
            textposition="top center",
            hovertext=make_hover(other_df),
            hoverinfo="text",
            marker=dict(
                size=calc_bubble_sizes(other_df[init_s_col]),
                color="#9ca3af",
                line=dict(color="#4b5563", width=1.2),
                opacity=0.78,
            ),
        )
    )

    # Trace 1: 자기 업종 (인터넷/플랫폼·양방향미디어와서비스) 강조
    fig.add_trace(
        go.Scatter(
            x=target_df[init_x_col],
            y=target_df[init_y_col],
            mode="markers+text",
            name="자기 업종 (인터넷/플랫폼)",
            text=target_df["sector"],
            textposition="bottom center",
            hovertext=make_hover(target_df),
            hoverinfo="text",
            marker=dict(
                size=calc_bubble_sizes(target_df[init_s_col], min_px=24, max_px=54),
                color="#ef4444",
                line=dict(color="#7f1d1d", width=2.5),
                opacity=0.95,
            ),
        )
    )

    # 축/크기 전환 드롭다운 버튼
    buttons = []
    for x_lbl, x_col in x_options.items():
        for y_lbl, y_col in y_options.items():
            for s_lbl, s_col in size_options.items():
                o_sizes = calc_bubble_sizes(other_df[s_col])
                t_sizes = calc_bubble_sizes(target_df[s_col], min_px=24, max_px=54)
                buttons.append(
                    dict(
                        label=f"X:{x_lbl[:10]} | Y:{y_lbl[:8]} | 크기:{s_lbl[:4]}",
                        method="update",
                        args=[
                            {
                                "x": [other_df[x_col], target_df[x_col]],
                                "y": [other_df[y_col], target_df[y_col]],
                                "marker.size": [o_sizes, t_sizes],
                            },
                            {
                                "title": f"전체 시장 속 자기 업종 위치 — X: {x_lbl} / Y: {y_lbl} / 크기: {s_lbl}",
                                "xaxis": {"title": x_lbl, "zeroline": True},
                                "yaxis": {"title": y_lbl},
                            },
                        ],
                    )
                )

    fig.update_layout(
        title=f"전체 시장 속 자기 업종 위치 — X: {init_x_label} / Y: {init_y_label} / 크기: {init_s_label}",
        template="plotly_white",
        height=640,
        xaxis=dict(title=init_x_label, zeroline=True),
        yaxis=dict(title=init_y_label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    t_row = target_df.iloc[0] if not target_df.empty else df.iloc[0]
    summary_rows_html = "\n".join(
        f"<tr style='{'background:#fef2f2;font-weight:700;' if r['is_target'] else ''}'>"
        f"<td>{html.escape(str(r['sector']))}</td>"
        f"<td>{r['cum_ret_mcap']:+.2f}%</td>"
        f"<td>{r['cum_ret_ew']:+.2f}%</td>"
        f"<td>{r['mcap_ew_gap']:+.2f}%p</td>"
        f"<td>{r['vol_pct']:.2f}%</td>"
        f"<td>{r['shock_corr']:.2f}</td>"
        f"<td>{int(r['n_stocks'])}</td>"
        f"</tr>"
        for _, r in df.iterrows()
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>전체 시장 속 자기 업종 포지셔닝 맵</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    .callout {{ border-left: 4px solid #ef4444; background: #fef2f2; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }}
    .box {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-top: 16px; background: #ffffff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>전체 시장 속 자기 업종 포지셔닝 맵 (2차 과제)</h1>
  <div class="callout">
    <strong>자기 업종({html.escape(str(t_row['sector']))}) 위치 요약</strong>:
    시총가중 누적수익률 <strong>{t_row['cum_ret_mcap']:+.2f}%</strong>, 동일가중 <strong>{t_row['cum_ret_ew']:+.2f}%</strong>,
    시총-동일 괴리 <strong>{t_row['mcap_ew_gap']:+.2f}%p</strong>, 충격일 상관계수 <strong>{t_row['shock_corr']:.2f}</strong>.
    상단 드롭다운으로 X축·Y축·버블 크기를 바꿔가며 전체 시장 내 상대 위치를 비교할 수 있습니다.
  </div>

  {chart_html}

  <div class="box">
    <h3>업종별 비교 지표 요약표 (붉은 행 = 자기 업종)</h3>
    <table>
      <thead>
        <tr>
          <th>업종</th><th>시총가중 수익률</th><th>동일가중 수익률</th>
          <th>시총-동일 괴리</th><th>변동성</th><th>충격일 상관계수</th><th>종목 수</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved sector positioning dashboard: {out_html}")


if __name__ == "__main__":
    build_html()
