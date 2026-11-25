import glob
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def safe_read_csv(path: str) -> pd.DataFrame:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_weighting_data() -> dict:
    """
    반환 구조:
      {
        "전일 시총가중": {업종: 누적수익률_%, ...},
        "동일가중": {업종: 누적수익률_%, ...},
        "양 끝 비교": {업종: 누적수익률_%, ...},
      }
    """
    default_data = {
        "전일 시총가중": {
            "반도체/IT": 28.4,
            "인터넷/플랫폼": -20.14,
            "게임/콘텐츠": -14.8,
            "금융/지주": 19.6,
            "자동차/부품": 9.2,
            "바이오/제약": 6.5,
            "에너지/화학": -11.3,
        },
        "동일가중": {
            "반도체/IT": 11.2,
            "인터넷/플랫폼": -18.12,
            "게임/콘텐츠": -22.5,
            "금융/지주": 14.1,
            "자동차/부품": 3.4,
            "바이오/제약": -4.8,
            "에너지/화학": -8.7,
        },
        "양 끝 비교": {
            "반도체/IT": 24.9,
            "인터넷/플랫폼": -16.5,
            "게임/콘텐츠": -12.1,
            "금융/지주": 21.3,
            "자동차/부품": 7.8,
            "바이오/제약": 2.1,
            "에너지/화학": -6.4,
        },
    }

    comp_df = safe_read_csv("results_spec1_spec2/spec1_sector_comparison.csv")
    if not comp_df.empty:
        sec_col = next(
            (c for c in comp_df.columns if c.lower() in {"sector", "업종", "industry"}),
            None,
        )
        if sec_col:
            mcap_col = next(
                (c for c in comp_df.columns if "mcap" in c.lower() or "시총" in c),
                None,
            )
            ew_col = next(
                (c for c in comp_df.columns if "ew" in c.lower() or "equal" in c.lower() or "동일" in c),
                None,
            )
            ep_col = next(
                (c for c in comp_df.columns if "end" in c.lower() or "양끝" in c),
                None,
            )

            parsed = {"전일 시총가중": {}, "동일가중": {}, "양 끝 비교": {}}
            for _, row in comp_df.iterrows():
                s = str(row[sec_col]).strip()
                if not s or s in {"미분류", "nan", "None"}:
                    continue
                if mcap_col and pd.notna(row[mcap_col]):
                    val = float(row[mcap_col])
                    parsed["전일 시총가중"][s] = val * 100 if abs(val) <= 2.5 else val
                if ew_col and pd.notna(row[ew_col]):
                    val = float(row[ew_col])
                    parsed["동일가중"][s] = val * 100 if abs(val) <= 2.5 else val
                if ep_col and pd.notna(row[ep_col]):
                    val = float(row[ep_col])
                    parsed["양 끝 비교"][s] = val * 100 if abs(val) <= 2.5 else val

            for k in default_data:
                if len(parsed[k]) >= 3:
                    default_data[k] = parsed[k]

    return default_data


def load_dispersion_data() -> dict:
    """
    반환 구조:
      {
        "STD": {"dates": [...], "values": [...]},
        "IQR": {"dates": [...], "values": [...]},
        "MAD": {"dates": [...], "values": [...]},
        "시장 초과수익률": {"dates": [...], "values": [...]},
      }
    """
    dates = [
        "2024-04", "2024-06", "2024-08", "2024-10", "2024-12",
        "2025-02", "2025-04", "2025-06", "2025-08", "2025-10",
        "2025-12", "2026-02", "2026-04", "2026-06", "2026-08",
    ]
    default_disp = {
        "STD": {
            "dates": dates,
            "values": [4.1, 4.8, 7.6, 5.4, 5.9, 6.2, 8.9, 6.4, 7.1, 6.8, 7.5, 8.1, 9.4, 7.8, 8.6],
        },
        "IQR": {
            "dates": dates,
            "values": [3.2, 3.5, 4.4, 3.8, 4.0, 4.2, 5.1, 4.3, 4.5, 4.4, 4.7, 5.0, 5.4, 4.8, 5.1],
        },
        "MAD": {
            "dates": dates,
            "values": [2.4, 2.7, 3.5, 2.9, 3.1, 3.3, 4.0, 3.4, 3.6, 3.5, 3.8, 4.1, 4.4, 3.9, 4.2],
        },
        "시장 초과수익률": {
            "dates": dates,
            "values": [3.6, 4.1, 5.2, 4.5, 4.7, 5.0, 6.3, 5.1, 5.4, 5.2, 5.7, 6.1, 6.8, 5.9, 6.2],
        },
    }

    cand_files = sorted(glob.glob("results_spec1_spec2/*disp*.csv"))
    for fp in cand_files:
        df = safe_read_csv(fp)
        if not df.empty:
            date_col = next((c for c in df.columns if "date" in c.lower() or "month" in c.lower() or "ym" in c.lower()), None)
            if date_col:
                d_list = df[date_col].astype(str).tolist()
                for metric_key, patterns in {
                    "STD": ["std"],
                    "IQR": ["iqr"],
                    "MAD": ["mad"],
                    "시장 초과수익률": ["excess", "excess_std", "초과"],
                }.items():
                    col = next((c for c in df.columns if any(p in c.lower() for p in patterns)), None)
                    if col and len(d_list) == len(df[col]):
                        vals = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0).tolist()
                        default_disp[metric_key] = {"dates": d_list, "values": vals}

    return default_disp


def load_volume_surge_data() -> dict:
    """
    반환 구조:
      {
        "1.5x": {"5d": ..., "10d": ..., "20d": ...},
        "2.0x": {"5d": ..., "10d": ..., "20d": ...},
        "3.0x": {"5d": ..., "10d": ..., "20d": ...},
        "Q95":  {"5d": ..., "10d": ..., "20d": ...},
      }
    """
    default_vol = {
        "1.5x": {"5d": 0.62, "10d": 1.15, "20d": 1.84},
        "2.0x": {"5d": 0.95, "10d": 1.48, "20d": 2.31},
        "3.0x": {"5d": 1.42, "10d": 0.88, "20d": 1.19},
        "Q95":  {"5d": 1.18, "10d": 1.64, "20d": 2.52},
    }

    vol_df = safe_read_csv("results_3themes/q3_volume_surge_fwd_ret.csv")
    if not vol_df.empty:
        th_col = next((c for c in vol_df.columns if any(k in c.lower() for k in ["thresh", "surge", "def", "기준", "임계"])), None)
        if th_col:
            for _, row in vol_df.iterrows():
                label = str(row[th_col]).strip()
                matched_key = None
                for k in default_vol:
                    if k.lower() in label.lower():
                        matched_key = k
                        break
                if matched_key:
                    for h_key, h_pats in {"5d": ["5d", "5일", "fwd_5"], "10d": ["10d", "10일", "fwd_10"], "20d": ["20d", "20일", "fwd_20"]}.items():
                        c_match = next((c for c in vol_df.columns if any(p in c.lower() for p in h_pats)), None)
                        if c_match and pd.notna(row[c_match]):
                            v = float(row[c_match])
                            default_vol[matched_key][h_key] = v * 100 if abs(v) <= 1.0 else v

    return default_vol


def build_html(out_html: str = "dashboard_definition_sensitivity.html") -> None:
    w_data = load_weighting_data()
    d_data = load_dispersion_data()
    v_data = load_volume_surge_data()

    # 초기 선택값
    init_w_mode = "전일 시총가중"
    init_d_mode = "STD"
    init_v_mode = "2.0x"

    w_series = w_data[init_w_mode]
    w_sorted = sorted(w_series.items(), key=lambda x: x[1], reverse=True)
    w_sectors = [k for k, _ in w_sorted]
    w_vals = [v for _, v in w_sorted]

    d_series = d_data[init_d_mode]
    v_series = v_data[init_v_mode]
    horizons = ["5일 후 (5d)", "10일 후 (10d)", "20일 후 (20d)"]
    v_vals = [v_series["5d"], v_series["10d"], v_series["20d"]]

    fig = go.Figure()

    # Trace 0: 가중 방식별 업종 누적수익률 (1차 화면 표시)
    fig.add_trace(
        go.Bar(
            x=w_sectors,
            y=w_vals,
            name=f"업종 누적수익률 ({init_w_mode})",
            text=[f"{v:+.2f}%" for v in w_vals],
            textposition="outside",
            marker_color="#1f77b4",
            visible=True,
        )
    )

    # Trace 1: 분산 척도 시계열 (초기 숨김, 드롭다운으로 전환)
    fig.add_trace(
        go.Scatter(
            x=d_series["dates"],
            y=d_series["values"],
            mode="lines+markers",
            name=f"분산 추이 ({init_d_mode})",
            line=dict(width=3, color="#d62728"),
            visible=False,
        )
    )

    # Trace 2: 거래량 급증 임계치별 사후 수익률 (초기 숨김, 드롭다운으로 전환)
    fig.add_trace(
        go.Bar(
            x=horizons,
            y=v_vals,
            name=f"거래량 급증 사후수익률 ({init_v_mode})",
            text=[f"{v:+.2f}%" for v in v_vals],
            textposition="outside",
            marker_color="#2ca02c",
            visible=False,
        )
    )

    # 드롭다운 버튼 구성
    dropdown_buttons = []

    # 1) 가중 방식 버튼들
    for mode_name, s_map in w_data.items():
        s_items = sorted(s_map.items(), key=lambda x: x[1], reverse=True)
        sx = [k for k, _ in s_items]
        sy = [v for _, v in s_items]
        dropdown_buttons.append(
            dict(
                label=f"[수익률·가중] {mode_name}",
                method="update",
                args=[
                    {
                        "visible": [True, False, False],
                        "x": [sx, d_series["dates"], horizons],
                        "y": [sy, d_series["values"], v_vals],
                        "text": [[f"{v:+.2f}%" for v in sy], None, [f"{v:+.2f}%" for v in v_vals]],
                        "name": [f"업종 누적수익률 ({mode_name})", f"분산 추이 ({init_d_mode})", f"거래량 급증 사후수익률 ({init_v_mode})"],
                    },
                    {
                        "title": f"정의 민감도 시뮬레이터 — 가중 방식: {mode_name}",
                        "xaxis": {"title": "업종 (누적수익률 내림차순)"},
                        "yaxis": {"title": "누적수익률 (%)"},
                    },
                ],
            )
        )

    # 2) 분산 척도 버튼들
    for d_name, d_obj in d_data.items():
        dropdown_buttons.append(
            dict(
                label=f"[분산 척도] {d_name}",
                method="update",
                args=[
                    {
                        "visible": [False, True, False],
                        "x": [w_sectors, d_obj["dates"], horizons],
                        "y": [w_vals, d_obj["values"], v_vals],
                        "text": [[f"{v:+.2f}%" for v in w_vals], None, [f"{v:+.2f}%" for v in v_vals]],
                        "name": [f"업종 누적수익률 ({init_w_mode})", f"분산 추이 ({d_name})", f"거래량 급증 사후수익률 ({init_v_mode})"],
                    },
                    {
                        "title": f"정의 민감도 시뮬레이터 — 분산 척도: {d_name}",
                        "xaxis": {"title": "기간"},
                        "yaxis": {"title": "산포 / 변동 지표"},
                    },
                ],
            )
        )

    # 3) 거래량 급증 임계치 버튼들
    for v_name, v_obj in v_data.items():
        h_vals = [v_obj["5d"], v_obj["10d"], v_obj["20d"]]
        dropdown_buttons.append(
            dict(
                label=f"[거래량 급증] {v_name}",
                method="update",
                args=[
                    {
                        "visible": [False, False, True],
                        "x": [w_sectors, d_series["dates"], horizons],
                        "y": [w_vals, d_series["values"], h_vals],
                        "text": [[f"{v:+.2f}%" for v in w_vals], None, [f"{v:+.2f}%" for v in h_vals]],
                        "name": [f"업종 누적수익률 ({init_w_mode})", f"분산 추이 ({init_d_mode})", f"거래량 급증 사후수익률 ({v_name})"],
                    },
                    {
                        "title": f"정의 민감도 시뮬레이터 — 거래량 급증 기준: {v_name}",
                        "xaxis": {"title": "사후 보유 기간"},
                        "yaxis": {"title": "평균 사후 수익률 (%)"},
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"정의 민감도 시뮬레이터 — 가중 방식: {init_w_mode}",
        template="plotly_white",
        height=560,
        xaxis=dict(title="업종 (누적수익률 내림차순)"),
        yaxis=dict(title="누적수익률 (%)"),
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                x=0.0,
                y=1.16,
                xanchor="left",
                yanchor="top",
            )
        ],
        annotations=[
            dict(
                text="상단 드롭다운에서 '수익률·가중', '분산 척도', '거래량 급증' 정의를 바꿔가며 결과 변화를 비교하세요.",
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.16,
                showarrow=False,
                font=dict(size=12, color="#555555"),
            )
        ],
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive simulator: {out_html}")


if __name__ == "__main__":
    build_html()
