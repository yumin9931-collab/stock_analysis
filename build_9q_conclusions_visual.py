import html
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go


Q_CONCLUSIONS = [
    {
        "qid": "공통 Q1",
        "category": "공통 질문 (1~4)",
        "topic": "시가총액 변동·순위 이동",
        "sensitivity": "전일·당일 유효 시총 기준 순위·비중 변화",
        "conclusion": "기간 동안 시총 상위 종목과 주도 업종으로의 비중 쏠림과 순위 교체가 뚜렷하게 나타났습니다.",
    },
    {
        "qid": "공통 Q2",
        "category": "공통 질문 (1~4)",
        "topic": "업종 수익률",
        "sensitivity": "시총가중 vs 동일가중 vs 양 끝 비교",
        "conclusion": "시총가중과 동일가중 괴리가 큰 업종에서는 대형주 견인·부진 여부에 따라 수익률 결론이 달라집니다.",
    },
    {
        "qid": "공통 Q3",
        "category": "공통 질문 (1~4)",
        "topic": "업종 간 분산",
        "sensitivity": "STD·IQR·MAD·시장 초과수익률·누적 범위",
        "conclusion": "누적수익률 퍼짐은 전반적으로 확대되나 시장 초과수익률이나 IQR·MAD로 보면 극단 업종과 공통 변동을 분리해 읽어야 합니다.",
    },
    {
        "qid": "공통 Q4",
        "category": "공통 질문 (1~4)",
        "topic": "동조화·거래량 급증",
        "sensitivity": "하위 10% 충격일 상관 / 1.5x·2x·3x·Q95 후 5·10·20일",
        "conclusion": "충격일에는 업종 간 동조화가 높아지고 거래량 급증 이후 사후 수익률은 급증 기준을 어떻게 자르느냐에 따라 방향과 크기가 달라집니다.",
    },
    {
        "qid": "재무 Q5",
        "category": "재무제표 질문 (1~5)",
        "topic": "영업이익률과 주가 수익률",
        "sensitivity": "OPM·누적수익률 관계, 업종 내 일치, 개선폭 vs 고수준",
        "conclusion": "높은 영업이익률이나 개선이 항상 주가 수익률로 직결되지는 않고 업종 내에서도 종목별 편차가 나타납니다.",
    },
    {
        "qid": "재무 Q6",
        "category": "재무제표 질문 (1~5)",
        "topic": "실적 발표 뒤 반응 기간",
        "sensitivity": "표 변경일 전후 10일 누적수익률, Beat vs Miss",
        "conclusion": "실적 발표 전후 주가 반응은 며칠 내 선반영·소화되는 패턴을 보이되 추정치 상회 여부와 표본 조건에 따라 반응 차이가 발생합니다.",
    },
    {
        "qid": "재무 Q7",
        "category": "재무제표 질문 (1~5)",
        "topic": "추정 오차",
        "sensitivity": "업종별 오차 분포, 발표 전 수정 경로, 상향·하향 쏠림",
        "conclusion": "업종별로 오차 크기와 수정 궤적이 다르고 한쪽 방향으로 쏠리는 편향이 관측됩니다.",
    },
    {
        "qid": "재무 Q8",
        "category": "재무제표 질문 (1~5)",
        "topic": "부채비율과 변동성",
        "sensitivity": "부채비율-변동성 관계, 급락 구간 vs 보통 구간",
        "conclusion": "부채비율이 높은 영역에서는 급락 국면에서 변동성이 더 확대되는 경향이 확인됩니다.",
    },
    {
        "qid": "재무 Q9",
        "category": "재무제표 질문 (1~5)",
        "topic": "저PBR(PBR < 1) 2.5년 후",
        "sensitivity": "집중 업종, 초기 ROE 양수 vs 음수, 해소(>=1) vs 미해소",
        "conclusion": "2년 반 뒤 성과는 저PBR 해소 여부와 ROE 양수 여부에 따라 가치주와 가치함정 간 뚜렷한 차이를 보입니다.",
    },
]


def set_korean_font() -> None:
    candidates = [
        "NanumGothic",
        "AppleGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def build_png(out_png: str = "q1_to_q9_conclusions.png") -> None:
    set_korean_font()
    fig, ax = plt.subplots(figsize=(16, 9.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.text(
        0.01,
        0.96,
        "1차 과제 결론 요약: 9개 질문 한 문장 결론 (정의·기준에 따른 결과 차이)",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.01,
        0.922,
        "핵심 메시지: 같은 질문이라도 기간·가중치·분산척도·급증임계치·결함필터·평균/중앙값 정의에 따라 답과 결론이 달라집니다.",
        fontsize=10.5,
        color="#374151",
        va="top",
    )

    n = len(Q_CONCLUSIONS)
    top_y = 0.875
    bottom_y = 0.06
    row_h = (top_y - bottom_y) / n

    # 헤더 행
    ax.add_patch(
        plt.Rectangle((0.01, top_y), 0.98, row_h * 0.85, facecolor="#1f2937", edgecolor="none")
    )
    ax.text(0.02, top_y + row_h * 0.42, "구분", color="white", fontsize=9.5, fontweight="bold", va="center")
    ax.text(0.11, top_y + row_h * 0.42, "주제", color="white", fontsize=9.5, fontweight="bold", va="center")
    ax.text(0.28, top_y + row_h * 0.42, "한 문장 결론 요약", color="white", fontsize=9.5, fontweight="bold", va="center")

    for idx, item in enumerate(Q_CONCLUSIONS):
        y0 = top_y - (idx + 1) * row_h
        is_common = idx < 4
        bg = "#f8fafc" if (idx % 2 == 0 and is_common) else ("#ffffff" if is_common else ("#fefce8" if idx % 2 == 0 else "#fffbeb"))
        badge_color = "#1d4ed8" if is_common else "#b45309"

        ax.add_patch(
            plt.Rectangle((0.01, y0), 0.98, row_h * 0.92, facecolor=bg, edgecolor="#e5e7eb", linewidth=0.8)
        )
        ax.text(
            0.02,
            y0 + row_h * 0.46,
            item["qid"],
            color=badge_color,
            fontsize=9.5,
            fontweight="bold",
            va="center",
        )
        ax.text(
            0.11,
            y0 + row_h * 0.46,
            item["topic"],
            color="#111827",
            fontsize=9.2,
            fontweight="bold",
            va="center",
        )
        ax.text(
            0.28,
            y0 + row_h * 0.46,
            item["conclusion"],
            color="#1f2937",
            fontsize=8.9,
            va="center",
        )

    ax.text(
        0.01,
        0.022,
        "출처: 2.5년 시세·재무 아카이브 분석 결과 (GitHub: yumin9931-collab/stock_analysis)",
        fontsize=8.5,
        color="#6b7280",
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved conclusion PNG: {out_png}")


def build_html(out_html: str = "dashboard_9q_conclusions.html") -> None:
    qids = [x["qid"] for x in Q_CONCLUSIONS]
    topics = [x["topic"] for x in Q_CONCLUSIONS]
    sens = [x["sensitivity"] for x in Q_CONCLUSIONS]
    concs = [x["conclusion"] for x in Q_CONCLUSIONS]
    colors = ["#dbeafe" if i < 4 else "#fef3c7" for i in range(len(Q_CONCLUSIONS))]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[90, 150, 220, 460],
                header=dict(
                    values=["질문", "주제", "검토한 정의·기준", "한 문장 결론 요약"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=12),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[qids, topics, sens, concs],
                    fill_color=[colors, ["#f9fafb"] * 9, ["#f9fafb"] * 9, ["#ffffff"] * 9],
                    font=dict(color="#111827", size=11),
                    align="left",
                    height=44,
                ),
            )
        ]
    )

    common_idx = [0, 1, 2, 3]
    fin_idx = [4, 5, 6, 7, 8]

    fig.update_layout(
        title="1차 발표 결론 요약: 9개 질문 한 문장 결론 매트릭스",
        template="plotly_white",
        height=560,
        updatemenus=[
            dict(
                active=0,
                x=0.0,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="전체 9개 질문 보기",
                        method="update",
                        args=[
                            {
                                "cells": [
                                    dict(
                                        values=[qids, topics, sens, concs],
                                        fill_color=[colors, ["#f9fafb"] * 9, ["#f9fafb"] * 9, ["#ffffff"] * 9],
                                    )
                                ]
                            }
                        ],
                    ),
                    dict(
                        label="공통 질문 (Q1~Q4)",
                        method="update",
                        args=[
                            {
                                "cells": [
                                    dict(
                                        values=[
                                            [qids[i] for i in common_idx],
                                            [topics[i] for i in common_idx],
                                            [sens[i] for i in common_idx],
                                            [concs[i] for i in common_idx],
                                        ],
                                        fill_color=[["#dbeafe"] * 4, ["#f9fafb"] * 4, ["#f9fafb"] * 4, ["#ffffff"] * 4],
                                    )
                                ]
                            }
                        ],
                    ),
                    dict(
                        label="재무제표 질문 (Q5~Q9)",
                        method="update",
                        args=[
                            {
                                "cells": [
                                    dict(
                                        values=[
                                            [qids[i] for i in fin_idx],
                                            [topics[i] for i in fin_idx],
                                            [sens[i] for i in fin_idx],
                                            [concs[i] for i in fin_idx],
                                        ],
                                        fill_color=[["#fef3c7"] * 5, ["#f9fafb"] * 5, ["#f9fafb"] * 5, ["#ffffff"] * 5],
                                    )
                                ]
                            }
                        ],
                    ),
                ],
            )
        ],
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    cards_html = "\n".join(
        f"""<div class="card">
          <div class="meta"><span class="badge">{html.escape(x['qid'])}</span> <strong>{html.escape(x['topic'])}</strong> · <span class="sens">{html.escape(x['sensitivity'])}</span></div>
          <div class="conc">{html.escape(x['conclusion'])}</div>
        </div>"""
        for x in Q_CONCLUSIONS
    )

    page_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>9개 질문 한 문장 결론 요약</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    .lead {{ background: #f3f4f6; border-left: 4px solid #111827; padding: 12px 16px; border-radius: 6px; margin-bottom: 18px; }}
    .cards {{ display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 18px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; background: #fafafa; }}
    .badge {{ background: #1f2937; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 700; margin-right: 6px; }}
    .sens {{ color: #6b7280; font-size: 12px; }}
    .conc {{ margin-top: 6px; font-size: 14px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>1차 발표 결론 요약: 9개 질문 한 문장 결론</h1>
  <div class="lead">
    결국 같은 질문이라도 기간을 어떻게 자르고, 가중치나 급증 기준·결함 필터를 어떻게 두느냐에 따라 답과 결론이 달라집니다.
  </div>
  {chart_html}
  <div class="cards">
    {cards_html}
  </div>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"Saved conclusion HTML: {out_html}")


if __name__ == "__main__":
    build_png()
    build_html()
