import os
import glob
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from bs4 import BeautifulSoup

SRC_DIR = "/mnt/data/stock"
OUT_DIR = "fin_parquet"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_METRICS = {
    "매출액",
    "영업이익",
    "당기순이익",
    "영업이익률",
    "순이익률",
    "ROE(지배주주)",
    "부채비율",
    "EPS(원)",
    "PER(배)",
    "BPS(원)",
    "PBR(배)",
}


def parse_one_html(args):
    date_str, filepath = args
    stem = os.path.basename(filepath).replace(".html", "")
    code = stem.split("_")[-1] if "_" in stem else stem
    name = stem.rsplit("_", 1)[0] if "_" in stem else stem

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "lxml")

        target_tbl = None
        for tbl in soup.find_all("table"):
            txt = tbl.get_text(" ", strip=True)
            if "주요재무정보" in txt and ("영업이익" in txt or "매출액" in txt):
                target_tbl = tbl
                break
        if target_tbl is None:
            return None

        # 기간 헤더 추출 (보통 thead의 마지막 tr에 연도/분기(E)가 들어있음)
        thead = target_tbl.find("thead")
        if not thead:
            return None
        header_trs = thead.find_all("tr")
        if not header_trs:
            return None
        period_cells = [th.get_text(" ", strip=True) for th in header_trs[-1].find_all(["th", "td"])]

        tbody = target_tbl.find("tbody")
        if not tbody:
            return None

        rows = []
        for tr in tbody.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            m_name = cells[0].strip()
            if m_name not in TARGET_METRICS:
                continue

            val_cells = cells[1:]
            # period_cells 길이가 val_cells보다 1만큼 더 길면(항목명 헤더 포함) 뒤쪽을 맞춤
            p_list = period_cells[-len(val_cells):] if len(period_cells) >= len(val_cells) else period_cells

            for idx, v_raw in enumerate(val_cells):
                p_raw = p_list[idx] if idx < len(p_list) else f"col_{idx}"
                if not p_raw or p_raw in ("IFRS연결", "IFRS별도"):
                    continue
                is_est = "(E)" in p_raw
                period_clean = p_raw.replace("(E)", "").strip()
                v_clean = v_raw.replace(",", "").strip()
                try:
                    val = float(v_clean) if v_clean not in ("", "-", "nan", "None", "N/A") else None
                except ValueError:
                    val = None
                rows.append({
                    "date": date_str,
                    "code": code,
                    "name": name,
                    "period": period_clean,
                    "is_est": is_est,
                    "metric": m_name,
                    "value": val,
                })

        return rows if rows else None
    except Exception:
        return None


def process_date(date_dir):
    d = os.path.basename(date_dir)
    out = os.path.join(OUT_DIR, f"{d}.parquet")
    if os.path.exists(out):
        os.remove(out)
    files = glob.glob(os.path.join(date_dir, "*.html"))
    if not files:
        return f"[빈폴더] {d}"

    all_rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(parse_one_html, [(d, fp) for fp in files], chunksize=100):
            if res:
                all_rows.extend(res)

    if all_rows:
        pd.DataFrame(all_rows).to_parquet(out, index=False)
        return f"[완료] {d} ({len(all_rows)}행)"
    return f"[없음] {d}"


if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(SRC_DIR, "20*")))
    for d in dirs[:3]:
        print(process_date(d))
