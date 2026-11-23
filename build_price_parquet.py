import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from bs4 import BeautifulSoup

SRC_DIR = "/mnt/data/stock"
OUT_DIR = "parquet"
os.makedirs(OUT_DIR, exist_ok=True)

def to_num(txt):
    s = re.sub(r"[^0-9.-]", "", str(txt or ""))
    if not s or s in ("-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def parse_html(args):
    date_str, fp = args
    stem = os.path.basename(fp).replace(".html", "")
    code = stem.split("_")[-1] if "_" in stem else stem
    name = stem.rsplit("_", 1)[0] if "_" in stem else stem
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        txt = soup.get_text("\n", strip=True)
        lines = [x.strip() for x in txt.splitlines() if x.strip()]

        close_val = None
        vol_val = None
        mktcap_val = None
        sector_val = None

        for i, ln in enumerate(lines):
            if ln in ("현재가", "종가") and i + 1 < len(lines) and close_val is None:
                close_val = to_num(lines[i + 1])
            elif ln == "거래량" and i + 1 < len(lines) and vol_val is None:
                vol_val = to_num(lines[i + 1])
            elif "시가총액" in ln and i + 1 < len(lines) and mktcap_val is None:
                mktcap_val = to_num(lines[i + 1])
            elif "업종" in ln and i + 1 < len(lines) and sector_val is None:
                cand = lines[i + 1]
                if len(cand) < 40 and not cand.replace(",", "").isdigit():
                    sector_val = cand

        if close_val is None and vol_val is None:
            return None
        return {
            "date": date_str,
            "code": code,
            "name": name,
            "sector": sector_val,
            "close": close_val,
            "volume": vol_val,
            "mktcap_100mn": mktcap_val,
        }
    except Exception:
        return None

def run_date(d_dir):
    d = os.path.basename(d_dir)
    out = os.path.join(OUT_DIR, f"{d}.parquet")
    if os.path.exists(out):
        return f"[스킵] {d}"
    files = glob.glob(os.path.join(d_dir, "*.html"))
    if not files:
        return f"[빈폴더] {d}"
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(parse_html, [(d, fp) for fp in files], chunksize=100):
            if r:
                rows.append(r)
    if rows:
        pd.DataFrame(rows).to_parquet(out, index=False)
        return f"[완료] {d} ({len(rows)}종목)"
    return f"[없음] {d}"

if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(SRC_DIR, "20*")))
    print("전체 날짜 폴더 수:", len(dirs))
    for d in dirs:
        print(run_date(d))
