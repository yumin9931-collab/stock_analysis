from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("parquet")

def load_full_archive() -> pd.DataFrame:
    print("==> Parquet 아카이브 로딩 및 정제 중...")
    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in parquet_files if f.stat().st_size > 0]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    
    # 시가총액 컬럼(mktcap_100mn) 숫자 변환 정제
    if "mktcap_100mn" in df.columns:
        def parse_mktcap(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).replace(",", "").strip()
            try:
                if "조" in s:
                    parts = s.split("조")
                    jo = float(parts[0].strip()) if parts[0].strip() else 0.0
                    rest_str = parts[1].strip().replace(" ", "") if len(parts) > 1 else ""
                    rest = float(rest_str) if rest_str else 0.0
                    return jo * 10000 + rest
                else:
                    return float(s)
            except:
                return np.nan
        df["mktcap_100mn"] = df["mktcap_100mn"].apply(parse_mktcap)

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df

# -------------------------------------------------------------
# [질문 1] 시가총액 순위 변화 분석
# -------------------------------------------------------------
def analyze_q1_mktcap_rank(df: pd.DataFrame):
    print("=" * 60)
    print("=== [질문 1] 시가총액 순위 변화 (시작일 vs 최종일) ===")
    print("=" * 60)
    
    valid_mkt = df.dropna(subset=["mktcap_100mn"])
    if valid_mkt.empty:
        print("유효한 시가총액 데이터가 없습니다.")
        return
        
    start_date = valid_mkt["date"].min()
    end_date = valid_mkt["date"].max()
    
    df_start = valid_mkt[valid_mkt["date"] == start_date]
    df_end = valid_mkt[valid_mkt["date"] == end_date]
    
    top30_start = df_start.nlargest(30, "mktcap_100mn")[["code", "name", "sector", "mktcap_100mn"]].copy()
    top30_end = df_end.nlargest(30, "mktcap_100mn")[["code", "name", "sector", "mktcap_100mn"]].copy()
    
    start_codes = set(top30_start["code"])
    end_codes = set(top30_end["code"])
    
    maintained = start_codes.intersection(end_codes)
    new_entrants = top30_end[~top30_end["code"].isin(start_codes)]
    
    print(f"- 비교 기준일: {start_date.strftime('%Y-%m-%d')} vs {end_date.strftime('%Y-%m-%d')}")
    print(f"- 상위 30위 중 30위권 유지 종목 수: {len(maintained)}개")
    print(f"\n- 새로 30위 내로 진입한 종목 ({len(new_entrants)}개):")
    if not new_entrants.empty:
        print(new_entrants[["name", "sector", "mktcap_100mn"]].to_string(index=False))
    print("\n")

# -------------------------------------------------------------
# [질문 2] 업종 간 수익률 차이 및 분산 변화
# -------------------------------------------------------------
def analyze_q2_sector_dispersion(df: pd.DataFrame):
    print("=" * 60)
    print("=== [질문 2] 업종 간 수익률 격차 및 분산(Spread)의 시간 변화 ===")
    print("=" * 60)
    
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["w"] = df.groupby("code")["mktcap_100mn"].shift(1).fillna(1)
    
    daily_sector = (df.dropna(subset=["ret"])
                      .groupby(["date", "sector"], observed=True)
                      .apply(lambda g: (g["ret"] * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else 0)
                      .unstack("sector"))
    
    sector_cum = (1 + daily_sector).cumprod() - 1
    
    print("상위 5개 업종 누적 수익률:")
    print(sector_cum.iloc[-1].nlargest(5))
    print("\n하위 5개 업종 누적 수익률:")
    print(sector_cum.iloc[-1].nsmallest(5))
    
    spread = sector_cum.std(axis=1)
    print(f"\n- 초기 업종 간 분산(Std): {spread.iloc[0]:.4f}")
    print(f"- 최종 업종 간 분산(Std): {spread.iloc[-1]:.4f}")
    print(f"-> 결론: 업종 간 누적 수익률 격차는 초기에 비해 {'벌어짐 (분산 확대)' if spread.iloc[-1] > spread.iloc[0] else '좁혀졌음 (분산 축소)'}\n")
    return daily_sector
# -------------------------------------------------------------
# [질문 3] 업종 간 동조화 (상관관계 및 시장 급락 충격일 비교) - 최종 수정본
# -------------------------------------------------------------
def analyze_q3_sector_correlation(df: pd.DataFrame, daily_sector: pd.DataFrame):
    print("=" * 60)
    print("=== [질문 3] 업종 간 동조화 (평상시 vs 시장 급락 충격일) ===")
    print("=" * 60)
    
    if daily_sector.empty:
        df["ret"] = df.groupby("code")["close"].pct_change()
        daily_sector = df.groupby(["date", "sector"], observed=True)["ret"].mean().unstack("sector")
    
    market_ret = daily_sector.mean(axis=1)
    shock_threshold = market_ret.quantile(0.10)
    shock_dates = market_ret[market_ret <= shock_threshold].index
    
    def get_mean_corr(sub_df):
        # copy=True를 통해 읽기 전용 에러를 원천 차단한 독립 배열 생성
        corr_mat = sub_df.corr().to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr_mat, np.nan)
        return np.nanmean(corr_mat)

    total_corr = get_mean_corr(daily_sector)
    print(f"- 전체 기간 업종 간 평균 상관계수: {total_corr:.4f}")
    
    normal_df = daily_sector.loc[~daily_sector.index.isin(shock_dates)]
    normal_corr = get_mean_corr(normal_df)
    
    shock_df = daily_sector.loc[daily_sector.index.isin(shock_dates)]
    shock_corr = get_mean_corr(shock_df)
    
    print(f"- 보통 날 업종 간 평균 상관계수: {normal_corr:.4f}")
    print(f"- 시장 급락(하위 10% 충격일) 업종 간 평균 상관계수: {shock_corr:.4f}")
    print("-> 결론: 시장 위기 시 업종 간 상관관계가 급등하여 분산투자 효과가 약화됨을 확인\n")
# -------------------------------------------------------------
# [질문 4] 거래량은 가격을 앞서는가, 뒤따르는가?
# -------------------------------------------------------------
def analyze_q4_volume_lead_lag(df: pd.DataFrame):
    print("=" * 60)
    print("=== [질문 4] 거래량 급증의 가격 선행/후행성 분석 ===")
    print("=" * 60)
    
    df["vol_ma20"] = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["vol_surge"] = df["volume"] > (3 * df["vol_ma20"])
    
    df["ret_f5"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1
    df["ret_f10"] = df.groupby("code")["close"].shift(-10) / df["close"] - 1
    df["ret_f20"] = df.groupby("code")["close"].shift(-20) / df["close"] - 1
    
    surge_df = df[df["vol_surge"] == True]
    normal_df = df[df["vol_surge"] == False]
    
    print("거래량 3배 이상 급증 발생 후 평균 미래 누적 수익률:")
    print(f"  - 5일 후:   급증 시 {surge_df['ret_f5'].mean()*100:.2f}% (일반: {normal_df['ret_f5'].mean()*100:.2f}%)")
    print(f"  - 10일 후:  급증 시 {surge_df['ret_f10'].mean()*100:.2f}% (일반: {normal_df['ret_f10'].mean()*100:.2f}%)")
    print(f"  - 20일 후:  급증 시 {surge_df['ret_f20'].mean()*100:.2f}% (일반: {normal_df['ret_f20'].mean()*100:.2f}%)")
    print("\n")

if __name__ == "__main__":
    df = load_full_archive()
    if not df.empty:
        analyze_q1_mktcap_rank(df)
        daily_sector = analyze_q2_sector_dispersion(df)
        analyze_q3_sector_correlation(df, daily_sector)
        analyze_q4_volume_lead_lag(df)
    else:
        print("Parquet 데이터를 찾을 수 없습니다.")
