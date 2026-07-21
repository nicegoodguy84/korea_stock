import os
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import urllib.request
import re
import warnings
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import io
import base64
import numpy as np
import platform

warnings.filterwarnings('ignore')

# OS별 폰트 동적 설정 (크로스플랫폼 대응 및 GitHub Actions 리눅스 한글 깨짐 방지)
system_name = platform.system()
if system_name == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    # Linux (GitHub Actions 환경)
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False

def get_dynamic_theme_stocks():
    """[제공 소스 유지] 실시간 거래대금 및 상승률 기준 상위 주도 테마의 종목들을 동적으로 수집합니다."""
    print("🔥 [동적 엔진] 현재 시장의 실시간 주도 테마 및 구성 종목을 추적합니다...")
    theme_stock_codes = set()
    try:
        url = "https://finance.naver.com/sise/theme.naver"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('cp949', errors='ignore')
            
        theme_ids = re.findall(r'theme_detail\.naver\?themeNo=(\d+)', html)
        target_themes = list(dict.fromkeys(theme_ids))[:5]
        
        for theme_no in target_themes:
            detail_url = f"https://finance.naver.com/sise/theme_detail.naver?themeNo={theme_no}"
            detail_req = urllib.request.Request(detail_url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                with urllib.request.urlopen(detail_req, timeout=5) as detail_res:
                    detail_html = detail_res.read().decode('cp949', errors='ignore')
                stock_codes = re.findall(r'href="/item/main\.naver\?code=(\d{6})"', detail_html)
                for code in stock_codes:
                    theme_stock_codes.add(code)
            except Exception:
                continue
        print(f"✅ 주도 테마 수집 완료: 실시간 급등 테마주 {len(theme_stock_codes)}개 확보")
    except Exception as e:
        print(f"⚠️ 실시간 테마주 수집 중 오류 발생 (기본 시총 기반으로 우회 진행): {e}")
    return list(theme_stock_codes)


def detect_vcp_and_pivot(df, lookback=40):
    """마크 미너비니의 핵심인 VCP(변동성 축소 패턴) 및 피벗 돌파를 정밀하게 연산합니다."""
    df_recent = df.tail(lookback).copy()
    if len(df_recent) < lookback:
        return False, 1.0, 1.0, "관망"

    std_recent = df_recent['Close'].tail(5).std()
    std_past = df_recent['Close'].iloc[:-10].std()
    vcp_ratio = round(std_recent / std_past, 2) if std_past > 0 else 1.0

    vol_recent_shrink = df_recent['Volume'].tail(3).mean()
    vol_past_shrink = df_recent['Volume'].mean()
    vol_shrink_ratio = round(vol_recent_shrink / vol_past_shrink, 2) if vol_past_shrink > 0 else 1.0

    high_20d = df_recent['High'].iloc[-20:-2].max()
    current_close = df_recent['Close'].iloc[-1]
    
    if vcp_ratio <= 0.65 and vol_shrink_ratio <= 0.70:
        m_point = "1차 타점 (VCP 수렴 완료)"
    elif current_close >= high_20d and df_recent['Volume'].iloc[-1] > vol_past_shrink * 1.5:
        m_point = "2차 타점 (피벗 거래량 돌파)"
    else:
        m_point = "조건 미달 (수렴 진행중)"

    return True, vcp_ratio, vol_shrink_ratio, m_point


def calculate_minervini_base(df_hist):
    """와인스타인 2단계(Stage 2) 안에서 미너비니 베이스의 카운트를 계산합니다."""
    if len(df_hist) < 200:
        return 1
    
    df_hist['MA200_slope'] = df_hist['MA200'].diff(5)
    stage2_df = df_hist[df_hist['MA200_slope'] > 0]
    
    if len(stage2_df) < 20:
        return 1
        
    base_count = 1
    highest_price = stage2_df['Close'].iloc[0]
    in_correction = False
    
    for idx, row in stage2_df.iterrows():
        price = row['Close']
        if price > highest_price:
            highest_price = price
            if in_correction:
                base_count += 1
                in_correction = False
        elif price < highest_price * 0.88:
            in_correction = True
            
    return min(base_count, 4)


def generate_chart_image(ticker, name, df_hist, w_point, m_point, base_stage, rs_rating):
    """[색상 보완] 현재 Base는 U자형 곡선, 직전 모든 Base는 기존 요청의 '타원형(선명화)'으로 복구"""
    df_plot = df_hist.tail(120).copy() 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df_plot.index, df_plot['Close'], label='현재가', color='#1e293b', linewidth=2)
    ax1.plot(df_plot.index, df_plot['MA20'], label='20일선', color='#ef4444', linestyle='--', alpha=0.6)
    ax1.plot(df_plot.index, df_plot['MA50'], label='50일선', color='#3b82f6', linestyle='--', alpha=0.5)
    ax1.plot(df_plot.index, df_plot['MA150'], label='150일선(와인스타인 기준)', color='#10b981', linewidth=2)
    ax1.plot(df_plot.index, df_plot['MA200'], label='200일선(미너비니 기준)', color='#8b5cf6', linewidth=1.5, alpha=0.5)

    if len(df_hist) >= 200:
        df_hist['MA200_slope'] = df_hist['MA200'].diff(5)
        stage2_df = df_hist[df_hist['MA200_slope'] > 0]
        
        if len(stage2_df) >= 20:
            bases_info = [] 
            current_base = 1
            base_start_idx = stage2_df.index[0]
            highest_price = stage2_df['Close'].iloc[0]
            in_correction = False
            
            for idx, row in stage2_df.iterrows():
                price_c = row['Close']
                if price_c > highest_price:
                    if in_correction:
                        bases_info.append((base_start_idx, idx, highest_price, current_base))
                        current_base += 1
                        base_start_idx = idx
                        in_correction = False
                    highest_price = price_c
                elif price_c < highest_price * 0.90:
                    in_correction = True
            
            bases_info.append((base_start_idx, stage2_df.index[-1], highest_price, current_base))
            
            for start_dt, end_dt, h_price, b_num in bases_info:
                if end_dt >= df_plot.index[0] and start_dt <= df_plot.index[-1]:
                    plot_start = max(start_dt, df_plot.index[0])
                    plot_end = min(end_dt, df_plot.index[-1])
                    try:
                        y_start = df_plot.loc[plot_start, 'Close']
                        y_end = df_plot.loc[plot_end, 'Close']
                        y_min = df_plot.loc[plot_start:plot_end, 'Close'].min()
                        y_max = df_plot.loc[plot_start:plot_end, 'Close'].max()
                    except KeyError:
                        continue
                    
                    x_start = mdates.date2num(plot_start)
                    x_end = mdates.date2num(plot_end)
                    x_mid = (x_start + x_end) / 2
                    width = x_end - x_start
                    
                    if b_num == base_stage:
                        # 현재 Base는 U자형 곡선(Path) 유지
                        y_control = y_min - ((y_max - y_min) * 0.3 if (y_max > y_min) else h_price * 0.05)
                        path_data = [
                            (patches.Path.MOVETO, (x_start, y_start)),
                            (patches.Path.CURVE3, (x_mid, y_control)),
                            (patches.Path.CURVE3, (x_end, y_end))
                        ]
                        codes, verts = zip(*path_data)
                        path = patches.Path(verts, codes)
                        ax1.add_patch(patches.PathPatch(path, edgecolor='#f59e0b', facecolor='none', lw=2.5, alpha=0.9, zorder=4))
                        ax1.text(mdates.num2date(x_mid), y_control, f" 현재 Base {b_num}기 (곡선 수렴) ", color='#d97706', fontsize=9, fontweight='bold', ha='center', va='top')
                    elif b_num < base_stage:
                        # [가독성 보완] 직전 Base 타원형의 선 색상을 선명한 인디고 블루(#4338ca)로 수정 및 불투명도 향상
                        ellipse = patches.Ellipse(xy=(x_mid, (y_max+y_min)/2), width=width, height=(y_max-y_min)*1.2,
                                                   edgecolor='#4338ca', facecolor='#e0e7ff', alpha=0.25, lw=2.0, linestyle='--', zorder=3)
                        ax1.add_patch(ellipse)
                        ax1.text(mdates.num2date(x_mid), y_min * 0.96, f" 직전 Base {b_num}기 ", color='#4338ca', fontsize=8, fontweight='bold', ha='center', va='top')

    info_text = f"▶ RS 상대강도: {rs_rating}점\n▶ 미너비니 타점: {m_point} [{base_stage}기]\n▶ 와인스타인 스테이지: {w_point}"
    ax1.text(0.02, 0.92, info_text, transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cbd5e1', alpha=0.9))
    ax1.set_title(f"📈 {name} ({ticker}) 정통 융합 스크리닝 차트", fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    colors = ['#ef4444' if row['Close'] >= row['Open'] else '#3b82f6' for idx, row in df_plot.iterrows()]
    ax2.bar(df_plot.index, df_plot['Volume'], color=colors, alpha=0.7, width=0.6)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(axis='both', labelsize=9)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def generate_combined_html_report(df_result, today_str, total_scanned, passed_count, chart_list):
    """최종 통합 웹 보고서 템플릿 마크업 빌더"""
    # 한국 시간(KST: UTC+9)으로 실행 생성 일시 계산
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime("%Y년 %m월 %d일 %H시 %M분")

    table_rows = ""
    for idx, row in df_result.iterrows():
        rank = idx + 1
        rating = row['추천등급']
        badge_style = "bg-danger text-white" if rating == "강력매수" else ("bg-primary text-white" if rating == "매수" else "bg-warning text-dark")
        vcp_active = "text-success font-bold" if row['변동성축소비율'] <= 0.70 else ""
        vol_active = "text-success font-bold" if row['거래량축소비율'] <= 0.70 else ""

        table_rows += f"""
        <tr>
            <td class="text-center font-bold" style="font-size: 1.1rem; color: #1e293b;">{rank}위</td>
            <td><span class="ticker-badge">{row['종목코드']}</span></td>
            <td><strong>{row['종목명']}</strong></td>
            <td>{row['현재가']:,}원</td>
            <td class="text-center"><span class="badge {badge_style}" style="padding: 6px 12px; border-radius: 20px; font-weight: bold;">{rating}</span></td>
            <td style="font-size: 0.9rem;">
                <strong>와인스타인:</strong> {row['와인스타인지점']}<br>
                <strong>미너비니:</strong> {row['미너비니지점']} <span class="badge bg-secondary">베이스 {row['미너비니베이스']}기</span>
            </td>
            <td class="text-center {vcp_active}">{row['변동성축소비율']}</td>
            <td class="text-center {vol_active}">{row['거래량축소비율']}</td>
            <td class="text-center text-danger"><strong>{row['최근거래량증가(배)']}배</strong></td>
            <td class="text-center font-bold text-primary" style="font-size: 1.1rem; background-color: #f0fdf4;">{row['RS상대강도(백분위)']}점</td>
            <td class="text-center text-dark font-bold" style="font-size: 1.05rem; background-color: #faf5ff;">
                <strong>{row['종합점수']}점</strong>
                <div style="font-size: 0.75rem; color: #6b7280; font-weight: normal; margin-top: 4px;">
                    RS:{row['score_rs']} | VCP:{row['score_vcp']} | Vol:{row['score_vol']} | PV:{row['score_pivot']}
                </div>
            </td>
        </tr>
        """
        
    chart_sections = ""
    for chart in chart_list:
        chart_sections += f"""
        <div class="row align-items-center border-bottom py-4 bg-white px-3 my-3 rounded-3 shadow-sm" style="display: flex;">
            <div style="flex: 0 0 25%; padding-right: 20px;">
                <h4 class="fw-bold text-dark mb-1" style="margin: 0 0 5px 0; font-size: 1.2rem;">{chart['rank']}위. {chart['name']}</h4>
                <p class="text-muted small mb-3" style="margin: 0 0 15px 0; color: #6c757d; font-size: 0.9rem;">[{chart['ticker']}]</p>
                <div class="p-3 bg-light rounded-3 mb-2" style="background: #f8fafc; padding: 15px; border-radius: 8px; font-size: 0.88rem; border: 1px solid #e2e8f0;">
                    <div style="margin-bottom: 8px;"><strong>추천 등급:</strong> <span class="badge bg-danger" style="background-color:#dc3545; color:white; padding:3px 8px; border-radius:10px;">{chart['rating']}</span></div>
                    <div style="margin-bottom: 8px;"><strong>미너비니 단계:</strong> 베이스 {chart['base_stage']}기 현황</div>
                    <div style="margin-bottom: 8px; color: #2563eb;"><strong>🔥 RS 상대강도:</strong> <strong>{chart['rs_rating']}점</strong></div>
                    <div style="margin-bottom: 8px;"><strong>150일선 이격:</strong> {chart['disparity']}%</div>
                    <div class="fw-bold text-primary" style="font-size: 1.05rem; border-top: 1px solid #ddd; padding-top: 5px; margin-top: 5px; font-weight: bold; color: #0d6efd;">종합 스코어: {chart['score']}점 / 100점</div>
                    <div class="mt-2 text-muted" style="font-size: 0.8rem; line-height: 1.4; color:#6c757d; margin-top:8px;">
                        <span style="display:block;">▪ 주도주 RS 점수: {chart['score_rs']}점 / 30</span>
                        <span style="display:block;">▪ VCP 압축 점수: {chart['score_vcp']}점 / 30</span>
                        <span style="display:block;">▪ 거래공백 점수: {chart['score_vol']}점 / 25</span>
                        <span style="display:block;">▪ 피벗매물대 점수: {chart['score_pivot']}점 / 15</span>
                    </div>
                </div>
            </div>
            <div style="flex: 0 0 75%; text-align: center;">
                <img src="data:image/png;base64,{chart['img_base64']}" class="img-fluid rounded border shadow-xs" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #dee2e6;" alt="차트">
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>미너비니 정통 VCP & 와인스타인 스테이지2 융합 초수익 리포트</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f4f6f9; font-family: 'Malgun Gothic', sans-serif; color: #334155; padding: 40px 0; }}
        .card {{ border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .theory-title {{ border-left: 5px solid #4f46e5; padding-left: 12px; font-weight: 700; }}
        .ticker-badge {{ background-color: #f1f5f9; color: #334155; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-weight: bold; }}
        .stat-card {{ background: linear-gradient(135deg, #4f46e5, #3b82f6); color: white; border-radius: 16px; padding: 25px; text-align: center; }}
        .table th {{ background-color: #f8fafc; color: #64748b; font-weight: 600; text-align: center; }}
        .font-bold {{ font-weight: bold; }}
        .update-time {{ background-color: #e2e8f0; color: #475569; display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.95rem; }}
    </style>
</head>
<body>
    <div class="container" style="max-width: 1350px;">
        <div class="text-center mb-5">
            <h1 class="fw-extrabold" style="color: #0f172a;">📈 정통 추세추종 융합(와인스타인 2Stage × 미너비니 VCP) 스크리너</h1>
            <p class="text-muted fs-5 mb-2">분석 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]} | 가짜 돌파를 배제한 완전체 시스템</p>
            <div class="update-time mt-1">⏰ 리포트 산출 일시: {now_kst} (KST)</div>
        </div>
        <div class="row mb-4">
            <div class="col-md-4"><div class="stat-card"><h5>총 스캔 후보군</h5><h2 class="display-5 fw-bold">{total_scanned}개</h2><p class="mb-0">테마 및 시총 상위 추출 종목</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #10b981, #059669);"><h5>융합 추세 필터 통과</h5><h2 class="display-5 fw-bold">{passed_count}개</h2><p class="mb-0">중장기 정배열 & RS 만족 종목</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #f59e0b, #d97706);"><h5>최종 타점 포착</h5><h2 class="display-5 fw-bold">{len(df_result)}개</h2><p class="mb-0">VCP 압축 또는 피벗 돌파 임박</p></div></div>
        </div>
        <div class="card p-4">
            <h3 class="theory-title mb-4">🔍 대가들의 계량적 조건 만족 주도주 종합 랭킹 리스트</h3>
            <div class="table-responsive">
                <table class="table table-hover align-middle">
                    <thead>
                        <tr>
                            <th>순위</th><th>종목코드</th><th>종목명</th><th>현재가</th><th>추천등급</th>
                            <th>예상 진입 타점</th><th>변동성축소</th><th>거래량축소</th><th>최근 거래량 증가율</th>
                            <th style="background-color: #e8f5e9;">RS 상대강도</th><th>종합점수 (세부항목)</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>
        <div class="card p-4">
            <h3 class="theory-title mb-4">📊 대가들의 기술적 정합성 입체 차트 분석</h3>
            <div class="container-fluid px-0">{chart_sections}</div>
        </div>
    </div>
</body>
</html>
"""
    file_html = f"정통_융합_추세돌파_리포트_{today_str}.html"
    with open(file_html, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    return file_html


def get_combined_screener():
    print("🚀 [와인스타인 2단계 X 미너비니 VCP 정통 스크리너] 시스템 가동")
    
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        df_krx = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        df_krx = df_krx[df_krx['Code'].str.len() == 6]
        
        marcap_col = next((c for c in ['MarCap', 'MarketCap', '시가총액'] if c in df_krx.columns), None)
        df_large_cap = df_krx.sort_values(by=marcap_col, ascending=False).head(400) if marcap_col else df_krx.head(400)
            
        theme_codes = get_dynamic_theme_stocks()
        df_theme_stocks = df_krx[df_krx['Code'].isin(theme_codes)]
        df_targets = pd.concat([df_large_cap, df_theme_stocks], ignore_index=True).drop_duplicates(subset=['Code'])
        
    except Exception as e:
        print(f"❌ 시장 기본 데이터 로드 에러: {e}")
        return None

    today = datetime.today()
    start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    total_scanned = len(df_targets)
    raw_rs_scores = {}
    history_cache = {}
    valid_targets = []
    
    print(f"\n⚡ 1단계: 전 종목 대상 정통 RS 상대강도 스코어 계산 중...")
    for idx, row in tqdm(df_targets.iterrows(), total=total_scanned):
        code = row['Code']
        try:
            df_hist = fdr.DataReader(code, start_date, end_date)
            if df_hist is None or len(df_hist) < 200:
                continue
            
            c_now = df_hist['Close'].iloc[-1]
            c_1m  = df_hist['Close'].iloc[-20]
            c_3m  = df_hist['Close'].iloc[-60]
            c_6m  = df_hist['Close'].iloc[-120]
            c_12m = df_hist['Close'].iloc[0]
            
            raw_rs = (((c_now - c_1m) / c_1m) * 4) + (((c_now - c_3m) / c_3m) * 2) + (((c_now - c_6m) / c_6m) * 2) + (((c_now - c_12m) / c_12m) * 2)
            raw_rs_scores[code] = raw_rs
            history_cache[code] = df_hist
            valid_targets.append(row)
        except Exception:
            continue

    if not raw_rs_scores:
        print("❌ 유효한 RS 연산 결과 데이터가 없습니다.")
        return None
        
    rs_series = pd.Series(raw_rs_scores)
    rs_ratings = (rs_series.rank(pct=True) * 100).round(1).to_dict()

    passed_count = 0
    screener_list = []
    
    print(f"\n⚙️ 2단계: 양대 거장의 조건 결합(와인스타인 2Stage ∩ 미너비니 트렌드 템플릿) 검증...")
    
    for row in valid_targets:
        code = row['Code']
        name = row['Name']
        df_hist = history_cache[code]
        rs_rating = rs_ratings.get(code, 0.0)
        
        if rs_rating < 70.0:
            continue

        try:
            df_hist['MA20'] = df_hist['Close'].rolling(window=20).mean()
            df_hist['MA50'] = df_hist['Close'].rolling(window=50).mean()
            df_hist['MA150'] = df_hist['Close'].rolling(window=150).mean()
            df_hist['MA200'] = df_hist['Close'].rolling(window=200).mean()
            
            high_52w = df_hist['High'].rolling(window=250, min_periods=1).max().iloc[-1]
            low_52w = df_hist['Low'].rolling(window=250, min_periods=1).min().iloc[-1]
            
            last_row = df_hist.iloc[-1]
            current_close = last_row['Close']
            
            ma20, ma50, ma150, ma200 = last_row['MA20'], last_row['MA50'], last_row['MA150'], last_row['MA200']
            if pd.isna([current_close, ma20, ma50, ma150, ma200]).any():
                continue

            w_stage2 = (current_close > ma150) and (ma150 > df_hist['MA150'].iloc[-20])
            m_template = (current_close > ma50) and (ma50 > ma150) and (ma150 > ma200) and (ma200 > df_hist['MA200'].iloc[-20])
            dist_from_high = ((high_52w - current_close) / high_52w) * 100
            dist_from_low = ((current_close - low_52w) / low_52w) * 100
            trend_safety = (dist_from_high <= 25.0) and (dist_from_low >= 30.0)

            if w_stage2 and m_template and trend_safety:
                passed_count += 1
                
                success, vcp_ratio, vol_shrink_ratio, m_point = detect_vcp_and_pivot(df_hist)
                if not success:
                    continue
                
                disparity_150 = round((current_close / ma150) * 100, 1)
                w_point = "Stage 2A (돌파 초입 우량)" if disparity_150 <= 112.0 else "Stage 2B (추세 확장 국면)"
                
                box_base_stage = calculate_minervini_base(df_hist)
                
                score_rs = round((rs_rating / 100) * 30)
                score_vcp = 30 if vcp_ratio <= 0.70 else 10
                score_vol = 25 if vol_shrink_ratio <= 0.70 else 10
                score_pivot = 15 if dist_from_high <= 10.0 else 5
                
                score = score_rs + score_vcp + score_vol + score_pivot
                rating = "강력매수" if score >= 80 else ("매수" if score >= 55 else "관망/유지")

                screener_list.append({
                    '종목코드': code, '종목명': name, '현재가': int(current_close), '추천등급': rating,
                    '와인스타인지점': w_point, '미너비니지점': m_point, '미너비니베이스': box_base_stage,
                    '변동성축소비율': vcp_ratio, '거래량축소비율': vol_shrink_ratio,
                    '최근거래량증가(배)': round(df_hist['Volume'].iloc[-1] / df_hist['Volume'].iloc[-2], 2),
                    '150일선이격도': disparity_150, 'RS상대강도(백분위)': rs_rating, '종합점수': score,
                    'score_rs': score_rs, 'score_vcp': score_vcp, 'score_vol': score_vol, 'score_pivot': score_pivot
                })
                tqdm.write(f" 🎯 [타점포착] {code} ({name}) | RS: {rs_rating}점 | VCP비율: {vcp_ratio} | 스코어: {score}점")
        except Exception:
            continue

    if not screener_list:
        print("\n❌ 양대 대가의 정통 조건(2단계 진입+VCP수렴)을 동시에 충족하는 종목이 오늘 시장에 없습니다.")
        return None
        
    df_result = pd.DataFrame(screener_list)
    df_result = df_result.sort_values(by=['종합점수', 'RS상대강도(백분위)'], ascending=[False, False]).reset_index(drop=True)
    df_result_for_charts = df_result.copy()
    
    df_result.index = df_result.index + 1
    df_result.index.name = '순위'
    
    today_str = datetime.today().strftime("%Y%m%d")
    df_result.to_csv(f"정통_융합_추세돌파_스크리닝_{today_str}.csv", index=True, encoding='utf-8-sig')
    
    print(f"\n📊 조건 만족 종목 ({len(df_result_for_charts)}개) 기술적 연동 차트 빌드 중...")
    chart_list = []
    
    for idx, row in df_result_for_charts.iterrows():
        code = row['종목코드']
        df_hist_target = history_cache.get(code)
        if df_hist_target is not None:
            img_base64 = generate_chart_image(
                ticker=code, name=row['종목명'], df_hist=df_hist_target,
                w_point=row['와인스타인지점'], m_point=row['미너비니지점'],
                base_stage=row['미너비니베이스'], rs_rating=row['RS상대강도(백분위)']
            )
            
            # 뉴스 크롤링을 제외하고 차트 데이터셋만 즉시 구축
            chart_list.append({
                'rank': idx + 1, 'ticker': code, 'name': row['종목명'], 'rating': row['추천등급'],
                'base_stage': row['미너비니베이스'], 'disparity': row['150일선이격도'],
                'rs_rating': row['RS상대강도(백분위)'], 'score': row['종합점수'], 'img_base64': img_base64,
                'score_rs': row['score_rs'], 'score_vcp': row['score_vcp'], 'score_vol': row['score_vol'], 'score_pivot': row['score_pivot']
            })
            
    file_html = generate_combined_html_report(df_result.reset_index(drop=True), today_str, total_scanned, passed_count, chart_list)
    print(f"\n🎉 [엔진 종료] 스크리닝서 출력 완료!\n🌐 HTML 종합 보고서: {file_html}")
    return df_result

if __name__ == "__main__":
    result = get_combined_screener()
