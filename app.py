# =========================================================
# AI 股票趨勢預測與綜合決策系統
# =========================================================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from prophet import Prophet
from datetime import datetime, timedelta
from gnews import GNews
import requests
import holidays
import matplotlib.font_manager as fm
import os
import sys

def setup_chinese_font():
    """解決 Matplotlib 中文顯示問題 (使用 Google 思源黑體)"""
    font_url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    font_path = 'NotoSansCJKtc-Regular.otf'

    if not os.path.exists(font_path):
        print("📥 正在下載 Google 官方繁體中文字型 (思源黑體)...")
        response = requests.get(font_url)
        with open(font_path, 'wb') as f:
            f.write(response.content)
        print("✅ 字型下載完成！")

    fm.fontManager.addfont(font_path)
    custom_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = custom_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 環境設定
    setup_chinese_font()
    
    # 2. 設定要分析的股票
    ticker_symbol = "2887.TW"  # 💡 只要在這裡修改股票代號即可
    print(f"🔄 正在與伺服器連線，準備下載 {ticker_symbol} 的資料，請稍候...")

    # ---------------------------------------------------------
    # 防呆與資料下載區塊 (包含爬取 Yahoo 奇摩股市中文名稱)
    # ---------------------------------------------------------
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(period="2y")

    if stock_data.empty:
        print("\n" + "❌"*20)
        print(f"【錯誤】無法從 Yahoo Finance 取得 {ticker_symbol} 的股價資料！")
        print("可能原因：1. 股票代號輸入錯誤  2. 已下市  3. 伺服器異常")
        sys.exit()

    stock_id = ticker_symbol.replace(".TW", "").replace(".TWO", "")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)

        start_idx = res.text.find('<title>') + 7
        end_idx = res.text.find('</title>')
        title_text = res.text[start_idx:end_idx]

        chinese_name = title_text.split('(')[0].strip()

        if chinese_name and "Yahoo" not in chinese_name:
            display_name = f"{ticker_symbol} {chinese_name}"
            stock_name_for_news = chinese_name
        else:
            display_name = ticker_symbol
            stock_name_for_news = stock_id
    except Exception:
        display_name = ticker_symbol
        stock_name_for_news = stock_id

    print(f"✅ 成功取得標的：【{display_name}】")

    # ---------------------------------------------------------
    # 總體經濟與產業週期 (大環境順逆風)
    # ---------------------------------------------------------
    macro_tickers = {
        '^TWII': '台灣加權指數 (大盤趨勢)',
        '^SOX': '費城半導體指數 (產業週期)',
        '^TNX': '美10年期公債殖利率 (資金成本)'
    }
    macro_results = {}
    macro_score = 0

    for sym, name in macro_tickers.items():
        try:
            m_data = yf.Ticker(sym).history(period="6mo")['Close']
            if not m_data.empty:
                curr_val = m_data.iloc[-1]
                ma60 = m_data.rolling(60).mean().iloc[-1]

                if sym == '^TNX':
                    is_tailwind = curr_val < ma60
                    status = "資金寬鬆 🟢" if is_tailwind else "資金緊縮 🔴"
                    macro_score += 1 if is_tailwind else -1
                else:
                    is_tailwind = curr_val > ma60
                    status = "多頭/擴張 🟢" if is_tailwind else "空頭/收縮 🔴"
                    macro_score += 1 if is_tailwind else -1

                macro_results[name] = f"目前 {curr_val:.2f} | 季線 {ma60:.2f} ➔ {status}"
        except Exception:
            macro_results[name] = "無法取得資料"

    if macro_score >= 2:
        env_status = "大環境順風 🌬️ (多頭動能強)"
    elif macro_score <= -2:
        env_status = "大環境逆風 🌪️ (系統性風險較高)"
    else:
        env_status = "大環境中性 ⚖️ (震盪整理)"

    # ---------------------------------------------------------
    # 基本面評估
    # ---------------------------------------------------------
    info = ticker.info
    dividend_yield = info.get('dividendYield', 0)
    trailing_yield = info.get('trailingAnnualDividendYield', 0)
    final_yield = dividend_yield if dividend_yield else trailing_yield
    yield_str = f"{final_yield:.2f}%" if final_yield > 1 else f"{final_yield * 100:.2f}%" if final_yield else "目前無提供或無配息"

    pe_ratio = info.get('trailingPE', None)
    pe_str = f"{pe_ratio:.2f} 倍" if pe_ratio else "無資料"
    eps = info.get('trailingEps', None)
    eps_str = f"{eps:.2f} 元" if eps else "無資料"
    pb_ratio = info.get('priceToBook', None)
    pb_str = f"{pb_ratio:.2f} 倍" if pb_ratio else "無資料"

    # ---------------------------------------------------------
    # 籌碼面 (FinMind 三大法人 + OBV 資金動能)
    # ---------------------------------------------------------
    chip_text = ""
    try:
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={stock_id}&start_date={start_date}"
        r = requests.get(url, timeout=5)
        chip_data = r.json()

        if chip_data.get('msg') == 'success' and len(chip_data.get('data', [])) > 0:
            df_chips = pd.DataFrame(chip_data['data'])
            df_chips['net_buy'] = (df_chips['buy'] - df_chips['sell']) / 1000
            recent_date = df_chips['date'].max()
            df_recent = df_chips[df_chips['date'] == recent_date]

            foreign = df_recent[df_recent['name'] == 'Foreign_Investor']['net_buy'].sum()
            trust = df_recent[df_recent['name'] == 'Investment_Trust']['net_buy'].sum()
            dealer = df_recent[df_recent['name'].str.contains('Dealer')]['net_buy'].sum()

            chip_text += f"   🔹 最新交易日 ({recent_date}) 三大法人買賣超：\n"
            chip_text += f"      外資：{foreign:,.0f} 張 | 投信：{trust:,.0f} 張 | 自營商：{dealer:,.0f} 張"
        else:
            chip_text += "   ⚠️ 無法取得三大法人最新數據。"
    except Exception:
        chip_text += "   ⚠️ 籌碼資料連線異常。"

    obv = [0]
    for i in range(1, len(stock_data)):
        if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i-1]:
            obv.append(obv[-1] + stock_data['Volume'].iloc[i])
        elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i-1]:
            obv.append(obv[-1] - stock_data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    stock_data['OBV'] = obv
    obv_trend = "資金流入 (大戶偏多) 📈" if stock_data['OBV'].iloc[-1] > stock_data['OBV'].iloc[-5] else "資金流出 (大戶偏空) 📉"

    # ---------------------------------------------------------
    # 資料前處理與 Prophet 模型預測
    # ---------------------------------------------------------
    df = stock_data.reset_index()
    df['Date'] = df['Date'].dt.tz_localize(None)
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet = df_prophet.dropna()

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,      # 🟢 啟用週季節性
        yearly_seasonality=True,      # 🟢 啟用年季節性
        changepoint_prior_scale=0.15, 
        changepoint_range=0.98        
    )
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    future = future[future['ds'].dt.weekday < 5] 
    forecast = model.predict(future)

    # ---------------------------------------------------------
    # 繪製趨勢主圖表
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"📊 【{display_name}】AI 預測圖表生成中...")
    print("="*60)

    fig1 = model.plot(forecast, figsize=(12, 6))
    ax = fig1.gca()

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='歷史實際收盤價 (Black Dots)')
    blue_line = mlines.Line2D([], [], color='#0072B2', linewidth=2, label='AI 預測主要趨勢 (Solid Blue Line)')
    light_blue_patch = mpatches.Patch(color='#0072B2', alpha=0.2, label='預測信賴區間 / 波動範圍 (Light Blue Area)')
    ax.legend(handles=[black_dot, blue_line, light_blue_patch], loc='best', fontsize=11, framealpha=0.9, edgecolor='gray')

    plt.title(f'{display_name} 股價 AI 預測與趨勢分析 (已啟用季節性機制)', fontsize=14, fontweight='bold')
    plt.xlabel('日期 (Month / Day)', fontsize=12)
    plt.ylabel('股價 (Price)', fontsize=12)

    ax.grid(which='major', color='gray', linestyle='-', alpha=0.4)
    ax.grid(which='minor', color='gray', linestyle=':', alpha=0.15)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 文字報告整合
    # ---------------------------------------------------------
    history_last_date = df_prophet['ds'].max()
    future_predictions = forecast[forecast['ds'] > history_last_date].copy()

    print("\n" + "★"*60)
    print(f"📄 【{display_name}】決策指揮中心 (分析基準: {history_last_date.strftime('%Y-%m-%d')})")
    print("★"*60)

    print("\n🌍 【總體經濟與產業週期 (大環境評估)】")
    print(f"   🚩 綜合環境判定：{env_status}")
    for name, status in macro_results.items():
        print(f"   🔹 {name}: {status}")

    print("\n💰 【基本面與籌碼面評估 (個股體質)】")
    print(f"   [基本面] EPS: {eps_str} | 本益比: {pe_str} | 淨值比: {pb_str} | 殖利率: {yield_str}")
    print(f"   [技術籌碼] OBV 近五日動能：{obv_trend}")
    print(chip
