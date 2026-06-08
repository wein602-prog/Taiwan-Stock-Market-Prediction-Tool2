import streamlit as st
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
import tempfile

# --- 網頁設定 ---
st.set_page_config(page_title="AI 股票決策指揮中心", page_icon="📈", layout="centered")
st.title("📈 AI 股票預測與決策指揮中心")

# --- 🛠️ 防呆優化 1：安全寫入字型 (避免雲端權限問題) ---
@st.cache_resource
def load_font():
    font_url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    font_path = os.path.join(tempfile.gettempdir(), 'NotoSansCJKtc-Regular.otf')
    
    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url, timeout=10)
            with open(font_path, 'wb') as f:
                f.write(response.content)
        except Exception:
            return
            
    try:
        fm.fontManager.addfont(font_path)
        custom_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = custom_font.get_name() 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        pass

load_font()

# --- 側邊欄：使用者輸入 ---
st.sidebar.header("設定區")
ticker_symbol = st.sidebar.text_input("請輸入股票代號 (例如: 2887.TW)", value="2887.TW")
analyze_button = st.sidebar.button("🚀 開始分析")

# --- 主程式區塊 ---
if analyze_button:
    with st.spinner(f"正在連線伺服器，全力運算 {ticker_symbol} 的數據中..."):
        
        # ==========================================
        # 1. 取得股價資料與中文名稱
        # ==========================================
        ticker = yf.Ticker(ticker_symbol)
        stock_data = ticker.history(period="2y")
        
        if stock_data.empty:
            st.error(f"❌ 無法從 Yahoo Finance 取得 {ticker_symbol} 的股價資料！請確認代號是否正確。")
            st.stop()
            
        stock_id = ticker_symbol.replace(".TW", "").replace(".TWO", "")
        
        # 🛠️ 防呆優化 2：增加爬蟲安全防護
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
            res = requests.get(url, headers=headers, timeout=5)
            
            start_idx = res.text.find('<title>')
            if start_idx != -1:
                start_idx += 7
                end_idx = res.text.find('</title>')
                chinese_name = res.text[start_idx:end_idx].split('(')[0].strip()
                display_name = f"{ticker_symbol} {chinese_name}" if "Yahoo" not in chinese_name else ticker_symbol
                stock_name_for_news = chinese_name if "Yahoo" not in chinese_name else stock_id
            else:
                display_name, stock_name_for_news = ticker_symbol, stock_id
        except:
            display_name, stock_name_for_news = ticker_symbol, stock_id

        st.success(f"✅ 成功取得標的：【{display_name}】")

        # ==========================================
        # 2. 總體經濟環境
        # ==========================================
        macro_tickers = {'^TWII': '台灣加權指數', '^SOX': '費城半導體指數', '^TNX': '美10年期公債殖利率'}
        macro_results, macro_score = {}, 0
        for sym, name in macro_tickers.items():
            try:
                m_data = yf.Ticker(sym).history(period="6mo")['Close']
                if not m_data.empty:
                    curr_val = m_data.iloc[-1]
                    ma60 = m_data.rolling(60).mean().iloc[-1]
                    if sym == '^TNX':
                        is_tailwind = curr_val < ma60
                        macro_score += 1 if is_tailwind else -1
                    else:
                        is_tailwind = curr_val > ma60
                        macro_score += 1 if is_tailwind else -1
                    status = "🟢 偏多/寬鬆" if is_tailwind else "🔴 偏空/緊縮"
                    macro_results[name] = f"目前 {curr_val:.2f} | 季線 {ma60:.2f} ➔ {status}"
            except:
                macro_results[name] = "⚠️ 無法取得"

        env_status = "大環境順風 🌬️ (多頭動能強)" if macro_score >= 2 else "大環境逆風 🌪️ (系統性風險較高)" if macro_score <= -2 else "大環境中性 ⚖️ (震盪整理)"

        # ==========================================
        # 3. 基本面資料
        # ==========================================
        info = ticker.info
        dividend_yield = info.get('dividendYield', 0)
        trailing_yield = info.get('trailingAnnualDividendYield', 0)
        final_yield = dividend_yield if dividend_yield else trailing_yield
        yield_str = f"{final_yield:.2f}%" if final_yield > 1 else f"{final_yield * 100:.2f}%" if final_yield else "無資料"

        pe_ratio = info.get('trailingPE', None)
        pe_str = f"{pe_ratio:.2f} 倍" if pe_ratio else "無資料"
        eps = info.get('trailingEps', None)
        eps_str = f"{eps:.2f} 元" if eps else "無資料"
        pb_ratio = info.get('priceToBook', None)
        pb_str = f"{pb_ratio:.2f} 倍" if pb_ratio else "無資料"

        # ==========================================
        # 4. FinMind 三大法人與 OBV 籌碼
        # ==========================================
        chip_text = ""
        try:
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={stock_id}&start_date={start_date}"
            r = requests.get(url, timeout=5)
            chip_data = r.json()

            if chip_data.get('msg') == 'success' and len(chip_data.get('data', [])) > 0:
                df_chips = pd.DataFrame(chip_data['data'])
                # 🛠️ 防呆優化 3：確保欄位存在
                if 'buy' in df_chips.columns and 'sell' in df_chips.columns:
                    df_chips['net_buy'] = (df_chips['buy'] - df_chips['sell']) / 1000
                    recent_date = df_chips['date'].max()
                    df_recent = df_chips[df_chips['date'] == recent_date]

                    foreign = df_recent[df_recent['name'] == 'Foreign_Investor']['net_buy'].sum()
                    trust = df_recent[df_recent['name'] == 'Investment_Trust']['net_buy'].sum()
                    dealer = df_recent[df_recent['name'].str.contains('Dealer')]['net_buy'].sum()

                    chip_text = f"最新 ({recent_date})：外資 **{foreign:,.0f}** 張 | 投信 **{trust:,.0f}** 張 | 自營商 **{dealer:,.0f}** 張"
                else:
                    chip_text = "⚠️ 法人資料格式異動"
            else:
                chip_text = "⚠️ 無法取得三大法人最新數據"
        except:
            chip_text = "⚠️ 籌碼資料連線異常"

        # 🛠️ 防呆優化 4：確保 OBV 計算長度安全
        obv = [0]
        if len(stock_data) > 1:
            for i in range(1, len(stock_data)):
                if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + stock_data['Volume'].iloc[i])
                elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - stock_data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
        stock_data['OBV'] = obv
        
        if len(stock_data) >= 5:
            obv_trend = "🟢 資金流入 (大戶偏多)" if stock_data['OBV'].iloc[-1] > stock_data['OBV'].iloc[-5] else "🔴 資金流出 (大戶偏空)"
        else:
            obv_trend = "⚪ 資料不足無法計算"

        # =========================================================
        # 5. Prophet 模型預測
        # =========================================================
        df = stock_data.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)
        df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()
        
        model = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=False,     
            yearly_seasonality=False,     
            changepoint_prior_scale=0.15, 
            changepoint_range=0.98        
        )
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30) 
        future = future[future['ds'].dt.weekday < 5] 
        forecast = model.predict(future)

        # =========================================================
        # 🟢 輸出畫面：圖表區
        # =========================================================
        st.subheader("📊 AI 趨勢預測圖")
        
        # 🛠️ 防呆優化 5：安全繪圖，避免 Matplotlib 產生執行緒衝突
        fig1, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax.xaxis.set_minor_locator(mdates.DayLocator())

        black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='歷史實際收盤價')
        blue_line = mlines.Line2D([], [], color='#0072B2', linewidth=2, label='AI 預測主要趨勢')
        light_blue_patch = mpatches.Patch(color='#0072B2', alpha=0.2, label='預測信賴區間 / 波動範圍')
        ax.legend(handles=[black_dot, blue_line, light_blue_patch], loc='best', fontsize=9, framealpha=0.9, edgecolor='gray')

        plt.title(f'{display_name} 股價 AI 預測與趨勢分析', fontsize=14, fontweight='bold')
        plt.xlabel('日期 (Month / Day)', fontsize=12)
        plt.ylabel('股價 (Price)', fontsize=12)
        
        ax.grid(which='major', color='gray', linestyle='-', alpha=0.4)
        ax.grid(which='minor', color='gray', linestyle=':', alpha=0.15)
        st.pyplot(fig1)

        # =========================================================
        # 🟢 輸出畫面：決策指揮中心報告
        # =========================================================
        st.markdown("---")
        st.subheader("📄 決策指揮中心報告")
        
        st.markdown(f"**🌍 總經大環境：** {env_status}")
        for name, status in macro_results.items():
            st.write(f"🔹 {name}: {status}")
            
        st.write("") 
            
        st.markdown("**💰 基本面評估：**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EPS", eps_str)
        col2.metric("本益比", pe_str)
        col3.metric("淨值比", pb_str)
        col4.metric("殖利率", yield_str)

        st.markdown("**🕵️‍♂️ 籌碼動能：**")
        st.write(f"- OBV 近五日動能：{obv_trend}")
        st.write(f"- 三大法人：{chip_text}")
            
        # =========================================================
        # 🟢 輸出畫面：AI 推演
        # =========================================================
        st.markdown("---")
        st.subheader("📈 AI 未來 5 個有效交易日推演")
        
        history_last_date = df_prophet['ds'].max()
        future_predictions = forecast[forecast['ds'] > history_last_date].copy()
        
        first_price, last_price = None, None
        valid_days = 0
        day_mapping = {0: '週一', 1: '週二', 2: '週三', 3: '週四', 4: '週五', 5: '週六', 6: '週日'}
        
        # 🛠️ 防呆優化 6：相容不同版本的 holidays 套件
        try:
            tw_holidays = holidays.country_holidays('TW', years=[datetime.now().year, datetime.now().year + 1])
        except:
            tw_holidays = holidays.TW(years=[datetime.now().year, datetime.now().year + 1])
        
        for _, row in future_predictions.iterrows():
            if valid_days >= 5: break
            curr_date = row['ds']
            weekday = curr_date.weekday()
            
            # 🛠️ 防呆優化 7：將 Timestamp 轉為純 date 格式，避免比對錯誤
            if curr_date.date() in tw_holidays: 
                holiday_name = tw_holidays.get(curr_date.date())
                st.warning(f"📅 **{curr_date.strftime('%Y-%m-%d')} ({day_mapping[weekday]})** | 🛑 今日休市 ({holiday_name})，暫無交易預測")
                continue
            
            if first_price is None: first_price = row['yhat']
            last_price = row['yhat']
            
            st.info(f"📅 **{curr_date.strftime('%Y-%m-%d')} ({day_mapping[weekday]})** | 期望價: **${row['yhat']:.2f}** (區間: ${row['yhat_lower']:.2f} ~ ${row['yhat_upper']:.2f})")
            valid_days += 1

        # =========================================================
        # 🟢 輸出畫面：綜合策略建議
        # =========================================================
        st.markdown("---")
        st.subheader("💡 AI 操盤總結與策略建議")
        is_macro_good = macro_score >= 0 
        is_trend_up = last_price > first_price if (last_price and first_price) else False 
        is_obv_good = "流入" in obv_trend 
        
        st.write("📌 **當前盤勢結構 (多空交叉比對)：**")
        st.write(f"1. 總經大環境：{'偏多 🟢' if is_macro_good else '偏空 / 保守 🔴'}")
        st.write(f"2. 技術與籌碼：{'大戶資金流入 🟢' if is_obv_good else '大戶資金流出 🔴'}")
        st.write(f"3. AI 短期預測：{'趨勢向上 🟢' if is_trend_up else '趨勢向下 🔴'}")
        
        st.write("🎯 **綜合行動建議：**")
        if is_macro_good and is_trend_up and is_obv_good:
            st.success("🔥 **【強勢多頭格局 - 積極做多】**\n\n狀態：大環境順風、大戶資金持續進駐，且 AI 預測未來走高，個股處於極佳的攻擊位置。\n\n策略：可順勢放大資金部位。持有者建議續抱讓利潤奔跑；空手者若遇盤中量縮回檔，可視為切入點。")
        elif not is_macro_good and is_trend_up and is_obv_good:
            st.info("⚡ **【逆風突圍 / 跌深反彈 - 短線偏多】**\n\n狀態：大盤環境不佳，但該股有特定資金逆勢進駐，AI 預測短線有上漲空間。\n\n策略：屬於「逆勢抗跌股」。建議以「短進短出」為主，嚴格控制資金水位，並設定移動停利點。")
        elif is_macro_good and not is_trend_up and not is_obv_good:
            st.warning("⚠️ **【順風修正 / 獲利了結 - 觀望回檔】**\n\n狀態：大環境好，但個股出現資金提款跡象，AI 預測短期將面臨向下修正。\n\n策略：切勿因大盤好就盲目接刀。建議空手者先觀望，等待量縮止跌；持有者可考慮逢高減碼。")
        elif not is_macro_good and not is_trend_up and not is_obv_good:
            st.error("❄️ **【弱勢空頭格局 - 嚴控風險】**\n\n狀態：系統性風險較高，個股遭到大戶拋售，AI 預測趨勢全面走弱。\n\n策略：建議保持高現金水位，切勿輕易往下攤平。耐心等待總經大環境翻轉。")
        else:
            st.info(f"⚖️ **【多空交戰 / 震盪整理 - 區間操作或回歸基本面】**\n\n狀態：各項指標出現分歧。盤勢正處於方向選擇的過渡期。\n\n策略：雜訊較多，建議採取「區間高出低進」。若標的為 ETF，可檢視目前殖利率 ({yield_str}) 是否符合存股預期，採定期定額佈局；若想賺價差，建議縮手等待明確信號。")
            
        # =========================================================
        # 🟢 輸出畫面：完整新聞列表
        # =========================================================
        st.markdown("---")
        st.subheader("📰 近期相關新聞")
        try:
            google_news = GNews(language='zh-Hant', country='TW', max_results=3)
            news_items = google_news.get_news(f"{stock_name_for_news} 股票")
            if news_items:
                for i, news in enumerate(news_items, 1):
                    title = news.get('title', '無標題')
                    publisher = news.get('publisher', {}).get('title', '未知來源')
                    pub_date = str(news.get('published date', ''))[:16]
                    url = news.get('url', '#')
                    
                    st.markdown(f"**{i}. [{title}]({url})**")
                    st.caption(f"來源: {publisher} | 時間: {pub_date}")
            else:
                st.write("目前找不到最新新聞。")
        except:
            st.write("新聞模組發生異常。")
