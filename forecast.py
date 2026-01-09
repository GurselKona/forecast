import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
import os
from tqdm import tqdm

# Uyarıları kapat
warnings.filterwarnings("ignore")

# =============================================================================
# --- KULLANICI AYARLARI (BURAYI DÜZENLEYİN) ---
# =============================================================================

file_path = 'prices.xlsx' 

# MOD SEÇİMİ: 'EXPOST' (Geçmiş Testi) veya 'FUTURE' (Gelecek Tahmini)
ANALYSIS_MODE = 'FUTURE' 

# --- AYAR 1: 'EXPOST' MODU İÇİN TARİH ARALIĞI ---
# Geçmişte modelin nasıl çalıştığını görmek istediğiniz aralık
EXPOST_START = '2025-12-31'
EXPOST_END   = '2026-1-9'

# --- AYAR 2: 'FUTURE' MODU İÇİN GÜN SAYISI ---
# Son veriden itibaren kaç gün ileriye tahmin yapılacak?
FUTURE_DAYS = 30  

# =============================================================================

def load_gold_data(path):
    """Excel'den altın verisini bulur, temizler ve datetime indeksli döner."""
    try:
        df = pd.read_excel(path)
        if not any(df.columns.astype(str).str.contains('Gold', case=False)):
            df = pd.read_excel(path, skiprows=1)
    except Exception as e:
        print(f"Dosya Okuma Hatası: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]
    gold_col = next((c for c in df.columns if 'gold' in c.lower()), None)
    if gold_col is None: return None
    
    gold_idx = list(df.columns).index(gold_col)
    date_col = df.columns[gold_idx - 1]

    data = df[[date_col, gold_col]].dropna()
    data.columns = ['Date', 'Gold']
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data.dropna(subset=['Date', 'Gold']).sort_values('Date').set_index('Date')

def run_analysis():
    if not os.path.exists(file_path):
        print(f"HATA: '{file_path}' dosyası bulunamadı!")
        return

    gold_raw = load_gold_data(file_path)
    if gold_raw is None or gold_raw.empty:
        print("Veri yüklenemedi.")
        return

    # Veriyi günlük frekansa oturt ve eksikleri doldur
    gold_daily = gold_raw.reindex(pd.date_range(gold_raw.index.min(), gold_raw.index.max(), freq='D')).ffill()
    
    # Log Getiriler
    gold_daily['Returns'] = 100 * np.log(gold_daily['Gold'] / gold_daily['Gold'].shift(1))
    gold_daily = gold_daily.replace([np.inf, -np.inf], np.nan).dropna(subset=['Returns'])
    
    print(f"Veri Seti Tarih Aralığı: {gold_daily.index.min().date()} - {gold_daily.index.max().date()}")

    # -------------------------------------------------------------------------
    # MOD 1: EX-POST ANALİZ (Geçmiş Veri Üzerinde Test)
    # -------------------------------------------------------------------------
    if ANALYSIS_MODE.upper() == 'EXPOST':
        print(f"\n--- MOD: EX-POST (GEÇMİŞ TESTİ) ---")
        print(f"Hedef Aralık: {EXPOST_START} ile {EXPOST_END} arası")
        
        test_dates = gold_daily.loc[EXPOST_START:EXPOST_END].index
        
        if len(test_dates) == 0:
            print("HATA: Seçilen tarih aralığında veri bulunamadı.")
            return

        forecast_results = []
        
        for current_date in tqdm(test_dates, desc="Ex-Post Analiz"):
            # Son 10 yıllık veriyi al
            window_start = current_date - pd.Timedelta(days=3652)
            train_data = gold_daily.loc[window_start : current_date - pd.Timedelta(days=1), 'Returns']
            y = train_data.values[np.isfinite(train_data.values)]
            
            if len(y) < 100: continue
            
            try:
                model = arch_model(y, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
                res = model.fit(disp='off')
                
                # 7 Günlük Tahmin (Expost analizde standart olarak haftalık bakıyoruz)
                horizon = 7
                fc = res.forecast(horizon=horizon)
                means = fc.mean.iloc[-1].values
                vars = fc.variance.iloc[-1].values
                
                # Fiyat Dönüşümü
                p_start = gold_daily.loc[:current_date - pd.Timedelta(days=1), 'Gold'].iloc[-1]
                
                cum_mu = np.sum(means)
                cum_sigma = np.sqrt(np.sum(vars))
                
                pred_end = p_start * np.exp(cum_mu / 100)
                low_end = p_start * np.exp((cum_mu - 1.96 * cum_sigma) / 100)
                high_end = p_start * np.exp((cum_mu + 1.96 * cum_sigma) / 100)
                
                # Gerçek Değer
                target_date = current_date + pd.Timedelta(days=horizon-1)
                actual_val = np.nan
                if target_date in gold_daily.index:
                    actual_val = gold_daily.loc[target_date, 'Gold']
                
                forecast_results.append({
                    'Date': current_date, # Tahmin tarihi
                    'Actual_Price_T+7': actual_val,
                    'Predicted_Price': pred_end,
                    'Lower_Bound': low_end,
                    'Upper_Bound': high_end
                })
            except:
                continue

        if forecast_results:
            res_df = pd.DataFrame(forecast_results).set_index('Date')
            
            plt.figure(figsize=(14, 7))
            plt.plot(res_df.index, res_df['Actual_Price_T+7'], label='Gerçekleşen Fiyat (T+7)', color='black', alpha=0.6)
            plt.plot(res_df.index, res_df['Predicted_Price'], label='Model Tahmini', color='blue', linestyle='--')
            plt.fill_between(res_df.index, res_df['Lower_Bound'], res_df['Upper_Bound'], color='blue', alpha=0.15, label='%95 Güven Aralığı')
            plt.title(f'Ex-Post Analiz: Haftalık GARCH Tahminleri ({EXPOST_START} - {EXPOST_END})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            print("Ex-post analiz tamamlandı ve grafik çizildi.")
        else:
            print("Tahmin üretilemedi.")

    # -------------------------------------------------------------------------
    # MOD 2: FUTURE FORECAST (Gelecek Tahmini)
    # -------------------------------------------------------------------------
    elif ANALYSIS_MODE.upper() == 'FUTURE':
        print(f"\n--- MOD: FUTURE (GELECEK TAHMİNİ) ---")
        print(f"Verideki son tarih ({gold_daily.index.max().date()}) baz alınarak {FUTURE_DAYS} gün ileriye tahmin yapılıyor...")
        
        # Tüm veriyi kullan
        y = gold_daily['Returns'].values
        y = y[np.isfinite(y)]
        
        try:
            model = arch_model(y, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
            res = model.fit(disp='off')
            
            # Gelecek N gün için varyans tahmini
            fc = res.forecast(horizon=FUTURE_DAYS)
            
            # Son günün (T) varyans tahminlerini al (h.1, h.2 ... h.N)
            forecast_vars = fc.variance.iloc[-1].values
            forecast_means = fc.mean.iloc[-1].values # Genelde sabittir
            
            last_price = gold_daily['Gold'].iloc[-1]
            last_date = gold_daily.index.max()
            
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(FUTURE_DAYS)]
            
            predictions = []
            
            # Kümülatif olarak ilerle
            current_cum_mu = 0
            current_cum_var = 0
            
            for i in range(FUTURE_DAYS):
                # O güne kadar olan kümülatif getiri ve varyans
                # Not: GARCH varyansları toplanırken basit toplama yaklaşımı kullanılır (Bağımsızlık varsayımı ile)
                current_cum_mu += forecast_means[i]
                current_cum_var += forecast_vars[i]
                
                cum_sigma = np.sqrt(current_cum_var)
                
                pred_p = last_price * np.exp(current_cum_mu / 100)
                low_p = last_price * np.exp((current_cum_mu - 1.96 * cum_sigma) / 100)
                high_p = last_price * np.exp((current_cum_mu + 1.96 * cum_sigma) / 100)
                
                predictions.append({
                    'Date': future_dates[i],
                    'Predicted_Price': pred_p,
                    'Lower_Bound': low_p,
                    'Upper_Bound': high_p
                })
            
            pred_df = pd.DataFrame(predictions).set_index('Date')
            
            # Grafik: Son 60 gün + Gelecek Tahmini
            recent_history = gold_daily['Gold'].iloc[-60:]
            
            plt.figure(figsize=(14, 7))
            
            # Geçmiş Veri
            plt.plot(recent_history.index, recent_history.values, label='Geçmiş (Son 60 Gün)', color='black')
            
            # Gelecek Tahmin Fanı
            plt.plot(pred_df.index, pred_df['Predicted_Price'], label='Gelecek Tahmini (Ortalama)', color='red', linestyle='--')
            plt.fill_between(pred_df.index, pred_df['Lower_Bound'], pred_df['Upper_Bound'], color='red', alpha=0.2, label='%95 Beklenti Aralığı')
            
            plt.title(f'GARCH Modeli: Gelecek {FUTURE_DAYS} Günlük Altın Fiyat Projeksiyonu')
            plt.xlabel('Tarih')
            plt.ylabel('Fiyat')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print("\nGelecek tahminleri üretildi:")
            print(pred_df.head())
            pred_df.to_csv(f'altin_gelecek_{FUTURE_DAYS}_gun_tahmini.csv')
            
        except Exception as e:
            print(f"Modelleme hatası: {e}")

if __name__ == "__main__":
    run_analysis()