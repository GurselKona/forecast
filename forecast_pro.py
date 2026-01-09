import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Uyarıları kapat
warnings.filterwarnings("ignore")

class VolatilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Finansal Volatilite Tahmin Aracı (GARCH)")
        self.root.geometry("550x700")
        
        # --- PENCEREYİ ÖNE GETİRME (YENİ EKLENEN KISIM) ---
        self.root.lift() # Pencereyi yukarı kaldır
        self.root.attributes('-topmost', True) # En üstte kalmaya zorla
        self.root.after_idle(self.root.attributes, '-topmost', False) # Zorlamayı kaldır (Kullanıcı başka yere tıklayabilsin)
        self.root.focus_force() # Klavyeyi/fareyi buraya odakla
        # --------------------------------------------------
        
        # --- DEĞİŞKENLER ---
        self.file_path = tk.StringVar()
        self.selected_asset = tk.StringVar()
        self.analysis_mode = tk.StringVar(value="FUTURE")
        self.future_days = tk.IntVar(value=30)
        self.expost_start = tk.StringVar(value="2025-01-01")
        self.expost_end = tk.StringVar(value="2025-12-31")
        self.status_var = tk.StringVar(value="Hazır - Lütfen Dosya Yükleyin")
        
        self.create_widgets()

    def create_widgets(self):
        # 1. DOSYA SEÇİMİ
        frame_file = ttk.LabelFrame(self.root, text="Veri Kaynağı (Excel)", padding=10)
        frame_file.pack(fill="x", padx=10, pady=5)
        
        btn_browse = ttk.Button(frame_file, text="Dosya Seç...", command=self.load_file)
        btn_browse.pack(side="left", padx=5)
        
        lbl_file = ttk.Label(frame_file, textvariable=self.file_path, font=("Arial", 8), foreground="gray")
        lbl_file.pack(side="left", padx=5)

        # 2. VARLIK SEÇİMİ
        frame_asset = ttk.LabelFrame(self.root, text="Analiz Edilecek Varlık", padding=10)
        frame_asset.pack(fill="x", padx=10, pady=5)
        
        self.combo_asset = ttk.Combobox(frame_asset, textvariable=self.selected_asset, state="readonly")
        self.combo_asset.pack(fill="x")
        self.combo_asset.set("Lütfen dosya seçin")

        # 3. MOD SEÇİMİ
        frame_mode = ttk.LabelFrame(self.root, text="Analiz Modu", padding=10)
        frame_mode.pack(fill="x", padx=10, pady=5)
        
        ttk.Radiobutton(frame_mode, text="Gelecek Tahmini (Future Forecast)", variable=self.analysis_mode, value="FUTURE", command=self.toggle_inputs).pack(anchor="w", pady=2)
        ttk.Radiobutton(frame_mode, text="Geçmiş Testi (Ex-Post Backtest)", variable=self.analysis_mode, value="EXPOST", command=self.toggle_inputs).pack(anchor="w", pady=2)

        # 4. PARAMETRELER
        self.frame_params = ttk.LabelFrame(self.root, text="Parametreler", padding=10)
        self.frame_params.pack(fill="x", padx=10, pady=5)
        
        # Future Inputs
        self.lbl_days = ttk.Label(self.frame_params, text="Kaç Gün İleri:")
        self.ent_days = ttk.Entry(self.frame_params, textvariable=self.future_days, width=10)
        
        # Expost Inputs
        self.lbl_start = ttk.Label(self.frame_params, text="Başlangıç (YYYY-MM-DD):")
        self.ent_start = ttk.Entry(self.frame_params, textvariable=self.expost_start, width=15)
        self.lbl_end = ttk.Label(self.frame_params, text="Bitiş (YYYY-MM-DD):")
        self.ent_end = ttk.Entry(self.frame_params, textvariable=self.expost_end, width=15)
        
        self.toggle_inputs()

        # 5. ÇALIŞTIR BUTONU
        btn_run = ttk.Button(self.root, text="ANALİZİ BAŞLAT", command=self.run_analysis)
        btn_run.pack(fill="x", padx=20, pady=20, ipady=5)
        
        # 6. DURUM ÇUBUĞU
        lbl_status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=5)
        lbl_status.pack(side="bottom", fill="x")

    def toggle_inputs(self):
        for widget in self.frame_params.winfo_children():
            widget.grid_forget()
            
        if self.analysis_mode.get() == "FUTURE":
            self.lbl_days.grid(row=0, column=0, padx=5, pady=5, sticky="e")
            self.ent_days.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        else:
            self.lbl_start.grid(row=0, column=0, padx=5, pady=5, sticky="e")
            self.ent_start.grid(row=0, column=1, padx=5, pady=5, sticky="w")
            self.lbl_end.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            self.ent_end.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if path:
            self.file_path.set(path)
            self.status_var.set("Dosya taranıyor...")
            self.root.update()
            
            try:
                # Başlıkları oku
                df_preview = pd.read_excel(path, nrows=5)
                cols = list(df_preview.columns)
                
                # 'Date' veya 'Tarih' içermeyen sütunları listele
                potential_assets = []
                for col in cols:
                    c_str = str(col).lower()
                    if 'date' not in c_str and 'tarih' not in c_str and 'unnamed' not in c_str:
                        potential_assets.append(col)
                
                if potential_assets:
                    self.combo_asset['values'] = potential_assets
                    self.combo_asset.current(0)
                    self.status_var.set(f"Dosya yüklendi. {len(potential_assets)} varlık bulundu.")
                else:
                    # Başlık bulamazsa tüm sütunları göster
                    self.combo_asset['values'] = cols
                    self.combo_asset.current(0)
                    self.status_var.set("Dosya yüklendi.")
                    
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya okunamadı: {e}")

    def get_clean_data(self, asset_name):
        # --- BU KISIM VERİ YAPINIZA GÖRE DÜZELTİLDİ ---
        path = self.file_path.get()
        if not path: return None
        
        try:
            df = pd.read_excel(path)
            cols = list(df.columns)
            
            # Varlığın kolon indeksi
            try:
                asset_idx = cols.index(asset_name)
            except ValueError:
                messagebox.showerror("Hata", "Sütun bulunamadı.")
                return None
            
            if asset_idx == 0:
                messagebox.showerror("Hata", "Seçilen sütun en başta, solunda tarih yok!")
                return None
                
            # Tarih hemen soldaki sütundur
            date_idx = asset_idx - 1
            
            # Konum (iloc) ile çekiyoruz
            data = df.iloc[:, [date_idx, asset_idx]].copy()
            data.columns = ['Date', 'Price']
            
            # Temizlik
            data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            
            data = data.dropna(subset=['Date', 'Price']).sort_values('Date').set_index('Date')
            
            if not data.empty:
                # Günlük frekans ve fillna
                data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='D')).ffill()
                # Log Getiri
                data['Returns'] = 100 * np.log(data['Price'] / data['Price'].shift(1))
                data = data.replace([np.inf, -np.inf], np.nan).dropna()
                return data
            else:
                raise ValueError("Veri boş.")
                
        except Exception as e:
            messagebox.showerror("Veri Hatası", str(e))
            return None

    def run_analysis(self):
        asset = self.selected_asset.get()
        if not asset or "Lütfen" in asset:
            messagebox.showwarning("Uyarı", "Lütfen bir varlık seçin.")
            return

        df = self.get_clean_data(asset)
        if df is None: return
        
        self.status_var.set(f"İşleniyor: {asset}...")
        self.root.update()

        mode = self.analysis_mode.get()
        
        try:
            if mode == "FUTURE":
                self.run_future(df, asset)
            else:
                self.run_expost(df, asset)
            self.status_var.set("İşlem Tamamlandı.")
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz hatası: {e}")
            self.status_var.set("Hata oluştu.")

    def run_future(self, df, asset_name):
        days = self.future_days.get()
        y = df['Returns'].values
        
        # Model
        model = arch_model(y, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        res = model.fit(disp='off')
        fc = res.forecast(horizon=days)
        
        forecast_vars = fc.variance.iloc[-1].values
        forecast_means = fc.mean.iloc[-1].values
        
        last_price = df['Price'].iloc[-1]
        last_date = df.index.max()
        
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        predictions = []
        
        current_cum_mu = 0
        current_cum_var = 0
        
        for i in range(days):
            current_cum_mu += forecast_means[i]
            current_cum_var += forecast_vars[i]
            cum_sigma = np.sqrt(current_cum_var)
            
            pred_p = last_price * np.exp(current_cum_mu / 100)
            low_p = last_price * np.exp((current_cum_mu - 1.96 * cum_sigma) / 100)
            high_p = last_price * np.exp((current_cum_mu + 1.96 * cum_sigma) / 100)
            
            predictions.append({'Date': future_dates[i], 'Price': pred_p, 'Low': low_p, 'High': high_p})
            
        pred_df = pd.DataFrame(predictions).set_index('Date')
        
        # Grafik
        plt.figure(figsize=(12, 7))
        history = df['Price'].iloc[-90:]
        plt.plot(history.index, history.values, label='Geçmiş (Son 90 Gün)', color='black')
        plt.plot(pred_df.index, pred_df['Price'], label='Tahmin (Ortalama)', color='red', linestyle='--')
        plt.fill_between(pred_df.index, pred_df['Low'], pred_df['High'], color='red', alpha=0.2, label='%95 Güven Aralığı')
        plt.title(f"{asset_name} - Gelecek {days} Günlük Tahmin")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def run_expost(self, df, asset_name):
        start_date = self.expost_start.get()
        end_date = self.expost_end.get()
        
        # Tarih kontrolü
        try:
            test_dates = df.loc[start_date:end_date].index
        except:
             messagebox.showerror("Hata", "Seçilen tarihler veride bulunamadı.")
             return
             
        if len(test_dates) == 0:
            messagebox.showerror("Hata", "Seçilen aralıkta veri yok.")
            return
            
        forecast_results = []
        
        # İlerlemeyi göstermek için basit döngü
        total = len(test_dates)
        step = max(1, total // 20) # Arayüzü çok dondurmamak için her adımda update etme
        
        for i, current_date in enumerate(test_dates):
            if i % step == 0:
                self.status_var.set(f"İşleniyor: {i}/{total} gün...")
                self.root.update()

            window_start = current_date - pd.Timedelta(days=365*5)
            train = df.loc[window_start : current_date - pd.Timedelta(days=1), 'Returns']
            
            if len(train) < 100: continue
            
            try:
                model = arch_model(train, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
                res = model.fit(disp='off')
                fc = res.forecast(horizon=7)
                
                cum_mu = np.sum(fc.mean.iloc[-1].values)
                cum_sigma = np.sqrt(np.sum(fc.variance.iloc[-1].values))
                
                p_start = df.loc[:current_date - pd.Timedelta(days=1), 'Price'].iloc[-1]
                
                pred = p_start * np.exp(cum_mu / 100)
                low = p_start * np.exp((cum_mu - 1.96 * cum_sigma) / 100)
                high = p_start * np.exp((cum_mu + 1.96 * cum_sigma) / 100)
                
                target_date = current_date + pd.Timedelta(days=6)
                actual = np.nan
                if target_date in df.index:
                    actual = df.loc[target_date, 'Price']
                    
                forecast_results.append({
                    'Date': current_date, 'Actual': actual, 'Pred': pred, 'Low': low, 'High': high
                })
            except:
                continue
                
        if not forecast_results:
            messagebox.showinfo("Bilgi", "Tahmin üretilemedi.")
            return
            
        res_df = pd.DataFrame(forecast_results).set_index('Date')
        
        plt.figure(figsize=(12, 7))
        plt.plot(res_df.index, res_df['Actual'], label='Gerçekleşen (T+7)', color='black', alpha=0.6)
        plt.plot(res_df.index, res_df['Pred'], label='Model Tahmini', color='blue', linestyle='--')
        plt.fill_between(res_df.index, res_df['Low'], res_df['High'], color='blue', alpha=0.15, label='Güven Aralığı')
        plt.title(f"{asset_name} - Ex-Post Doğrulama ({start_date} - {end_date})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = VolatilityApp(root)
    root.mainloop()