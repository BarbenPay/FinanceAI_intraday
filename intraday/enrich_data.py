import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION V7.2 (ULTIMATE MERGE) ---
STOCKS_DIR = "data_1min"
MACRO_DIR = "data_macro_1min"
OUTPUT_DIR = "data_enriched_v7_sota" 
COMMON_START_DATE = "2022-02-03"

# Paramètres Triple Barrier
BARRIER_TIMEOUT = 30
ATR_PERIOD = 14
BARRIER_WIDTH = 4.0 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def prepare_dataframe(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    return df[df.index >= COMMON_START_DATE]

def load_macro_data():
    macro_files = glob.glob(os.path.join(MACRO_DIR, "*.csv"))
    macro_dfs = []
    for f in macro_files:
        ticker = os.path.basename(f).replace('_1min.csv', '')
        try:
            df = pd.read_csv(f)
            df = prepare_dataframe(df)
            close = df['close']
            df[f'm_{ticker}_ret_1'] = np.log(close / close.shift(1))
            df[f'm_{ticker}_ret_60'] = np.log(close / close.shift(60)) 
            cols = [c for c in df.columns if c.startswith('m_')]
            macro_dfs.append(df[cols])
        except: pass
    
    if not macro_dfs: return pd.DataFrame()
    return pd.concat(macro_dfs, axis=1).fillna(0)

def apply_dynamic_barrier(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = np.max(ranges, axis=1).rolling(ATR_PERIOD).mean().values

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = np.ones(len(df)) # 1 = WAIT
    
    for i in range(len(closes) - BARRIER_TIMEOUT):
        curr_price = closes[i]
        curr_atr = atr[i]
        
        if np.isnan(curr_atr) or curr_atr == 0: continue
            
        target_up = curr_price + (BARRIER_WIDTH * curr_atr)
        target_down = curr_price - (BARRIER_WIDTH * curr_atr)
        
        window_h = highs[i+1 : i+1+BARRIER_TIMEOUT]
        window_l = lows[i+1 : i+1+BARRIER_TIMEOUT]
        
        hit_up = np.where(window_h >= target_up)[0]
        hit_down = np.where(window_l <= target_down)[0]
        
        first_up = hit_up[0] if len(hit_up) > 0 else 99999
        first_down = hit_down[0] if len(hit_down) > 0 else 99999
        
        if first_up == 99999 and first_down == 99999: labels[i] = 1
        elif first_up < first_down: labels[i] = 2 # BUY
        elif first_down < first_up: labels[i] = 0 # SELL
        else: labels[i] = 1

    df['Target'] = labels
    return df

def add_massive_features(df):
    """
    V7.2 ULTIMATE : Fusion V6 (Multi-Timeframe) + V7 (SOTA)
    Total attendu : ~87 features stationnaires.
    """
    df = df.copy()
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    v = df['volume']

    # ==========================================
    # 🏛️ PARTIE 1 : SOCLE V6 (Multi-Timeframe)
    # ==========================================
    df['log_ret'] = np.log(c / c.shift(1))

    # Périodes
    windows_osci = [7, 14, 40]      # RSI, CCI, ADX
    windows_trend = [20, 50, 200]   # SMA, EMA, Bollinger
    windows_mom = [5, 15, 60]       # Slope, Momentum

    # -- TREND --
    for w in windows_trend:
        # SMA & EMA Distances (Stationnaire : log distance)
        sma = ta.sma(c, length=w)
        ema = ta.ema(c, length=w)
        df[f'dist_sma_{w}'] = np.log(c / sma)
        df[f'dist_ema_{w}'] = np.log(c / ema)
        
        # Bollinger (Pos & Width sont déjà stationnaires)
        bb = ta.bbands(c, length=w, std=2)
        if bb is not None:
            bbp = next((col for col in bb.columns if col.startswith('BBP')), None)
            bbb = next((col for col in bb.columns if col.startswith('BBB')), None)
            if bbp: df[f'bb_pos_{w}'] = bb[bbp]
            if bbb: df[f'bb_width_{w}'] = bb[bbb]
            
        # Donchian (Stationnarisé)
        donchian = ta.donchian(h, l, length=w)
        if donchian is not None:
            d_low = donchian.iloc[:, 0]
            d_up = donchian.iloc[:, 2]
            denom = (d_up - d_low).replace(0, 1)
            df[f'donchian_pos_{w}'] = (c - d_low) / denom

    # -- OSCILLATEURS --
    for w in windows_osci:
        # RSI
        df[f'rsi_{w}'] = ta.rsi(c, length=w) / 100.0
        # CCI (Normalisé)
        df[f'cci_{w}'] = ta.cci(h, l, c, length=w) / 300.0
        # ADX
        adx = ta.adx(h, l, c, length=w)
        if adx is not None:
            col = next((col for col in adx.columns if col.startswith('ADX')), None)
            if col: df[f'adx_{w}'] = adx[col] / 100.0
        # Aroon
        aroon = ta.aroon(h, l, length=w)
        if aroon is not None:
            au = next((col for col in aroon.columns if col.startswith('AROONU')), None)
            ad = next((col for col in aroon.columns if col.startswith('AROOND')), None)
            if au: df[f'aroon_up_{w}'] = aroon[au] / 100.0
            if ad: df[f'aroon_down_{w}'] = aroon[ad] / 100.0
        # Chop
        df[f'chop_{w}'] = ta.chop(h, l, c, length=w) / 100.0

    # -- MOMENTUM --
    for w in windows_mom:
        # Slope (Pente) -> Divisée par le prix pour être stationnaire !
        slope = ta.slope(c, length=w)
        df[f'slope_{w}'] = slope / c
        # Returns cumulés
        df[f'ret_{w}min'] = df['log_ret'].rolling(w).sum()

    # MACD
    macd = ta.macd(c)
    if macd is not None:
        h_macd = next((col for col in macd.columns if col.startswith('MACDh')), None)
        if h_macd: df['macd_hist'] = macd[h_macd] / c

    # Time Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)


    # ==========================================
    # 🚀 PARTIE 2 : AJOUTS V7 SOTA (30 New)
    # ==========================================
    
    # 1. PRICE ACTION (Pure)
    rng = (h - l).replace(0, 1e-6)
    df['candle_body_rel'] = abs(c - o) / rng
    df['shadow_up'] = (h - np.maximum(c, o)) / rng
    df['shadow_down'] = (np.minimum(c, o) - l) / rng
    df['gap_open'] = np.log(o / c.shift(1))

    # 2. VOLATILITÉ AVANCÉE
    # Garman-Klass
    log_hl = np.log(h / l.replace(0, 1e-6))
    log_co = np.log(c / o.replace(0, 1e-6))
    df['vol_gk'] = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    
    # Efficiency Ratio
    change = c.diff(10).abs()
    path = c.diff(1).abs().rolling(10).sum()
    df['efficiency_ratio'] = change / (path + 1e-6)

    # 3. SMART MONEY (Volume)
    # MFI (déjà dans loop ? non mfi n'y était pas, on l'ajoute)
    df['mfi_14_sota'] = ta.mfi(h, l, c, v, length=14) / 100.0
    df['cmf_20'] = ta.cmf(h, l, c, v, length=20)
    
    # Force Index Normalisé
    fi = (c - c.shift(1)) * v
    df['force_idx'] = ta.ema(fi, length=13) / (v.rolling(20).mean() + 1e-6)
    
    # VWAP Distance (Ton Alpha V6.5)
    vwap_d = ta.vwap(h, l, c, v, anchor='D')
    df['dist_vwap'] = (c - vwap_d) / (vwap_d + 1e-6)
    df['dist_vwap'] = df['dist_vwap'].fillna(0)

    # Volume Z-Score
    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std()
    df['vol_zscore'] = (v - vol_mean) / (vol_std + 1e-6)

    # 4. MOMENTUM SOTA
    # Trix
    trix_df = ta.trix(c, length=14)
    if trix_df is not None: df['trix'] = trix_df.iloc[:, 0]
    
    # Williams %R
    df['willr_14'] = ta.willr(h, l, c, length=14) / -100.0
    
    # Fisher Transform
    fisher = ta.fisher(h, l, length=9)
    if fisher is not None: df['fisher_t'] = fisher.iloc[:, 0]

    # 5. STATS (Distribution)
    df['skew_20'] = c.rolling(20).skew()
    df['kurt_20'] = c.rolling(20).kurt()
    
    # Z-Score Price
    sma_20 = ta.sma(c, length=20)
    std_20 = c.rolling(20).std()
    df['zscore_price'] = (c - sma_20) / (std_20 + 1e-6)

    # Remplissage des NaN (dus aux rolling windows)
    df.fillna(0, inplace=True)

    return df

if __name__ == "__main__":
    macro_global = load_macro_data()
    stock_files = glob.glob(os.path.join(STOCKS_DIR, "*.csv"))
    print(f"🚀 Enrichissement V7.2 (ULTIMATE MERGE) sur {len(stock_files)} actions...")
    
    for f in tqdm(stock_files):
        try:
            ticker = os.path.basename(f).replace('_1min.csv', '')
            df = pd.read_csv(f)
            df = prepare_dataframe(df)
            
            if len(df) < 1000: continue 
            
            if not macro_global.empty:
                macro_aligned = macro_global.reindex(df.index, method='ffill').fillna(0)
                df = pd.concat([df, macro_aligned], axis=1)
            
            df = apply_dynamic_barrier(df)
            df = add_massive_features(df)
            
            cols_to_drop = ['open', 'high', 'low', 'close', 'volume']
            df_clean = df.drop(columns=cols_to_drop, errors='ignore')
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_clean.dropna(inplace=True)
            
            df_clean.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_stationary.csv"))
            
        except Exception as e:
            print(f"❌ Erreur {ticker}: {e}")