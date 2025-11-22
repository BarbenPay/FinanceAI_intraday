import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION V6.1 (MULTI-TIMEFRAME) ---
STOCKS_DIR = "data_1min"
MACRO_DIR = "data_macro_1min"
OUTPUT_DIR = "data_enriched_v6_heavy" # Nouveau dossier pour différencier

COMMON_START_DATE = "2022-02-03"

# Cible Triple Barrier
BARRIER_TIMEOUT = 30
ATR_PERIOD = 14


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
    df = df[df.index >= COMMON_START_DATE]
    return df

def load_macro_data():
    print(f"⏳ Chargement Macro (Mode V6.1)...")
    macro_files = glob.glob(os.path.join(MACRO_DIR, "*.csv"))
    macro_dfs = []
    
    for f in macro_files:
        ticker = os.path.basename(f).replace('_1min.csv', '')
        try:
            df = pd.read_csv(f)
            df = prepare_dataframe(df)
            
            # On crée plusieurs horizons pour la macro aussi !
            # Court terme (1h) et Moyen terme (4h)
            close = df['close']
            df[f'm_{ticker}_ret_1'] = np.log(close / close.shift(1))
            df[f'm_{ticker}_ret_60'] = np.log(close / close.shift(60)) 
            
            cols = [c for c in df.columns if c.startswith('m_')]
            macro_dfs.append(df[cols])
        except: pass

    if not macro_dfs: return pd.DataFrame()
    macro_global = pd.concat(macro_dfs, axis=1)
    macro_global.fillna(0, inplace=True)
    return macro_global

def get_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()

def apply_dynamic_barrier(df):
    atr = get_atr(df, ATR_PERIOD).values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = np.ones(len(df)) # Par défaut 1 (WAIT)
    
    # --- NOUVEAUX PARAMÈTRES SYMÉTRIQUES ---
    # On veut des mouvements forts dans les DEUX sens
    BARRIER_WIDTH = 4.0
    
    for i in range(len(closes) - BARRIER_TIMEOUT):
        curr_price = closes[i]
        curr_atr = atr[i]
        
        if np.isnan(curr_atr) or curr_atr == 0: continue
            
        # Barrières symétriques
        target_up = curr_price + (BARRIER_WIDTH * curr_atr)
        target_down = curr_price - (BARRIER_WIDTH * curr_atr)
        
        window_h = highs[i+1 : i+1+BARRIER_TIMEOUT]
        window_l = lows[i+1 : i+1+BARRIER_TIMEOUT]
        
        hit_up = np.where(window_h >= target_up)[0]
        hit_down = np.where(window_l <= target_down)[0]
        
        first_up = hit_up[0] if len(hit_up) > 0 else 99999
        first_down = hit_down[0] if len(hit_down) > 0 else 99999
        
        if first_up == 99999 and first_down == 99999:
            labels[i] = 1 # WAIT (Le marché n'a pas assez bougé)
        elif first_up < first_down:
            labels[i] = 2 # BUY (Vraie hausse détectée)
        elif first_down < first_up:
            labels[i] = 0 # SELL (Vraie baisse détectée)
        else:
            labels[i] = 1 # Conflit simultané -> On considère comme du bruit (WAIT)

    df['Target'] = labels
    return df

def add_massive_features(df):
    """
    V6.1 : STRATÉGIE MULTI-TIMEFRAME
    On génère chaque indicateur sur [Fast, Mid, Slow]
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # -- 1. LES BASES --
    df['log_ret'] = np.log(close / close.shift(1))
    
    # -- 2. TIMEFRAMES --
    # Fast (Scalping), Mid (Intraday), Slow (Trend)
    # Périodes adaptées à la minute
    windows_osci = [7, 14, 40]      # RSI, CCI, ADX
    windows_trend = [20, 50, 200]   # SMA, EMA, Bollinger
    windows_mom = [5, 15, 60]       # Slope, Momentum
    
    # -- 3. TREND FOLLOWERS --
    for w in windows_trend:
        # SMA & EMA Distances
        sma = ta.sma(close, length=w)
        ema = ta.ema(close, length=w)
        df[f'dist_sma_{w}'] = np.log(close / sma)
        df[f'dist_ema_{w}'] = np.log(close / ema)
        
        # Bollinger (Width & Pos)
        bb = ta.bbands(close, length=w, std=2)
        if bb is not None:
            bbp = next((c for c in bb.columns if c.startswith('BBP')), None)
            bbb = next((c for c in bb.columns if c.startswith('BBB')), None)
            if bbp: df[f'bb_pos_{w}'] = bb[bbp]
            if bbb: df[f'bb_width_{w}'] = bb[bbb]
            
        # Donchian (Breakout)
        donchian = ta.donchian(high, low, length=w)
        if donchian is not None:
            d_low = donchian.iloc[:, 0]
            d_up = donchian.iloc[:, 2]
            denom = (d_up - d_low).replace(0, 1)
            df[f'donchian_pos_{w}'] = (close - d_low) / denom

    # -- 4. OSCILLATEURS (Multi-périodes) --
    for w in windows_osci:
        # RSI
        df[f'rsi_{w}'] = ta.rsi(close, length=w) / 100.0
        
        # CCI (Normalisé)
        df[f'cci_{w}'] = ta.cci(high, low, close, length=w) / 300.0
        
        # ADX (Force)
        adx = ta.adx(high, low, close, length=w)
        if adx is not None:
            col = next((c for c in adx.columns if c.startswith('ADX')), None)
            if col: df[f'adx_{w}'] = adx[col] / 100.0
            
        # Aroon
        aroon = ta.aroon(high, low, length=w)
        if aroon is not None:
            au = next((c for c in aroon.columns if c.startswith('AROONU')), None)
            ad = next((c for c in aroon.columns if c.startswith('AROOND')), None)
            if au: df[f'aroon_up_{w}'] = aroon[au] / 100.0
            if ad: df[f'aroon_down_{w}'] = aroon[ad] / 100.0
            
        # Choppiness
        df[f'chop_{w}'] = ta.chop(high, low, close, length=w) / 100.0

    # -- 5. MOMENTUM & GÉOMÉTRIE --
    for w in windows_mom:
        # Slope (Pente)
        slope = ta.slope(close, length=w)
        df[f'slope_{w}'] = slope / close
        
        # Returns cumulés (Momentum pur)
        df[f'ret_{w}min'] = df['log_ret'].rolling(w).sum()

    # -- 6. MACD (Un seul réglage standard suffit souvent, mais on peut ajouter un rapide)
    macd = ta.macd(close) # 12, 26
    if macd is not None:
        h = next((c for c in macd.columns if c.startswith('MACDh')), None)
        if h: df['macd_hist'] = macd[h] / close

    # -- 7. TIME --
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    return df

# --- MAIN ---
if __name__ == "__main__":
    macro_global = load_macro_data()
    stock_files = glob.glob(os.path.join(STOCKS_DIR, "*.csv"))
    print(f"🚀 Enrichissement V6.1 (Heavy) sur {len(stock_files)} actions...")
    
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
            cols_final = [c for c in df.columns if c not in cols_to_drop]
            
            df_clean = df[cols_final].copy()
            
            # --- CORRECTION INFINIS ---
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_clean.dropna(inplace=True)
            
            df_clean.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_stationary.csv"))
            
        except Exception as e:
            print(f"❌ Erreur {ticker}: {e}")

    print(f"\n🎉 Terminé. Données V6.1 dans '{OUTPUT_DIR}'")