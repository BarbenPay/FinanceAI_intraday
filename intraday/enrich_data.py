import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
STOCKS_DIR = "data_1min"
MACRO_DIR = "data_macro_1min"
OUTPUT_DIR = "data_enriched_v4"

# Cible (Triple Barrier) adaptée au 1-min
BARRIER_PROFIT = 0.015 
BARRIER_STOP = 0.010   
BARRIER_TIMEOUT = 240  

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def prepare_dataframe(df):
    """Nettoie un DF: index datetime, tri, suppression doublons et timezones"""
    # Si l'index n'est pas datetime, on le convertit (soit via colonne 'date', soit index)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    # Retire la timezone pour éviter les conflits (UTC vs None)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    # Tri et suppression des doublons temporels (garder le dernier en cas de conflit)
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    return df

def load_macro_data():
    """Charge et fusionne toutes les macros en un seul DataFrame large"""
    print("⏳ Chargement des données Macro...")
    macro_files = glob.glob(os.path.join(MACRO_DIR, "*.csv"))
    
    # On crée une liste de DataFrames
    macro_dfs = []
    
    for f in macro_files:
        ticker = os.path.basename(f).replace('_1min.csv', '')
        try:
            df = pd.read_csv(f)
            df = prepare_dataframe(df)
            
            # On ne garde que le Close et le Volume
            df = df[['close', 'volume']].rename(columns={
                'close': f'macro_{ticker}_close',
                'volume': f'macro_{ticker}_vol'
            })
            
            # Calcul des rendements immédiatement (sur la time-series brute de la macro)
            df[f'macro_{ticker}_ret'] = df[f'macro_{ticker}_close'].pct_change()
            
            macro_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Erreur chargement macro {ticker}: {e}")

    if not macro_dfs:
        print("❌ Aucune donnée macro trouvée !")
        return pd.DataFrame()

    # Fusion intelligente : On fait un outer join pour avoir une timeline globale
    # Cela permet de ne rien perdre, on alignera plus tard sur l'action
    print("   Fusion des macros...")
    macro_global = pd.concat(macro_dfs, axis=1) # Concat sur les colonnes, aligne les index automatiquement
    
    # Remplissage : Si une macro a un trou, on garde la valeur d'avant
    macro_global.ffill(inplace=True)
    macro_global.dropna(how='all', inplace=True)
    
    print(f"✅ Macro globale chargée : {macro_global.shape} lignes")
    return macro_global

def add_massive_features(df):
    """Ajoute une tonne d'indicateurs techniques"""
    # Copie pour éviter les warnings de fragmentation
    df = df.copy()
    
    # 1. Moyennes Mobiles
    periods = [5, 9, 14, 21, 50, 200]
    for p in periods:
        df.ta.sma(length=p, append=True)
        df.ta.ema(length=p, append=True)
    
    # 2. Oscillateurs
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.cci(length=20, append=True)
    df.ta.willr(append=True)
    
    # 3. Volatilité
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    
    # 4. Momentum
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    
    # 5. Volume & Time
    df['ret_vol'] = df['volume'].pct_change()
    
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df.index.minute / 60)
    
    return df

def apply_triple_barrier(df):
    """Génère la cible BUY/SELL/WAIT"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = np.ones(len(df)) # 1 = WAIT
    
    for i in range(len(closes) - BARRIER_TIMEOUT):
        curr = closes[i]
        target_up = curr * (1 + BARRIER_PROFIT)
        target_down = curr * (1 - BARRIER_STOP)
        
        # Fenêtre future
        window_high = highs[i+1 : i+1+BARRIER_TIMEOUT]
        window_low = lows[i+1 : i+1+BARRIER_TIMEOUT]
        
        # On check si barrière touchée
        hit_up = np.where(window_high >= target_up)[0]
        hit_down = np.where(window_low <= target_down)[0]
        
        first_up = hit_up[0] if len(hit_up) > 0 else 99999
        first_down = hit_down[0] if len(hit_down) > 0 else 99999
        
        if first_up < first_down and first_up < 99999:
            labels[i] = 2 # BUY
        elif first_down < first_up and first_down < 99999:
            labels[i] = 0 # SELL
            
    labels[-BARRIER_TIMEOUT:] = np.nan
    df['Target'] = labels
    return df

# --- MAIN ---
if __name__ == "__main__":
    # 1. Charger la Macro
    macro_global = load_macro_data()
    
    stock_files = glob.glob(os.path.join(STOCKS_DIR, "*.csv"))
    print(f"🚀 Enrichissement de {len(stock_files)} actions...")
    
    for f in tqdm(stock_files):
        try:
            ticker = os.path.basename(f).replace('_1min.csv', '')
            df = pd.read_csv(f)
            df = prepare_dataframe(df)
            
            if len(df) < 1000: continue 
            
            # --- LA CLÉ DU SUCCÈS : ALIGNEMENT ---
            # On prend la macro, et on la force à s'adapter aux dates de l'Action
            # reindex(index_de_l_action) : Ne garde que les lignes correspondant à l'action
            # method='ffill' : Si la minute exacte n'existe pas en macro, prend la précédente
            macro_aligned = macro_global.reindex(df.index, method='ffill')
            
            # Il peut rester des NaN au tout début si l'action a des données avant la macro
            # On peut bfill (remplir avec la première valeur future) pour le début
            macro_aligned.bfill(inplace=True)
            
            # Fusion Colonnes
            df = pd.concat([df, macro_aligned], axis=1)
            
            # 2. Indicateurs Techniques
            df = add_massive_features(df)
            
            # 3. Cible
            df = apply_triple_barrier(df)
            
            # 4. Nettoyage Final
            df.dropna(inplace=True)
            
            # 5. Sauvegarde
            save_path = os.path.join(OUTPUT_DIR, f"{ticker}_enriched.csv")
            df.to_csv(save_path)
            
        except Exception as e:
            print(f"❌ Erreur {ticker}: {e}")

    print("\n🎉 Enrichissement terminé.")