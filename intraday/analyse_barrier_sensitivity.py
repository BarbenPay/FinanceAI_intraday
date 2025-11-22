import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data_1min"  # On travaille sur les données brutes
THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] # Les seuils à tester (Multiplicateur ATR)
BARRIER_TIMEOUT = 30    # Ton réglage actuel (30 min)
ATR_PERIOD = 14

def get_atr(df, period=14):
    """Calcule l'ATR (Volatilité)"""
    # On s'assure que les données sont float
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def analyze_thresholds():
    files = glob.glob(os.path.join(DATA_DIR, "*_1min.csv"))
    
    if not files:
        print(f"❌ Aucun fichier trouvé dans {DATA_DIR}")
        return

    # Dictionnaire pour stocker les résultats globaux
    # Structure : { 2.0: {0: 1500, 1: 500, 2: 1500}, ... }
    results = {th: {0: 0, 1: 0, 2: 0} for th in THRESHOLDS}
    
    total_files = len(files)
    print(f"🔬 Analyse de sensibilité sur {total_files} fichiers...")
    print(f"   seuils testés : {THRESHOLDS}")
    print(f"   Timeout : {BARRIER_TIMEOUT} min")

    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            # On nettoie vite fait
            df = df[['open', 'high', 'low', 'close']].dropna()
            
            # Calcul ATR (Une seule fois par fichier !)
            atr_series = get_atr(df, ATR_PERIOD)
            
            # Conversion numpy pour la vitesse extrême
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            atrs = atr_series.values
            
            n_samples = len(closes) - BARRIER_TIMEOUT
            if n_samples < 100: continue

            # On boucle sur les lignes (c'est le plus lent, on optimise)
            # Astuce : On peut pré-calculer les fenêtres
            # Mais restons simples et robustes
            
            for i in range(n_samples):
                curr_price = closes[i]
                curr_atr = atrs[i]
                
                if np.isnan(curr_atr) or curr_atr == 0: continue
                
                # Fenêtre future
                window_h = highs[i+1 : i+1+BARRIER_TIMEOUT]
                window_l = lows[i+1 : i+1+BARRIER_TIMEOUT]
                
                # Optimisation : On calcule les max/min de la fenêtre une fois
                max_h = np.max(window_h)
                min_l = np.min(window_l)
                
                # Maintenant on teste tous les seuils d'un coup
                for th in THRESHOLDS:
                    # Barrière Symétrique
                    target_up = curr_price + (th * curr_atr)
                    target_down = curr_price - (th * curr_atr)
                    
                    # Check rapide : est-ce que ça a touché au moins un des deux ?
                    touch_up = max_h >= target_up
                    touch_down = min_l <= target_down
                    
                    res = 1 # Default WAIT
                    
                    if not touch_up and not touch_down:
                        res = 1 # WAIT (Rien touché)
                    elif touch_up and not touch_down:
                        res = 2 # BUY
                    elif not touch_up and touch_down:
                        res = 0 # SELL
                    else:
                        # Les deux touchés : il faut savoir LEQUEL en premier
                        # C'est plus coûteux, on le fait seulement si nécessaire
                        hit_up = np.where(window_h >= target_up)[0][0]
                        hit_down = np.where(window_l <= target_down)[0][0]
                        
                        if hit_up < hit_down: res = 2
                        elif hit_down < hit_up: res = 0
                        else: res = 1 # Conflit exact -> Bruit
                    
                    results[th][res] += 1

        except Exception as e:
            print(f"⚠️ Erreur {f}: {e}")

    # --- AFFICHAGE DU RAPPORT ---
    print("\n" + "="*65)
    print(f"{'SEUIL (ATR)':<12} | {'SELL (0)':<12} | {'WAIT (1)':<12} | {'BUY (2)':<12} | {'% WAIT'}")
    print("="*65)
    
    for th in THRESHOLDS:
        counts = results[th]
        total = sum(counts.values())
        if total == 0: continue
        
        pct_wait = (counts[1] / total) * 100
        pct_sell = (counts[0] / total) * 100
        pct_buy = (counts[2] / total) * 100
        
        # On met en évidence la ligne qui s'approche de 33% WAIT
        marker = "✅" if 30 <= pct_wait <= 40 else ""
        
        print(f"{th:<12} | {pct_sell:5.1f}%       | {pct_wait:5.1f}%       | {pct_buy:5.1f}%       | {marker}")

    print("="*65)

if __name__ == "__main__":
    analyze_thresholds()