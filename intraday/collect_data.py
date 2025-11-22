import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---

TIINGO_API_KEYS = [
    
]

# Jusqu'où remonter ?
GOAL_DATE = datetime(2021, 1, 1)

# Configuration des dossiers et des tickers associés
# On sépare bien les deux mondes pour garder tes deux dossiers distincts.
CONFIGS = [
    {
        "folder": "data_1min",
        "type": "stock", # Majoritairement des actions
        "tickers": [
            "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
            "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
            "MARA", "MSTR"
        ]
    },
    {
        "folder": "data_macro_1min",
        "type": "macro", # Mixte ETFs + Crypto
        "tickers": [
            "SPY", "QQQ", "VXX", "IEF", "UUP", "btcusd"
        ]
    }
]

# Variables globales de gestion d'état
CURRENT_KEY_INDEX = 0
CONSECUTIVE_429 = 0

# --- FONCTIONS UTILITAIRES ---

def get_file_path(folder, ticker):
    """Génère le chemin de fichier complet."""
    return os.path.join(folder, f"{ticker}_1min.csv")

def get_current_start_date(folder, ticker):
    """Lit le CSV dans le dossier spécifique et retourne la date la plus ancienne."""
    file_path = get_file_path(folder, ticker)
    
    if not os.path.exists(file_path):
        return datetime.now()
    
    try:
        df = pd.read_csv(file_path)
        if df.empty: return datetime.now()
        
        first_date = pd.to_datetime(df['date'].iloc[0])
        if first_date.tzinfo is not None:
            first_date = first_date.tz_localize(None)
            
        return first_date
    except Exception as e:
        print(f"⚠️ Erreur lecture {ticker} (Dossier: {folder}): {e}")
        return datetime.now()

def merge_and_save(folder, ticker, new_df):
    """Fusionne les nouvelles données avec l'existant dans le bon dossier."""
    file_path = get_file_path(folder, ticker)
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, index_col=0)
        old_df.index = pd.to_datetime(old_df.index)
        full_df = pd.concat([new_df, old_df])
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        full_df.sort_index(inplace=True)
    else:
        full_df = new_df

    full_df.to_csv(file_path)
    return full_df.index.min(), len(full_df)

def download_previous_segment(ticker, current_start_date):
    """
    Télécharge les données. 
    Détecte automatiquement si c'est une Crypto ou une Action/ETF
    pour utiliser le bon endpoint de l'API Tiingo.
    """
    global CURRENT_KEY_INDEX, CONSECUTIVE_429

    # On utilise 25 jours pour être sûr de ne pas manquer de jours (mois courts/longs)
    end_date = current_start_date
    start_date = end_date - timedelta(days=25)
    
    fmt = '%Y-%m-%d'
    current_key = TIINGO_API_KEYS[CURRENT_KEY_INDEX]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {current_key}'
    }

    # --- LOGIQUE DE SÉLECTION DE L'ENDPOINT ---
    # Si le ticker contient 'usd' (ex: btcusd, ethusd), on tape l'API Crypto
    is_crypto = "btcusd" in ticker.lower() or "ethusd" in ticker.lower()

    if is_crypto:
        url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={ticker}&startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min"
    else:
        url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min&columns=date,open,high,low,close,volume"
    
    try:
        # Petit print esthétique pour savoir ce qu'on fait
        type_icon = "🪙" if is_crypto else "📈"
        print(f"⬇️  {type_icon} {ticker}:DL {start_date.strftime(fmt)} -> {end_date.strftime(fmt)} ... ", end="")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            CONSECUTIVE_429 = 0
            data = response.json()
            
            if not data:
                print("⚠️  Vide (Pas de data).")
                return pd.DataFrame() # Vide

            # --- PARSING DIFFÉRENCIÉ ---
            if is_crypto:
                # L'API Crypto renvoie une liste d'objets [{'ticker':..., 'priceData': [...]}]
                if isinstance(data, list) and len(data) > 0 and 'priceData' in data[0]:
                    raw_data = data[0]['priceData']
                else:
                    print("⚠️ Format Crypto inattendu.")
                    return pd.DataFrame()
            else:
                # L'API IEX renvoie directement la liste
                raw_data = data

            df = pd.DataFrame(raw_data)
            if df.empty:
                print("⚠️  DataFrame vide après parsing.")
                return pd.DataFrame()

            # Standardisation
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # On garde uniquement les colonnes OHLCV classiques
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in df.columns if c in cols_to_keep]]
            
            print(f"✅ {len(df)} lignes.")
            return df

        elif response.status_code == 429:
            print(f"\n🛑 Limite atteinte (Clé {CURRENT_KEY_INDEX + 1}).")
            CONSECUTIVE_429 += 1
            
            # Rotation de clé
            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(TIINGO_API_KEYS)
            print(f"🔀 Changement vers clé {CURRENT_KEY_INDEX + 1}")

            if CONSECUTIVE_429 >= len(TIINGO_API_KEYS):
                print("❗ Toutes les clés sont fatiguées. Pause forcée de 60s.")
                time.sleep(60)
                CONSECUTIVE_429 = 0
            
            return None # Indique qu'il faut réessayer

        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Exception interne: {e}")
        return pd.DataFrame()

# --- BOUCLE PRINCIPALE ---

if __name__ == "__main__":
    # 1. Création des dossiers si nécessaire
    for config in CONFIGS:
        if not os.path.exists(config['folder']):
            os.makedirs(config['folder'])
            print(f"📁 Dossier créé : {config['folder']}")

    print(f"🏗️  Lancement de l'Archéologue Unifié. Objectif : {GOAL_DATE.date()}")

    while True:
        # 2. ANALYSE DE L'ETAT DE TOUS LES FICHIERS (Stocks + Macro)
        status_list = []
        finished_count = 0
        total_tickers = 0

        print("\n🔍 Analyse de l'ensemble du parc...")
        
        # On parcourt nos deux configurations (Stocks et Macro)
        for config in CONFIGS:
            folder = config['folder']
            for ticker in config['tickers']:
                total_tickers += 1
                start_date = get_current_start_date(folder, ticker)
                
                if start_date <= GOAL_DATE:
                    finished_count += 1
                else:
                    # On ajoute à la liste : (Ticker, Date, Dossier)
                    status_list.append({
                        'ticker': ticker,
                        'date': start_date,
                        'folder': folder
                    })

        # 3. CONDITION DE SORTIE
        if finished_count == total_tickers:
            print("\n🎉 MISSION ACCOMPLIE ! Tout l'historique (Stocks & Macro) est téléchargé.")
            break
            
        # 4. CHOIX DE LA CIBLE (Le plus récent, donc le moins complet)
        # On trie pour avoir la date la plus GRANDE en premier
        status_list.sort(key=lambda x: x['date'], reverse=True)
        
        target = status_list[0] # Le gagnant est ici
        target_ticker = target['ticker']
        target_date = target['date']
        target_folder = target['folder']
        
        print(f"🎯 Priorité : {target_ticker} (dans {target_folder}) | Début actuel : {target_date.date()}")
        
        # 5. ACTION
        new_data = download_previous_segment(target_ticker, target_date)
        
        if new_data is None:
            # Cas du 429 (Rate Limit), on boucle pour retenter (la clé a changé)
            time.sleep(1)
            continue
            
        if not new_data.empty:
            new_min, total_lines = merge_and_save(target_folder, target_ticker, new_data)
            print(f"   💾 Sauvegardé dans {target_folder}. Nouveau début : {new_min}")
        else:
            # 6. GESTION DES TROUS (Weekends, Jours fériés, ou Avant IPO)
            # Si vide, on insère une ligne 'dummy' (factice) pour marquer le terrain 
            # et forcer le script à reculer, sinon il boucle à l'infini sur la même date.
            dummy_date = target_date - timedelta(days=25)
            
            # Protection IPO : Si on est très loin dans le passé et que c'est toujours vide, 
            # c'est peut-être que l'action n'existait pas.
            # Ici, on se contente de reculer.
            
            print(f"   ⚠️ Période vide. Insertion marqueur à {dummy_date.date()} pour avancer.")
            dummy_df = pd.DataFrame({'open': [0]}, index=[dummy_date])
            dummy_df.index.name = 'date'
            merge_and_save(target_folder, target_ticker, dummy_df)

        # 7. TEMPO API
        time.sleep(1.5) # Un peu de repos pour Tiingo