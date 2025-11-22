import requests
import pandas as pd
import os
import time
import json
from datetime import datetime, timedelta

# --- CONFIGURATION ---

TIINGO_API_KEYS = [
    # Ajoute tes clés API ici
    "TA_CLE_API_ICI" 
]

# Jusqu'où remonter ?
GOAL_DATE = datetime(2021, 1, 1)

# Fichier de suivi pour gérer les trous sans polluer les CSV
STATE_FILE = "download_state.json"

# Configuration des dossiers et des tickers associés
CONFIGS = [
    {
        "folder": "data_1min",
        "type": "stock",
        "tickers": [
            "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
            "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
            "MARA", "MSTR"
        ]
    },
    {
        "folder": "data_macro_1min",
        "type": "macro",
        "tickers": [
            "SPY", "QQQ", "VXX", "IEF", "UUP", "btcusd"
        ]
    }
]

# Variables globales de gestion d'état
CURRENT_KEY_INDEX = 0
CONSECUTIVE_429 = 0

# --- FONCTIONS DE GESTION D'ÉTAT (JSON) ---

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def update_tracker(ticker, last_date_checked):
    """Met à jour la date atteinte pour ce ticker dans le fichier JSON"""
    state = load_state()
    # On stocke la date en string ISO pour le JSON
    # Si last_date_checked est déjà une string, on ne touche pas
    if isinstance(last_date_checked, str):
        state[ticker] = last_date_checked
    else:
        state[ticker] = last_date_checked.isoformat()
    save_state(state)

# --- FONCTIONS UTILITAIRES ---

def get_file_path(folder, ticker):
    """Génère le chemin de fichier complet."""
    return os.path.join(folder, f"{ticker}_1min.csv")

def get_current_start_date(folder, ticker):
    """
    Retourne la date la plus ancienne entre le fichier CSV (données réelles)
    et le tracker JSON (zones vides déjà vérifiées).
    """
    # 1. Date du fichier CSV (Réel)
    csv_date = datetime.now()
    file_path = get_file_path(folder, ticker)
    
    if os.path.exists(file_path):
        try:
            # Lecture légère juste pour récupérer la première date
            # On lit seulement quelques lignes pour aller vite
            df = pd.read_csv(file_path, nrows=5) 
            if not df.empty and 'date' in df.columns:
                csv_date = pd.to_datetime(df['date'].iloc[0])
            elif not df.empty:
                 # Cas où la date est dans l'index
                df = pd.read_csv(file_path, index_col=0, nrows=5)
                csv_date = pd.to_datetime(df.index[0])
        except Exception as e:
            print(f"⚠️ Erreur lecture CSV {ticker}: {e}")
    
    if csv_date.tzinfo is not None:
        csv_date = csv_date.tz_localize(None)

    # 2. Date du Tracker JSON (Mémoire des trous)
    state = load_state()
    json_date = csv_date # Par défaut, on prend celle du CSV
    
    if ticker in state:
        try:
            json_date = datetime.fromisoformat(state[ticker])
            if json_date.tzinfo is not None:
                json_date = json_date.tz_localize(None)
        except: pass

    # On prend la date la plus ancienne des deux pour continuer à remonter le temps
    return min(csv_date, json_date)

def merge_and_save(folder, ticker, new_df):
    """Fusionne les nouvelles données avec l'existant dans le bon dossier."""
    file_path = get_file_path(folder, ticker)
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, index_col=0)
        old_df.index = pd.to_datetime(old_df.index)
        full_df = pd.concat([new_df, old_df])
        # Suppression doublons
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        full_df.sort_index(inplace=True)
    else:
        full_df = new_df

    full_df.to_csv(file_path)
    return full_df.index.min(), len(full_df)

def download_previous_segment(ticker, current_start_date):
    """
    Télécharge les données Tiingo. 
    """
    global CURRENT_KEY_INDEX, CONSECUTIVE_429

    # On utilise 25 jours pour être sûr de ne pas manquer de jours
    end_date = current_start_date
    start_date = end_date - timedelta(days=25)
    
    fmt = '%Y-%m-%d'
    current_key = TIINGO_API_KEYS[CURRENT_KEY_INDEX]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {current_key}'
    }

    # Logique Crypto vs Action
    is_crypto = "btcusd" in ticker.lower() or "ethusd" in ticker.lower()

    if is_crypto:
        url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={ticker}&startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min"
    else:
        url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min&columns=date,open,high,low,close,volume"
    
    try:
        type_icon = "🪙" if is_crypto else "📈"
        print(f"⬇️  {type_icon} {ticker}:DL {start_date.strftime(fmt)} -> {end_date.strftime(fmt)} ... ", end="")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            CONSECUTIVE_429 = 0
            data = response.json()
            
            if not data:
                print("⚠️  Vide (Pas de data API).")
                return pd.DataFrame() 

            # Parsing
            if is_crypto:
                if isinstance(data, list) and len(data) > 0 and 'priceData' in data[0]:
                    raw_data = data[0]['priceData']
                else:
                    return pd.DataFrame()
            else:
                raw_data = data

            df = pd.DataFrame(raw_data)
            if df.empty:
                print("⚠️  DF vide.")
                return pd.DataFrame()

            # Standardisation
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in df.columns if c in cols_to_keep]]
            
            # FILTRE DE SÉCURITÉ : On vire les lignes à 0 ou NaN dès la réception
            df = df.dropna()
            if 'open' in df.columns:
                df = df[df['open'] > 0]
            
            print(f"✅ {len(df)} lignes.")
            return df

        elif response.status_code == 429:
            print(f"\n🛑 Limite atteinte (Clé {CURRENT_KEY_INDEX + 1}).")
            CONSECUTIVE_429 += 1
            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(TIINGO_API_KEYS)
            print(f"🔀 Changement vers clé {CURRENT_KEY_INDEX + 1}")

            if CONSECUTIVE_429 >= len(TIINGO_API_KEYS):
                print("❗ Toutes les clés sont fatiguées. Pause forcée de 60s.")
                time.sleep(60)
                CONSECUTIVE_429 = 0
            return None 

        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Exception interne: {e}")
        return pd.DataFrame()

# --- BOUCLE PRINCIPALE ---

if __name__ == "__main__":
    # 1. Création des dossiers
    for config in CONFIGS:
        if not os.path.exists(config['folder']):
            os.makedirs(config['folder'])
            print(f"📁 Dossier vérifié : {config['folder']}")

    print(f"🏗️  Lancement de l'Archéologue Unifié (Mode Safe - Sans Zéros).")
    print(f"🎯 Objectif : {GOAL_DATE.date()}")

    while True:
        # 2. ANALYSE DE L'ETAT
        status_list = []
        finished_count = 0
        total_tickers = 0

        # print("\n🔍 Analyse de l'avancement...")
        
        for config in CONFIGS:
            folder = config['folder']
            for ticker in config['tickers']:
                total_tickers += 1
                start_date = get_current_start_date(folder, ticker)
                
                if start_date <= GOAL_DATE:
                    finished_count += 1
                else:
                    status_list.append({
                        'ticker': ticker,
                        'date': start_date,
                        'folder': folder
                    })

        # 3. CONDITION DE SORTIE
        if finished_count == total_tickers:
            print("\n🎉 MISSION ACCOMPLIE ! Historique complet téléchargé.")
            break
            
        # 4. CHOIX DE LA CIBLE (Date la plus récente = Priorité)
        status_list.sort(key=lambda x: x['date'], reverse=True)
        
        target = status_list[0]
        target_ticker = target['ticker']
        target_date = target['date']
        target_folder = target['folder']
        
        print(f"🎯 Cible : {target_ticker} (Début actuel : {target_date.date()})")
        
        # 5. ACTION
        new_data = download_previous_segment(target_ticker, target_date)
        
        if new_data is None:
            time.sleep(1)
            continue
            
        if not new_data.empty:
            # Cas A : On a des données
            new_min, total_lines = merge_and_save(target_folder, target_ticker, new_data)
            print(f"   💾 Sauvegardé. Nouveau début CSV : {new_min}")
            
            # Mise à jour du tracker JSON avec la date réelle des données
            update_tracker(target_ticker, new_min)
            
        else:
            # Cas B : Trou (Pas de données API)
            # Au lieu de créer du faux data, on met juste à jour le carnet de bord
            
            dummy_date = target_date - timedelta(days=25)
            print(f"   ⚠️  Zone vide détectée. On note {dummy_date.date()} dans le tracker JSON.")
            
            # On NE touche PAS au fichier CSV, on met juste à jour la mémoire
            update_tracker(target_ticker, dummy_date)

        # 7. TEMPO API
        time.sleep(1.5)