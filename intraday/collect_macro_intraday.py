import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TIINGO_API_KEYS = [
    "",
    "",
    ""
]

CURRENT_KEY_INDEX = 0
CONSECUTIVE_429 = 0

DATA_FOLDER = "data_macro_1min"

# LISTE DES TICKERS MACRO (ETFs Proxies + Crypto)
# SPY = S&P 500
# QQQ = NASDAQ
# VXX = VIX (Volatilité)
# IEF = US 10Y Treasury (Taux)
# UUP = DXY (Dollar Index)
# btcusd = Bitcoin
TICKERS = [
    "SPY", "QQQ", "VXX", "IEF", "UUP", "btcusd"
]

# Jusqu'où remonter ? (ex: 2021-01-01)
GOAL_DATE = datetime(2021, 1, 1)

# --- FONCTIONS ---

def get_current_start_date(ticker):
    """Lit le CSV et retourne la date la plus ancienne enregistrée."""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_1min.csv")
    
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
        print(f"⚠️ Erreur lecture {ticker}: {e}")
        return datetime.now()

def merge_and_save(ticker, new_df):
    """Fusionne et sauvegarde."""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_1min.csv")
    
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

def download_previous_month(ticker, current_start_date):
    """Télécharge le mois précédent (Gère IEX pour les stocks, Crypto pour BTC)."""
    global CURRENT_KEY_INDEX, CONSECUTIVE_429

    end_date = current_start_date
    start_date = end_date - timedelta(days=25) # On prend 25 jours pour être sûr (les mois varient)
    
    fmt = '%Y-%m-%d'
    current_key = TIINGO_API_KEYS[CURRENT_KEY_INDEX]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {current_key}'
    }

    # --- DÉTECTION DU TYPE D'ACTIF ---
    if "btcusd" in ticker.lower():
        # API CRYPTO
        url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={ticker}&startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min"
        is_crypto = True
    else:
        # API IEX (ACTIONS/ETFS)
        url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min&columns=date,open,high,low,close,volume"
        is_crypto = False
    
    try:
        print(f"⬇️  {ticker}: Téléchargement {start_date.strftime(fmt)} -> {end_date.strftime(fmt)} ... ", end="")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            CONSECUTIVE_429 = 0
            data = response.json()
            
            if not data:
                print("⚠️  Vide.")
                return pd.DataFrame()

            # --- PARSING SPÉCIFIQUE CRYPTO VS STOCK ---
            if is_crypto:
                # L'API Crypto renvoie une liste contenant un objet par ticker avec 'priceData'
                # Ex: [{'ticker': 'btcusd', 'priceData': [...]}]
                if isinstance(data, list) and len(data) > 0 and 'priceData' in data[0]:
                    raw_data = data[0]['priceData']
                else:
                    print("⚠️  Format Crypto inattendu.")
                    return pd.DataFrame()
            else:
                # L'API IEX renvoie directement la liste des bougies
                raw_data = data

            df = pd.DataFrame(raw_data)
            if df.empty:
                print("⚠️  Vide après parsing.")
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Nettoyage colonnes inutiles (tradesDone, etc pour crypto)
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in df.columns if c in cols_to_keep]]
            
            print(f"✅ {len(df)} lignes.")
            return df

        elif response.status_code == 429:
            print(f"\n🛑 Limite atteinte (Clé {CURRENT_KEY_INDEX}).")
            CONSECUTIVE_429 += 1
            if CONSECUTIVE_429 >= len(TIINGO_API_KEYS):
                print("❗ Pause forcée 60s.")
                time.sleep(60)
                CONSECUTIVE_429 = 0
            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(TIINGO_API_KEYS)
            return None # Retry

        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return pd.DataFrame()

# --- MAIN LOOP ---

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print(f"🏗️  Récupération MACRO Intraday jusqu'à {GOAL_DATE.date()}")

    while True:
        status_list = []
        finished_count = 0
        
        for ticker in TICKERS:
            start_date = get_current_start_date(ticker)
            if start_date <= GOAL_DATE:
                finished_count += 1
            else:
                status_list.append((ticker, start_date))
        
        if finished_count == len(TICKERS):
            print("\n🎉 TOUTES LES DONNÉES MACRO SONT TÉLÉCHARGÉES !")
            break
            
        # On prend celui qui a le moins d'historique
        status_list.sort(key=lambda x: x[1], reverse=True)
        target_ticker, target_date = status_list[0]
        
        new_data = download_previous_month(target_ticker, target_date)
        
        if new_data is None: continue # Rate limit retry
            
        if not new_data.empty:
            new_min, total = merge_and_save(target_ticker, new_data)
            print(f"   💾 Stocké. Remonte au : {new_min}")
        else:
            # Si vide, on insère un dummy pour forcer le recul et éviter la boucle infinie
            dummy_date = target_date - timedelta(days=25)
            print(f"   ⚠️ Période vide. On marque le terrain à {dummy_date.date()} pour avancer.")
            dummy_df = pd.DataFrame({'open': [0]}, index=[dummy_date])
            dummy_df.index.name = 'date'
            merge_and_save(target_ticker, dummy_df)

        time.sleep(1)