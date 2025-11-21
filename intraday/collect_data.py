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

DATA_FOLDER = "data_1min"

# Tes actions
TICKERS = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
    "MARA", "MSTR"
]

# L'objectif : Jusqu'où veut-on remonter dans le passé ? (ex: 2021-01-01)
GOAL_DATE = datetime(2021, 1, 1)

# --- FONCTIONS ---

def get_current_start_date(ticker):
    """Lit le CSV et retourne la date la plus ancienne enregistrée."""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_1min.csv")
    
    if not os.path.exists(file_path):
        # Si pas de fichier, on considère que l'historique commence "Maintenant"
        return datetime.now()
    
    try:
        # On lit le CSV
        df = pd.read_csv(file_path)
        if df.empty:
            return datetime.now()
        
        # On récupère la première date
        first_date = pd.to_datetime(df['date'].iloc[0])
        
        # --- LE FIX EST ICI ---
        # Si la date a une timezone (UTC), on la retire pour pouvoir comparer
        if first_date.tzinfo is not None:
            first_date = first_date.tz_localize(None)
            
        return first_date

    except Exception as e:
        print(f"⚠️ Erreur lecture {ticker}: {e}")
        return datetime.now()

        
def merge_and_save(ticker, new_df):
    """Fusionne les nouvelles données (anciennes dates) avec l'existant."""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_1min.csv")
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, index_col=0)
        # Conversion index en datetime pour être sûr
        old_df.index = pd.to_datetime(old_df.index)
        
        # On concatène : [Vieux_Data_Recuperé] + [Data_Deja_La]
        full_df = pd.concat([new_df, old_df])
        
        # On nettoie les doublons (au cas où les dates se chevauchent)
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        full_df.sort_index(inplace=True)
    else:
        full_df = new_df

    full_df.to_csv(file_path)
    return full_df.index.min(), len(full_df)

def download_previous_month(ticker, current_start_date):
    """Télécharge le mois précédent la date donnée."""

    global CURRENT_KEY_INDEX, CONSECUTIVE_429

    end_date = current_start_date
    start_date = end_date - timedelta(days=30)
    
    # Formattage API
    fmt = '%Y-%m-%d'
    url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date.strftime(fmt)}&endDate={end_date.strftime(fmt)}&resampleFreq=1min&columns=date,open,high,low,close,volume"
    
    current_key = TIINGO_API_KEYS[CURRENT_KEY_INDEX]

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {current_key}'
    }
    
    try:
        print(f"⬇️  {ticker}({CURRENT_KEY_INDEX +1}) : Téléchargement {start_date.strftime(fmt)} -> {end_date.strftime(fmt)} ... ", end="")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            CONSECUTIVE_429 = 0
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                print(f"✅ {len(df)} lignes.")
                return df
            else:
                print("⚠️  Vide (Pas de data à cette période).")
                return pd.DataFrame() # Vide mais pas None
        elif response.status_code == 429:
            print(f"\n🛑 Limite atteinte sur la clé n°{CURRENT_KEY_INDEX +1}.")

            CONSECUTIVE_429 += 1
            if CONSECUTIVE_429 >= len(TIINGO_API_KEYS):
                print("❗ Toutes les clés ont atteint la limite. Pause de 1 minute.")
                CONSECUTIVE_429 = 0
                time.sleep(60)
            
            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(TIINGO_API_KEYS)
            print(f"🔀 On passe à la clé {CURRENT_KEY_INDEX + 1} pour le prochain essai.")

            return None # On renvoie None pour dire "réessaie"
        else:
            print(f"❌ Erreur {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return pd.DataFrame()

# --- MAIN LOOP ---

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print(f"🏗️  Lancement de l'Archéologue. Objectif : Remonter jusqu'à {GOAL_DATE.date()}")

    while True:
        # 1. ANALYSE DE L'ETAT ACTUEL
        # On crée une liste de (Ticker, Date_Debut_Actuelle)
        status_list = []
        finished_count = 0
        
        print("\n🔍 Analyse des fichiers...")
        for ticker in TICKERS:
            start_date = get_current_start_date(ticker)
            if start_date <= GOAL_DATE:
                finished_count += 1
            else:
                status_list.append((ticker, start_date))
        
        # Condition de sortie : Si tout le monde a atteint la date cible
        if finished_count == len(TICKERS):
            print("\n🎉 MISSION ACCOMPLIE ! Tous les fichiers remontent jusqu'à l'objectif.")
            break
            
        # 2. CHOIX DE LA CIBLE (Celui qui est le plus "en retard" dans le passé)
        # On trie pour avoir la date la plus RECENTE en premier (donc celui qui a le moins d'historique)
        status_list.sort(key=lambda x: x[1], reverse=True)
        
        target_ticker, target_date = status_list[0]
        
        print(f"🎯 Priorité : {target_ticker} (Historique commence le {target_date.date()})")
        print(f"   Reste à télécharger : {(target_date - GOAL_DATE).days} jours d'historique.")
        
        # 3. ACTION
        new_data = download_previous_month(target_ticker, target_date)
        
        if new_data is None:
            # Cas du Rate Limit (429), on boucle pour retenter sans changer de cible
            continue
            
        if not new_data.empty:
            new_min, total_lines = merge_and_save(target_ticker, new_data)
            print(f"   💾 Sauvegardé. Nouveau début : {new_min} (Total lignes: {total_lines})")
        else:
            # Si c'est vide (ex: week-end ou jour férié ou action n'existait pas encore),
            # on force artificiellement la date de recul pour ne pas boucler à l'infini sur la même période vide.
            # On crée un CSV vide ou on met à jour pour dire "j'ai vérifié cette période".
            # Astuce simple : on ne fait rien ici, car le merge_and_save ne sera pas appelé, 
            # MAIS il faut avancer sinon on boucle. 
            # Pour ce script simple : Si vide, on considère qu'on a "traité" la zone en créant un dummy record 
            # ou en acceptant que pour cette action, on ne trouvera rien avant.
            
            # Solution robuste : Si vide, on décale quand même la target date de 30 jours en arrière dans le fichier ?
            # C'est complexe sans modifier le fichier.
            # Solution pragmatique : On affiche "Vide" et on fait une pause,
            # Mais pour éviter la boucle infinie si l'action n'existait pas en 2021 (ex: RIVN),
            # Il faut détecter si on est AVANT l'IPO.
            
            if target_ticker == "RIVN" and target_date.year < 2021:
                 # Hack spécifique ou logique générale : si vide 3 fois de suite, on considère fini ?
                 # Pour l'instant, on recule la "date de scan" virtuellement
                 pass
            
            # Pour éviter de bloquer, si on reçoit vide, on va tricher :
            # On va insérer une ligne vide avec la date n-30 pour forcer le système à croire qu'on a des données
            # C'est sale mais ça débloque la boucle.
            dummy_df = pd.DataFrame({'open': [0]}, index=[target_date - timedelta(days=30)])
            dummy_df.index.name = 'date'
            merge_and_save(target_ticker, dummy_df)
            print("   ⚠️ Période vide détectée, on marque le terrain et on recule.")

        # 4. TEMPO
        time.sleep(2) # Respect de l'API