import pandas as pd
import os

# --- CONFIGURATION ---
DATA_FOLDER = "data_1min"
TICKERS = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
    "MARA", "MSTR"
]

def analyze_data():
    print(f"{'TICKER':<10} | {'DEBUT':<20} | {'FIN':<20} | {'LIGNES':<10} | {'STATUS'}")
    print("-" * 80)
    
    total_files = 0
    
    for ticker in TICKERS:
        file_path = os.path.join(DATA_FOLDER, f"{ticker}_1min.csv")
        
        if os.path.exists(file_path):
            try:
                # On lit le CSV
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                
                start_date = df['date'].min()
                end_date = df['date'].max()
                count = len(df)
                
                # Vérification rapide (Est-ce qu'on a bien ~3 ans ?)
                # On considère que c'est "OK" si on a plus de 200,000 minutes (ordre de grandeur pour 2-3 ans)
                status = "✅ OK" if count > 100000 else "⚠️ PEU DE DATA"
                
                print(f"{ticker:<10} | {str(start_date):<20} | {str(end_date):<20} | {count:<10} | {status}")
                total_files += 1
                
            except Exception as e:
                print(f"{ticker:<10} | Erreur de lecture du fichier : {e}")
        else:
            print(f"{ticker:<10} | ❌ Fichier introuvable")

    print("-" * 80)
    print(f"Analyse terminée. {total_files}/{len(TICKERS)} fichiers trouvés.")

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Le dossier '{DATA_FOLDER}' n'existe pas. Lance le script de téléchargement d'abord.")
    else:
        analyze_data()