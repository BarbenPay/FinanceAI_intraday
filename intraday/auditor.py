import pandas as pd
import os

# --- CONFIGURATION ---
# On reprend exactement la même structure que dans ton script de téléchargement
CONFIGS = [
    {
        "folder": "data_1min",
        "type": "STOCKS",
        "tickers": [
            "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
            "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
            "MARA", "MSTR"
        ]
    },
    {
        "folder": "data_macro_1min",
        "type": "MACRO/CRYPTO",
        "tickers": [
            "SPY", "QQQ", "VXX", "IEF", "UUP", "btcusd"
        ]
    }
]

def analyze_data():
    total_files_found = 0
    total_tickers_configured = 0

    # On boucle sur les catégories (Dossiers)
    for config in CONFIGS:
        folder = config['folder']
        category = config['type']
        tickers = config['tickers']
        
        print(f"\n📊 ANALYSE DU DOSSIER : {folder} ({category})")
        # J'ai élargi un peu les colonnes pour la lisibilité
        print(f"{'TICKER':<10} | {'DEBUT':<25} | {'FIN':<25} | {'LIGNES':<10} | {'STATUS'}")
        print("-" * 95)

        # On boucle sur les tickers de ce dossier
        for ticker in tickers:
            total_tickers_configured += 1
            file_path = os.path.join(folder, f"{ticker}_1min.csv")
            
            if os.path.exists(file_path):
                try:
                    # On lit le CSV
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"{ticker:<10} | {'(Fichier Vide)':<20} | {'-':<20} | {0:<10} | ⚠️ VIDE")
                        continue

                    # Conversion date (essentiel pour avoir le vrai min/max)
                    # On gère le cas où la date est en colonne ou en index
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        start_date = df['date'].min()
                        end_date = df['date'].max()
                    else:
                        # Au cas où tu aurais sauvegardé avec l'index sans nom
                        df.index = pd.to_datetime(df.index)
                        start_date = df.index.min()
                        end_date = df.index.max()

                    count = len(df)
                    
                    # Critères de "Santé" du fichier
                    # > 100 000 minutes équivaut grossièrement à une bonne année de trading complète
                    if count > 150000:
                        status = "✅ EXCELLENT"
                    elif count > 50000:
                        status = "✅ OK"
                    else:
                        status = "⚠️ PEU DE DATA"
                    
                    print(f"{ticker:<10} | {str(start_date):<20} | {str(end_date):<20} | {count:<10} | {status}")
                    total_files_found += 1
                    
                except Exception as e:
                    print(f"{ticker:<10} | ❌ Erreur de lecture : {e}")
            else:
                print(f"{ticker:<10} | ❌ Fichier introuvable (Pas encore téléchargé ?)")
        
        print("-" * 95)

    print(f"\n🏁 Bilan Global : {total_files_found}/{total_tickers_configured} fichiers analysés.")

if __name__ == "__main__":
    # Vérification rapide que les dossiers existent
    missing_folders = [c['folder'] for c in CONFIGS if not os.path.exists(c['folder'])]
    
    if missing_folders:
        print(f"⚠️ Attention, dossiers introuvables : {missing_folders}")
        print("L'analyse risque d'échouer partiellement.")
    
    analyze_data()