import pandas as pd
import glob
import os
from tqdm import tqdm

# Tes dossiers de données brutes
FOLDERS = ['data_1min', 'data_macro_1min']

def clean_zeros():
    print("🧹 Démarrage du nettoyage des zéros...")
    for folder in FOLDERS:
        files = glob.glob(os.path.join(folder, "*.csv"))
        for f in tqdm(files, desc=f"Nettoyage {folder}"):
            try:
                # On charge tout
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                original_len = len(df)
                
                # FILTRE MAGIQUE : On ne garde que les prix > 0
                # On suppose que 'open' ne doit jamais être nul ou vide
                if 'open' in df.columns:
                    df = df[df['open'] > 0.000001] # Marge de sécurité float
                
                # Si on a supprimé des lignes, on sauvegarde
                if len(df) < original_len:
                    df.to_csv(f)
                    print(f"   ✨ {os.path.basename(f)} : {original_len - len(df)} lignes supprimées.")
            except Exception as e:
                print(f"❌ Erreur sur {f}: {e}")

if __name__ == "__main__":
    clean_zeros()