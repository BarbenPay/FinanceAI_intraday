import pandas as pd
import glob
import os
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = 'data_enriched_v6_heavy'
FUTURE_SPLIT_DATE = '2025-06-01' # Date de coupure Train/Test

def count_targets():
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    
    if not files:
        print(f"❌ Erreur : Aucun fichier trouvé dans '{DATA_DIR}'.")
        print("   As-tu bien lancé enrich_data.py (V6.1) ?")
        return

    # Compteurs
    train_counts = {0: 0, 1: 0, 2: 0}
    test_counts = {0: 0, 1: 0, 2: 0}
    total_rows = 0
    
    print(f"📊 Audit des Targets sur {len(files)} fichiers...")
    
    for f in tqdm(files):
        try:
            # On charge juste la colonne Target et l'index pour aller vite
            df = pd.read_csv(f, index_col=0, parse_dates=True, usecols=['date', 'Target'])
            
            # Nettoyage rapide (identique au training)
            df.dropna(inplace=True)
            
            # Split
            df_train = df[df.index < FUTURE_SPLIT_DATE]
            df_test = df[df.index >= FUTURE_SPLIT_DATE]
            
            # Comptage Train
            vc_train = df_train['Target'].value_counts()
            for cls in [0, 1, 2]:
                train_counts[cls] += vc_train.get(cls, 0)
                
            # Comptage Test
            vc_test = df_test['Target'].value_counts()
            for cls in [0, 1, 2]:
                test_counts[cls] += vc_test.get(cls, 0)
                
            total_rows += len(df)
            
        except Exception as e:
            print(f"⚠️ Erreur lecture {f}: {e}")

    # --- RAPPORT ---
    print("\n" + "="*60)
    print(f"🏁 RÉSULTATS DE L'AUDIT ({total_rows:,} lignes totales)")
    print("="*60)
    
    print("\n📁 DATASET D'ENTRAÎNEMENT (Ce que l'IA va voir)")
    print(f"   🔴 SELL (0) : {train_counts[0]:>10,}  ({train_counts[0]/sum(train_counts.values())*100:.1f}%)")
    print(f"   ⚪ WAIT (1) : {train_counts[1]:>10,}  ({train_counts[1]/sum(train_counts.values())*100:.1f}%)")
    print(f"   🟢 BUY  (2) : {train_counts[2]:>10,}  ({train_counts[2]/sum(train_counts.values())*100:.1f}%)")
    print(f"   👉 TOTAL    : {sum(train_counts.values()):>10,}")

    min_class = min(train_counts.values())
    print(f"\n💡 CONSEIL POUR 'TARGET_SAMPLES_PER_CLASS' :")
    print(f"   La classe limitante est : {min_class:,}")
    print(f"   Tu devrais fixer : TARGET_SAMPLES_PER_CLASS = {min_class}")

    print("\n" + "-"*60)
    print("📁 DATASET DE TEST (Pour validation)")
    print(f"   🔴 SELL (0) : {test_counts[0]:>10,}")
    print(f"   ⚪ WAIT (1) : {test_counts[1]:>10,}")
    print(f"   🟢 BUY  (2) : {test_counts[2]:>10,}")
    print("="*60)

if __name__ == "__main__":
    count_targets()