import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = "data_enriched_v6_heavy" # Le nouveau dossier propre
SAMPLE_FRACTION = 0.50 # On prend 50% des données pour aller vite (augmenter si besoin)
TOP_N = 35 # On garde les 35 meilleures features

print("--- SÉLECTION DES FEATURES (V6 HEAVY) ---")

if not os.path.exists(DATA_DIR):
    print(f"❌ Erreur : Le dossier '{DATA_DIR}' n'existe pas. Lance enrich_data.py d'abord.")
    exit()

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
data_chunks = []

print(f"🎲 Chargement échantillon aléatoire ({SAMPLE_FRACTION*100}%) sur {len(files)} fichiers...")

for f in tqdm(files):
    try:
        # Lecture optimisée
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # On ignore les petits fichiers
        if len(df) > 500:
            # On prend un échantillon aléatoire pour ne pas surcharger la RAM
            chunk = df.sample(frac=SAMPLE_FRACTION, random_state=42)
            data_chunks.append(chunk)
    except Exception as e:
        print(f"⚠️ Erreur lecture {f}: {e}")

if not data_chunks:
    print("❌ Erreur : Pas de données chargées.")
    exit()

# Fusion
full_df = pd.concat(data_chunks)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

print(f"📊 Analyse sur {len(full_df)} lignes cumulées.")

# --- PRÉPARATION ---
# On récupère toutes les colonnes sauf la Target
all_cols = full_df.columns.tolist()
features = [c for c in all_cols if c != 'Target']

print(f"🔎 Features candidates : {len(features)}")

X = full_df[features]
y = full_df['Target'].astype(int)

# --- ENTRAÎNEMENT DU JUGE ---
print("🧠 Entraînement du Random Forest (Cela peut prendre 1-2 min)...")
# n_jobs=-1 utilise tous les coeurs du CPU
rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=12, 
    n_jobs=-1, 
    random_state=42, 
    class_weight='balanced'
)
rf.fit(X, y)

# --- RÉSULTATS ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n🏆 --- TOP FEATURES V5 (STATIONNAIRE) ---")
selected_features = []

for i in range(min(TOP_N, len(features))):
    feat = features[indices[i]]
    score = importances[indices[i]]
    print(f"{i+1}. {feat:<25} ({score:.4f})")
    selected_features.append(feat)

# Sauvegarde
joblib.dump(selected_features, 'selected_features_v6.pkl')
print(f"\n✅ Liste sauvegardée dans 'selected_features_v6.pkl'")

# Graphique
plt.figure(figsize=(12, 10))
sns.barplot(x=importances[indices[:TOP_N]], y=[features[i] for i in indices[:TOP_N]], palette="viridis")
plt.title("Importance des Features (Données Stationnaires V6)")
plt.xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig("features_v6.png")
print("✅ Graphique sauvegardé : features_v6.png")