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
DATA_DIR = "data_enriched_v4"
SAMPLE_FRACTION = 0.15 
TOP_N = 25

print("--- SÉLECTION DES FEATURES (MODE STRICT / STATIONNAIRE) - CORRIGÉ ---")

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
data_chunks = []

print("🎲 Chargement échantillon...")
for f in tqdm(files):
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) > 1000:
            chunk = df.sample(frac=SAMPLE_FRACTION, random_state=42)
            data_chunks.append(chunk)
    except: pass

if not data_chunks:
    print("❌ Erreur : Pas de données chargées.")
    exit()

full_df = pd.concat(data_chunks)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

print(f"📊 Analyse sur {len(full_df)} lignes.")

# --- FILTRAGE STRICT ---
all_cols = full_df.columns.tolist()

features = []
for col in all_cols:
    is_excluded = False
    
    # 1. Exclusion CRITIQUE : La Cible et les prix bruts
    if col in ['Target', 'open', 'high', 'low', 'close', 'volume']:
        is_excluded = True
        
    # 2. Exclusion Macro Brute (Non stationnaire)
    if 'macro_' in col and '_close' in col:
        is_excluded = True
    if 'macro_' in col and '_vol' in col:
        is_excluded = True
    if 'macro_' in col and '_level' in col: # On vire aussi les niveaux (ex: VIX à 20)
        is_excluded = True
        
    # 3. On garde le reste (Indicateurs relatifs, returns, hour_sin...)
    if not is_excluded:
        features.append(col)

print(f"🔎 Features candidates (Nettoyées): {len(features)}")
# Vérification paranoïaque
if 'Target' in features:
    print("⚠️ ALERTE : Target est encore là ! Suppression forcée.")
    features.remove('Target')

X = full_df[features]
y = full_df['Target'].astype(int)

# Entraînement
print("🧠 Entraînement du Juge (Random Forest)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X, y)

# Résultats
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n🏆 --- TOP FEATURES V4 (FINAL) ---")
selected_features = []
for i in range(min(TOP_N, len(features))):
    feat = features[indices[i]]
    score = importances[indices[i]]
    print(f"{i+1}. {feat:<30} ({score:.4f})")
    selected_features.append(feat)

joblib.dump(selected_features, 'selected_features_v4.pkl')
print(f"\n✅ Sauvegardé dans 'selected_features_v4.pkl'")

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices[:TOP_N]], y=[features[i] for i in indices[:TOP_N]])
plt.title("Importance des Features (Sans Biais)")
plt.tight_layout()
plt.savefig("features_v4_strict.png")