import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = "data_enriched_v7_sota" # Dossier V7
SAMPLE_FRACTION = 0.10 # 10% suffit largement avec Permutation Importance
TOP_N = 35
CORR_THRESH = 0.90 # Seuil de purge des corrélations

print("--- SÉLECTION DES FEATURES V7 (Mode SOTA) ---")

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
data_chunks = []

# On charge un échantillon représentatif
for f in tqdm(files[:25]): 
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) > 1000:
            # On prend des blocs aléatoires pour avoir tous les régimes
            data_chunks.append(df.sample(frac=SAMPLE_FRACTION, random_state=42))
    except: pass

full_df = pd.concat(data_chunks)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

X = full_df.drop(columns=['Target'])
y = full_df['Target'].astype(int)

print(f"📊 Dataset: {len(X)} lignes, {len(X.columns)} features brutes.")

# 1. PURGE DES CORRÉLATIONS
print("✂️ Suppression des doublons (Corrélation > 0.9)...")
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESH)]

X_clean = X.drop(columns=to_drop)
print(f"   👉 {len(to_drop)} features supprimées. Reste : {len(X_clean.columns)}")

# 2. ENTRAÎNEMENT DU JUGE
# On garde shuffle=True ici car on veut savoir quelles features 
# sont intrinsèquement bonnes, peu importe l'ordre temporel pour ce test.
X_train, X_val, y_train, y_val = train_test_split(X_clean, y, test_size=0.3, random_state=42)

print("🧠 Entraînement Random Forest (Soyez patient)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

# 3. PERMUTATION IMPORTANCE (Le vrai test)
print("🕵️‍♂️ Calcul de l'impact réel (Permutation)...")
result = permutation_importance(rf, X_val, y_val, n_repeats=3, random_state=42, n_jobs=-1)
perm_sorted_idx = result.importances_mean.argsort()[::-1]

# 4. SAUVEGARDE TOP N
final_features = []
print("\n🏆 --- TOP FEATURES V7 ---")
for i in range(min(TOP_N, len(X_clean.columns))):
    idx = perm_sorted_idx[i]
    feat = X_clean.columns[idx]
    score = result.importances_mean[idx]
    print(f"{i+1}. {feat:<20} (Score: {score:.6f})")
    final_features.append(feat)

joblib.dump(final_features, 'selected_features_v7.pkl')
print("\n✅ Liste sauvegardée dans 'selected_features_v7.pkl'")

# Graphique
plt.figure(figsize=(12, 10))
sns.barplot(x=result.importances_mean[perm_sorted_idx[:TOP_N]], 
            y=X_clean.columns[perm_sorted_idx[:TOP_N]], palette="viridis")
plt.title("Importance par Permutation (V7)")
plt.tight_layout()
plt.savefig("features_v7.png")