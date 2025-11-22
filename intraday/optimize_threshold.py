import pandas as pd
import numpy as np
import joblib
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = 'data_enriched_v5_stationary'
FEATURES_PATH = 'selected_features_v5.pkl'
SCALER_PATH = 'scaler_v5.pkl'
MODEL_PATH = 'transformer_v5.keras'
FUTURE_SPLIT_DATE = '2025-06-01' 
SEQ_LEN = 60

# Liste des seuils à tester
THRESHOLDS = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

# --- MODEL ---
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.position_embeddings(positions)

def load_stuff():
    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = models.load_model(MODEL_PATH, custom_objects={'PositionalEmbedding': PositionalEmbedding})
    return model, scaler, features

def run_optimization():
    model, scaler, features = load_stuff()
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    
    print(f"🔍 TEST DE SENSIBILITÉ SUR {len(THRESHOLDS)} SEUILS...")
    
    # On stocke les résultats globaux pour chaque seuil
    # Structure: {0.60: {'wins': 0, 'total': 0}, 0.65: ...}
    stats = {th: {'wins': 0, 'total': 0, 'pnl_points': 0.0} for th in THRESHOLDS}
    
    # Pour aller vite, on charge tout, on prédit, et ENSUITE on filtre
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[df.index >= FUTURE_SPLIT_DATE]
            if len(df) < SEQ_LEN + 10: continue
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            X_raw = scaler.transform(df[features]).astype('float32')
            targets = df['Target'].values
            
            # Batch creation (step=5 pour aller très vite)
            X_batch = []
            indices_batch = []
            for i in range(0, len(X_raw) - SEQ_LEN, 5):
                X_batch.append(X_raw[i : i+SEQ_LEN])
                indices_batch.append(i + SEQ_LEN - 1)
                
            if not X_batch: continue
            
            X_batch = np.array(X_batch)
            preds = model.predict(X_batch, batch_size=4096, verbose=0)
            
            # Extraction des données brutes
            raw_probs = np.max(preds, axis=1) # La confiance (ex: 0.72)
            raw_actions = np.argmax(preds, axis=1) # L'action (0, 1, 2)
            
            # --- BOUCLE SUR LES SEUILS ---
            # On applique chaque filtre sur les mêmes prédictions (pas besoin de recalculer)
            for th in THRESHOLDS:
                # Masque : On ne garde que ceux qui dépassent le seuil TH
                mask = raw_probs >= th
                
                # Actions filtrées
                actions = raw_actions[mask]
                indices = np.array(indices_batch)[mask]
                
                for j, action in enumerate(actions):
                    if action == 1: continue # WAIT
                    
                    real = targets[indices[j]]
                    is_win = False
                    
                    if action == 2: # BUY
                        if real == 2: is_win = True
                    elif action == 0: # SELL
                        if real == 0: is_win = True
                        
                    stats[th]['total'] += 1
                    if is_win: stats[th]['wins'] += 1

        except: pass

    # --- AFFICHAGE DU TABLEAU ---
    print("\n📊 --- RÉSULTATS PAR SEUIL DE CONFIANCE ---")
    print(f"{'SEUIL':<10} | {'TRADES':<10} | {'WIN RATE':<10} | {'QUALITÉ'}")
    print("-" * 50)
    
    for th in THRESHOLDS:
        total = stats[th]['total']
        wins = stats[th]['wins']
        wr = (wins / total * 100) if total > 0 else 0
        
        quality = "🔴"
        if wr > 55: quality = "🟠"
        if wr > 60: quality = "🟢"
        if wr > 65: quality = "🔥"
        if wr > 70: quality = "💎"
        
        print(f"{th*100:.0f}%       | {total:<10} | {wr:.2f}%     | {quality}")

if __name__ == "__main__":
    run_optimization()