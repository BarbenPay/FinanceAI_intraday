import pandas as pd
import numpy as np
import joblib
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = 'data_enriched_v5_stationary'
FEATURES_PATH = 'selected_features_v5.pkl'
SCALER_PATH = 'scaler_v5.pkl'
MODEL_PATH = 'transformer_v5.keras'

# Paramètres financiers
INITIAL_CAPITAL = 10000      # Capital de départ
POSITION_SIZE = 2000         # Mise par trade
FEE_PER_TRADE = 0.001        # 0.1% (Frais Exchange)
SLIPPAGE = 0.0005            # 0.05% (Perte due à l'exécution)

# Paramètres Modèle
SEQ_LEN = 60
CONFIDENCE_THRESHOLD = 0.60  # Sniper Mode : On ne trade que si proba > 60%
FUTURE_SPLIT_DATE = '2025-06-01' 

# --- RECONSTRUCTION DU MODÈLE ---
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

def load_model_and_scaler():
    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Charge le modèle avec la couche personnalisée
    model = models.load_model(MODEL_PATH, custom_objects={'PositionalEmbedding': PositionalEmbedding})
    return model, scaler, features

def run_backtest():
    model, scaler, features = load_model_and_scaler()
    print(f"🤖 Modèle chargé. Features: {len(features)}")
    
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    
    total_trades = 0
    winning_trades = 0
    capital = INITIAL_CAPITAL
    
    # Courbe d'équité (Portefeuille)
    equity_curve = [capital]
    
    print(f"💼 Capital Initial : {capital}$ | Mise : {POSITION_SIZE}$")
    print(f"🎯 Sniper Mode : Confiance > {CONFIDENCE_THRESHOLD*100}%")
    print("⏳ Lancement du Backtest...")
    
    cumulative_pnl_dollars = 0 
    
    for f in tqdm(files):
        try:
            # 1. Chargement
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[df.index >= FUTURE_SPLIT_DATE] # Test set uniquement
            
            if len(df) < SEQ_LEN + 10: continue
            
            # Nettoyage
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Scaling
            X_raw = scaler.transform(df[features]).astype('float32')
            targets = df['Target'].values
            
            # Préparation des batchs (Step=1 pour précision max, ou Step=5 pour vitesse)
            X_batch = []
            indices_batch = []
            step = 1 
            
            for i in range(0, len(X_raw) - SEQ_LEN, step):
                seq = X_raw[i : i+SEQ_LEN]
                X_batch.append(seq)
                indices_batch.append(i + SEQ_LEN - 1)
                
            if not X_batch: continue
            
            X_batch = np.array(X_batch)
            
            # 2. Prédictions
            preds = model.predict(X_batch, batch_size=2048, verbose=0)
            
            # 3. Logique Sniper
            max_probs = np.max(preds, axis=1)      # La confiance (ex: 0.75)
            actions = np.argmax(preds, axis=1)     # L'action (0, 1, 2)
            
            # Si la confiance est trop faible, on force WAIT (1)
            actions = np.where(max_probs < CONFIDENCE_THRESHOLD, 1, actions)
            
            # 4. Calcul du PnL
            for j, action in enumerate(actions):
                if action == 1: continue # WAIT
                
                real_outcome = targets[indices_batch[j]]
                pnl_pct = 0
                
                # IA ACHÈTE (2)
                if action == 2:
                    if real_outcome == 2: pnl_pct = 0.015     # Gain (TP)
                    elif real_outcome == 0: pnl_pct = -0.010  # Perte (SL)
                    else: pnl_pct = 0.0                       # Timeout (Flat)
                
                # IA VEND (0)
                elif action == 0:
                    if real_outcome == 0: pnl_pct = 0.015     # Gain (TP Short)
                    elif real_outcome == 2: pnl_pct = -0.010  # Perte (SL Short)
                    else: pnl_pct = 0.0                       # Timeout
                
                # Application des frictions (Frais + Slippage)
                # On paie les frais et le slippage QUEL QUE SOIT le résultat
                pnl_pct = pnl_pct - FEE_PER_TRADE - SLIPPAGE
                
                # Impact sur le portefeuille
                dollar_result = POSITION_SIZE * pnl_pct
                cumulative_pnl_dollars += dollar_result
                
                total_trades += 1
                if pnl_pct > 0: winning_trades += 1
                
            # Mise à jour courbe après chaque fichier
            equity_curve.append(capital + cumulative_pnl_dollars)

        except Exception as e:
            print(f"Erreur {f}: {e}")
            
    # --- RÉSULTATS ---
    final_capital = capital + cumulative_pnl_dollars
    print("\n📊 --- RÉSULTATS DU BACKTEST (FINAL) ---")
    print(f"Trades Total : {total_trades}")
    
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        print(f"Win Rate : {win_rate:.2f}%")
    
    print(f"💰 Capital Final : {final_capital:.2f}$")
    perf = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    print(f"📈 Performance : {perf:+.2f}%")
    
    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Performance Finale (Sniper > {CONFIDENCE_THRESHOLD*100}%) - WinRate {win_rate:.1f}%")
    plt.xlabel("Temps (Actifs cumulés)")
    plt.ylabel("Capital ($)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("backtest_final_sniper.png")
    print("✅ Graphique sauvegardé : backtest_final_sniper.png")

if __name__ == "__main__":
    run_backtest()