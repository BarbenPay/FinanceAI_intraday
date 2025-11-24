import pandas as pd
import numpy as np
import joblib
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tqdm import tqdm

# --- CONFIG V7 ---
DATA_DIR = 'data_enriched_v7_sota'
FEATURES_PATH = 'selected_features_v7.pkl'
SCALER_PATH = 'scaler_v7.pkl'
MODEL_PATH = 'transformer_v7_sota.keras'
FUTURE_SPLIT_DATE = '2025-06-01' 
SEQ_LEN = 180

# Seuils à tester
THRESHOLDS = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50]

# --- SETUP ---
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + tf.cast(embedded_positions, inputs.dtype)
    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "output_dim": self.output_dim})
        return config

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.from_logits = from_logits
    def call(self, y_true, y_pred):
        if self.from_logits: y_pred = tf.nn.softmax(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.int32)
        y_pred_true_class = tf.gather(y_pred, y_true, batch_dims=1, axis=1)
        loss = - ((1 - y_pred_true_class) ** self.gamma) * tf.math.log(y_pred_true_class)
        return tf.reduce_mean(loss)
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'from_logits': self.from_logits})
        return config

def load_stuff():
    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = models.load_model(MODEL_PATH, custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss
    })
    return model, scaler, features

def run_optimization():
    model, scaler, features = load_stuff()
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    
    print(f"🔍 CALCUL DU 'REAL EDGE' (RECALL & PRECISION)...")
    
    all_probs = []
    all_targets = []
    
    print("   ⚡ Inference globale...")
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[df.index >= FUTURE_SPLIT_DATE]
            if len(df) < SEQ_LEN + 10: continue
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            X_raw = scaler.transform(df[features]).astype('float16')
            targets = df['Target'].values.astype('int8')
            
            X_batch = []
            target_batch = []
            for i in range(SEQ_LEN, len(X_raw), 5):
                X_batch.append(X_raw[i-SEQ_LEN : i])
                target_batch.append(targets[i-1])
                
            if not X_batch: continue
            
            preds = model.predict(np.array(X_batch), batch_size=4096, verbose=0)
            all_probs.extend(preds)
            all_targets.extend(target_batch)
        except: pass

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # CALCUL DES TOTAUX RÉELS
    total_real_buys = np.sum(all_targets == 2)
    total_real_sells = np.sum(all_targets == 0)
    
    print(f"\n📊 TOTAL DANS LE DATASET : {total_real_buys} Vrais BUY | {total_real_sells} Vrais SELL")
    print(f"🎲 HASARD (Baseline) : 33.33%")

    print("\n" + "="*110)
    print(f"{'SEUIL':<6} | {'NB TRADES':<10} | {'PRECISION BUY':<15} | {'RECALL BUY':<12} | {'EDGE vs RANDOM':<15}")
    print("="*110)
    
    for th in THRESHOLDS:
        p_sell = all_probs[:, 0]
        p_buy = all_probs[:, 2]
        
        # Logique de décision
        decisions = np.ones(len(all_targets), dtype=int)
        
        mask_buy = (p_buy > th) & (p_buy > p_sell)
        mask_sell = (p_sell > th) & (p_sell >= p_buy) # Priorité sell si égalité (arbitraire)
        
        decisions[mask_buy] = 2
        decisions[mask_sell] = 0
        
        # --- ANALYSE BUY ---
        pred_buys = (decisions == 2)
        correct_buys = (decisions == 2) & (all_targets == 2)
        
        nb_trades_buy = np.sum(pred_buys)
        
        if nb_trades_buy > 0:
            precision_buy = np.sum(correct_buys) / nb_trades_buy * 100
            recall_buy = np.sum(correct_buys) / total_real_buys * 100
            
            # EDGE : Combien de % au dessus du hasard (33.33%) ?
            edge = precision_buy - 33.33
            edge_str = f"+{edge:.2f}% 🔥" if edge > 10 else f"+{edge:.2f}%"
            
            print(f"{th:.2f}   | {nb_trades_buy:<10} | {precision_buy:6.2f}%         | {recall_buy:6.2f}%       | {edge_str}")
        else:
            print(f"{th:.2f}   | 0          | -               | -            | -")

if __name__ == "__main__":
    run_optimization()