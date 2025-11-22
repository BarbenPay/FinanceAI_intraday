import os
# --- SILENCE TENSORFLOW ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Cache les logs techniques (INFO/WARNING)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, mixed_precision, layers
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
import joblib
import glob
from tqdm import tqdm
import gc

# --- CONFIG ---
DATA_DIR = 'data_enriched_v6_heavy'
FEATURES_PATH = 'selected_features_v6.pkl'
SCALER_PATH = 'scaler_v6.pkl'
MODEL_PATH = 'transformer_v6_balanced.keras'
SEQ_LEN = 180
BATCH_SIZE = 1024 # Confortable pour l'inférence
FUTURE_SPLIT_DATE = '2025-06-01'

# --- SETUP GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- RECONSTRUCTION COUCHE CUSTOM ---
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.position_embeddings(positions)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

def evaluate_safe():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        return

    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print("🧠 Chargement du modèle...")
    model = models.load_model(MODEL_PATH, custom_objects={'PositionalEmbedding': PositionalEmbedding})
    
    # ⚡ ASTUCE : On compile avec jit_compile=False pour éviter la recompilation incessante
    # sur des fichiers de tailles variables. C'est beaucoup plus fluide pour le test.
    model.compile(jit_compile=False) 

    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    
    all_preds_classes = []
    all_targets = []
    
    print(f"🔮 Audit sur {len(files)} fichiers...")
    
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[df.index >= FUTURE_SPLIT_DATE]
            
            if len(df) <= SEQ_LEN: continue
            
            # Nettoyage
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Scaling (Optimisation mémoire : on cast tout de suite)
            data_values = scaler.transform(df[features]).astype('float16')
            target_values = df['Target'].values.astype('int32')
            
            # Slicing Rapide
            indices = range(SEQ_LEN, len(data_values), 5)
            batch_x = []
            batch_y = []
            
            for i in indices:
                batch_x.append(data_values[i-SEQ_LEN : i])
                batch_y.append(target_values[i-1])
            
            if not batch_x: continue
            
            X_file = np.array(batch_x)
            
            # Prédiction (Rapide et Silencieuse)
            preds_probs = model.predict(X_file, batch_size=BATCH_SIZE, verbose=0)
            preds_classes = np.argmax(preds_probs, axis=1)
            
            all_preds_classes.extend(preds_classes)
            all_targets.extend(batch_y)
            
            # Ménage
            del X_file, batch_x, batch_y, df, data_values, preds_probs
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n🛑 Arrêt manuel demandé.")
            break
        except Exception as e:
            pass

    if not all_targets:
        print("❌ Aucune donnée analysée.")
        return

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds_classes)
    
    print("\n" + "="*60)
    print(f"🏆 RÉSULTATS AUDIT FINAL (V6.3) - {len(y_true)} Échantillons")
    print("="*60)
    
    print(classification_report(y_true, y_pred, target_names=['SELL (0)', 'WAIT (1)', 'BUY (2)']))
    
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"💎 MCC Score : {mcc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔍 Matrice de Confusion :")
    print(f"SELL corrects : {cm[0,0]} / {np.sum(cm[0])} ({cm[0,0]/np.sum(cm[0]):.1%})")
    print(f"BUY  corrects : {cm[2,2]} / {np.sum(cm[2])} ({cm[2,2]/np.sum(cm[2]):.1%})")

if __name__ == "__main__":
    evaluate_safe()