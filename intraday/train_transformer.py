import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, utils
from tensorflow.keras import mixed_precision # Important pour RTX
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# --- ACTIVATION DU TURBO RTX (Mixed Precision) ---
# Permet de calculer 2x plus vite en utilisant moins de VRAM
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- CONFIGURATION V6.3 (BALANCED FINAL) ---
DATA_DIR = 'data_enriched_v6_heavy'
FEATURES_PATH = 'selected_features_v6.pkl'
SCALER_PATH = 'scaler_v6.pkl'
MODEL_PATH = 'transformer_v6_balanced.keras'

# Paramètres Temporels & Modèle
SEQ_LEN = 60
EMBED_DIM = 64         # Suffisant pour capter les patterns
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 3         # 3 Couches = Le bon équilibre complexité/vitesse
DROPOUT = 0.20

# Entraînement
BATCH_SIZE = 2048      # Gros batch pour stabilité et vitesse
EPOCHS = 50            
LEARNING_RATE = 0.001  # Vitesse standard pour AdamW
FUTURE_SPLIT_DATE = '2025-06-01'

# CIBLE D'ÉQUILIBRAGE (Basé sur ton audit : le WAIT est le facteur limitant)
# On prend 2 200 000 par classe pour avoir un dataset 33% / 33% / 33%
TARGET_SAMPLES_PER_CLASS = 2200000

# Setup GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# --- COUCHES DU MODÈLE ---
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

class BalancedGenerator(utils.Sequence):
    """Générateur qui prend les données déjà chargées en RAM"""
    def __init__(self, X_data, y_data, indices, batch_size, seq_len):
        self.X_data = X_data # Liste des arrays (un par fichier)
        self.y_data = y_data # Liste des targets
        self.indices = indices # Liste plate des tuples (file_idx, row_idx)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        
        # On prépare les conteneurs du batch
        X_b = np.empty((self.batch_size, self.seq_len, self.X_data[0].shape[1]), dtype='float16')
        y_b = np.empty((self.batch_size), dtype='int32')
        
        for i, (f_idx, r_idx) in enumerate(batch_idx):
            # Extraction de la fenêtre glissante [t-60 : t]
            X_b[i] = self.X_data[f_idx][r_idx-self.seq_len : r_idx]
            y_b[i] = self.y_data[f_idx][r_idx-1]
            
        return X_b, y_b

# --- CHARGEMENT ET PRÉPARATION ---
def load_and_balance_data(features):
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    if not files:
        print(f"❌ ERREUR : Aucun fichier dans {DATA_DIR}")
        exit()

    print(f"⏳ Chargement et Équilibrage ({TARGET_SAMPLES_PER_CLASS} par classe)...")
    
    X_temp, y_temp = [], []
    dfs_sample = []
    
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            # Nettoyage
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if not all(c in df.columns for c in features + ['Target']): continue
            
            # On garde Train uniquement pour l'apprentissage
            df_train = df[df.index < FUTURE_SPLIT_DATE]
            
            if len(df_train) > SEQ_LEN:
                # On garde un bout pour le scaler
                dfs_sample.append(df_train.sample(frac=0.05))
                
                # On stocke les données brutes
                data = df_train[features].values.astype('float32')
                targets = df_train['Target'].values.astype('int32')
                
                X_temp.append(data)
                y_temp.append(targets)
        except: pass

    # --- SCALER ---
    print("⚖️  Entraînement du Scaler...")
    full_sample = pd.concat(dfs_sample)
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=100000)
    scaler.fit(full_sample[features])
    joblib.dump(scaler, SCALER_PATH)
    del dfs_sample, full_sample
    gc.collect()
    
    # --- CONVERSION FLOAT16 (RAM) ---
    print("🔄 Conversion en Float16...")
    X_scaled = [scaler.transform(x).astype('float16') for x in X_temp]
    
    # --- ÉQUILIBRAGE ---
    print("🎯 Sélection des indices équilibrés...")
    candidates = {0: [], 1: [], 2: []}
    
    for f_idx, targets in enumerate(y_temp):
        # Indices valides (on doit avoir SEQ_LEN d'historique avant)
        valid_indices = np.arange(SEQ_LEN, len(targets))
        # On regarde la target correspondante
        valid_targets = targets[valid_indices-1]
        
        for i, tgt in zip(valid_indices, valid_targets):
            if tgt in [0, 1, 2]:
                candidates[tgt].append((f_idx, i))
                
    final_indices = []
    for cls in [0, 1, 2]:
        avail = len(candidates[cls])
        count = min(avail, TARGET_SAMPLES_PER_CLASS)
        
        # Tirage aléatoire sans remise
        if count > 0:
            indices = np.random.choice(len(candidates[cls]), count, replace=False)
            selected = [candidates[cls][i] for i in indices]
            final_indices.extend(selected)
            print(f"   ✅ Classe {cls} : {count} exemples.")
        else:
            print(f"   ⚠️ Classe {cls} vide !")
            
    np.random.shuffle(final_indices)
    return X_scaled, y_temp, final_indices

def load_test_data(features, scaler):
    files = glob.glob(os.path.join(DATA_DIR, "*_stationary.csv"))
    X_list, y_list, indices = [], [], []
    
    print("⏳ Préparation du Test Set (Réaliste)...")
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[df.index >= FUTURE_SPLIT_DATE] # Test only
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) > SEQ_LEN:
                data = scaler.transform(df[features]).astype('float16')
                tgt = df['Target'].values.astype('int32')
                
                X_list.append(data)
                y_list.append(tgt)
                
                # On prend 1 point sur 5 pour aller vite en validation
                f_indices = [(len(X_list)-1, i) for i in range(SEQ_LEN, len(data), 5)]
                indices.extend(f_indices)
        except: pass
    return X_list, y_list, indices

# --- BUILD MODEL ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model_balanced(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(EMBED_DIM)(inputs)
    x = PositionalEmbedding(sequence_length=SEQ_LEN, output_dim=EMBED_DIM)(x)
    
    for _ in range(NUM_LAYERS):
        x = transformer_encoder(x, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # SORTIE FLOAT32 OBLIGATOIRE (Mixed Precision)
    outputs = layers.Dense(n_classes, activation="softmax", dtype='float32')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists(FEATURES_PATH):
        print("❌ ERREUR: Lance select_features.py d'abord.")
        exit()
        
    features = joblib.load(FEATURES_PATH)
    print(f"✅ Features ({len(features)}) chargées.")
    
    # 1. Chargement Train
    X_train, y_train, train_idx = load_and_balance_data(features)
    
    # 2. Chargement Test
    scaler = joblib.load(SCALER_PATH)
    X_test, y_test, test_idx = load_test_data(features, scaler)
    
    print(f"📊 Train: {len(train_idx)} seq | Test: {len(test_idx)} seq")
    
    # Générateurs
    train_gen = BalancedGenerator(X_train, y_train, train_idx, BATCH_SIZE, SEQ_LEN)
    test_gen = BalancedGenerator(X_test, y_test, test_idx, BATCH_SIZE, SEQ_LEN)

    # Modèle
    model = build_model_balanced((SEQ_LEN, len(features)), 3)
    
    cbs = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss', verbose=1),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=0)
    ]

    print("\n🚀 GO TRAIN V6.3 (Balanced)...")
    model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=cbs)
    
    # Eval
    print("\n🔍 Audit Final...")
    model.load_weights(MODEL_PATH)
    
    all_preds, all_true = [], []
    for i in tqdm(range(len(test_gen))):
        X_b, y_b = test_gen[i]
        p = model.predict_on_batch(X_b)
        all_preds.append(np.argmax(p, axis=1))
        all_true.append(y_b)
        
    y_p = np.concatenate(all_preds)
    y_t = np.concatenate(all_true)
    
    print(classification_report(y_t, y_p, target_names=['SELL', 'WAIT', 'BUY']))
    print(f"🏆 MCC : {matthews_corrcoef(y_t, y_p):.4f}")