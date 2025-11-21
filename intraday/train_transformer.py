import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, classification_report
import joblib
from tqdm import tqdm

# --- CONFIG V4 (BIG DATA) ---
DATA_DIR = 'data_enriched_v4'
FEATURES_PATH = 'selected_features_v4.pkl'
SEQ_LEN = 60
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.10 # Réduit car on a beaucoup de données
BATCH_SIZE = 512 # Optimisé pour RTX 3060Ti
EPOCHS = 30
FUTURE_SPLIT_DATE = '2025-06-01' # On garde les 5 derniers mois pour le test (Rappel: tes données vont jusqu'en nov 2025)

# --- GPU SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

def load_data_v4():
    if not os.path.exists(FEATURES_PATH):
        print("ERREUR: 'selected_features_v4.pkl' manquant.")
        exit()
    
    features = joblib.load(FEATURES_PATH)
    print(f"✅ Features chargées ({len(features)}) : {features}")
    
    files = glob.glob(os.path.join(DATA_DIR, "*_enriched.csv"))
    train_dfs = []
    test_dfs = []
    
    print("⏳ Chargement des datasets en RAM...")
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            # Nettoyage ultime
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Sélection stricte des colonnes
            # On s'assure d'avoir toutes les features + Target
            cols_needed = features + ['Target']
            if not all(col in df.columns for col in cols_needed):
                continue
                
            df = df[cols_needed]
            
            # Split Temporel
            train = df[df.index < FUTURE_SPLIT_DATE]
            test = df[df.index >= FUTURE_SPLIT_DATE]
            
            if len(train) > SEQ_LEN: train_dfs.append(train)
            if len(test) > SEQ_LEN: test_dfs.append(test)
        except: pass

    # Scaling
    print("⚖️  Entraînement du Scaler (Quantile)...")
    full_train = pd.concat(train_dfs)
    
    # QuantileTransformer est lourd en CPU, on peut sous-échantillonner pour le fit si c'est trop lent
    # Mais pour la précision, on essaye sur tout.
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(full_train[features])
    
    joblib.dump(scaler, 'scaler_v4.pkl')
    
    return train_dfs, test_dfs, scaler, features

def make_dataset(dfs, scaler, features):
    X_list, y_list = [], []
    
    for df in tqdm(dfs, desc="Création Tenseurs"):
        data = scaler.transform(df[features])
        target = df['Target'].values
        
        # Optimisation : On ne prend pas TOUTES les fenêtres glissantes si on a trop de données
        # Sinon on va exploser la RAM (2 ans de 1-min sur 20 actions = millions de séquences)
        # Stratégie : On prend 1 séquence sur 2 ou sur 3 si besoin.
        # Ici on tente tout, si MemoryError, on mettra step=2
        
        # Utilisation de stride_tricks pour aller vite (Vue mémoire, pas de copie)
        # Attention: pour Keras il faut copier à la fin
        n_samples = len(data) - SEQ_LEN
        if n_samples <= 0: continue
        
        # Version boucle simple (plus sûre pour la RAM que stride_tricks sur gros volume)
        # On saute les steps pour éviter la redondance extrême du 1-min
        step = 2 
        for i in range(0, n_samples, step):
            X_list.append(data[i : i+SEQ_LEN])
            y_list.append(target[i+SEQ_LEN-1])
            
    return np.array(X_list, dtype='float32'), np.array(y_list, dtype='int32')

# --- ARCHITECTURE TRANSFORMER (Idem V3 mais adaptée) ---
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

def build_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(NUM_LAYERS):
        x = transformer_encoder(x, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- MAIN ---
if __name__ == "__main__":
    train_dfs, test_dfs, scaler, features = load_data_v4()
    
    print(f"\nConversion en Tenseurs (Step=2)...")
    X_train, y_train = make_dataset(train_dfs, scaler, features)
    X_test, y_test = make_dataset(test_dfs, scaler, features)
    
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    
    # Poids des classes (Essentiel pour le Triple Barrier)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    print(f"Poids des classes : {cw_dict}")

    model = build_model((SEQ_LEN, len(features)), 3)
    
    cbs = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        callbacks.ModelCheckpoint('transformer_v4.keras', save_best_only=True)
    ]

    print("\n--- GO TRAIN V4 ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=cbs
    )

    # Eval
    print("\n--- EVALUATION SOTA V4 ---")
    model.load_weights('transformer_v4.keras')
    
    # On prédit en gros batch pour aller vite
    preds = model.predict(X_test, batch_size=2048, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=['SELL', 'WAIT', 'BUY']))
    
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"🏆 MCC Global : {mcc:.4f}")
    
    # MCC Binaire
    buy_idx = 2
    y_test_bin = (y_test == buy_idx).astype(int)
    y_pred_bin = (y_pred == buy_idx).astype(int)
    mcc_buy = matthews_corrcoef(y_test_bin, y_pred_bin)
    print(f"🚀 MCC Signal Achat : {mcc_buy:.4f}")