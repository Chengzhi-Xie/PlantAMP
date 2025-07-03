#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProteinBERT_2.py

End-to-end optimized pipeline to fully utilize RTX 4090:
1. Detect and test GPU
2. Enable XLA JIT and mixed precision
3. Load ProteinBERT and pre-encode all sequences (input IDs + global attention mask)
4. Build tf.data.Dataset with cache/prefetch for train/validation/test
5. For each fold:
     - Instantiate a fresh FinetuningModelGenerator
     - Compile & fit with `model.fit(dataset)` in two stages (freeze head, then unfreeze)
     - Evaluate on fold's validation set
     - Save best model and metrics
6. Final training on full training set + test evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report
)

# ── 1. XLA JIT + threading setup ─────────────────────────────
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# ── 2. GPU detection & simple test ───────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPU detected. Please check CUDA drivers.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("✅ GPUs detected:", gpus)

with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0])
    print("✔ GPU test computation 1+2+3 =", tf.reduce_sum(x).numpy())

# ── 3. Mixed precision ───────────────────────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("🔧 Mixed precision policy:", mixed_precision.global_policy())

# ── 4. Paths & hyperparameters ──────────────────────────────
PROTEINBERT_SRC = "/root/protein_bert"
PAMP_PATH       = "/root/autodl-tmp/PBPAMP.txt"
OUTPUT_ROOT     = "/root/autodl-tmp/ProteinBERT_2"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
sys.path.append(PROTEINBERT_SRC)

SEQ_LEN         = 128
TEST_RATIO      = 1/6
N_SPLITS        = 5
RND_SEED        = 42
PATIENCE        = 2

BATCH_SIZE      = 32
FROZEN_LR       = 1e-3
UNFROZEN_LR     = 5e-5
EPOCHS_FROZEN   = 5
EPOCHS_UNFROZEN = 2
DROPOUT_RATE    = 0.3

# ── 5. Import ProteinBERT ────────────────────────────────────
from proteinbert import (
    load_pretrained_model,
    OutputType, OutputSpec,
    FinetuningModelGenerator
)
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# ── 6. Load & split PAMP.txt ────────────────────────────────
def load_pamp(path):
    seqs, labels = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        seqs.append(lines[i+1])
    return pd.DataFrame({'sequence': seqs, 'label': labels})

df = load_pamp(PAMP_PATH)
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO,
    stratify=df.label, random_state=RND_SEED
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"🚀 Samples: Train {len(train_df)} (+{train_df.label.sum()} positives), Test {len(test_df)} (+{test_df.label.sum()} positives)")

# ── 7. Pre-encode all sequences ──────────────────────────────
pre_gen, encoder = load_pretrained_model()
max_raw = SEQ_LEN - 2

def encode_list(seqs):
    truncated = [s[:max_raw] for s in seqs]
    ids, mask = encoder.encode_X(truncated, seq_len=SEQ_LEN)
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)

X_train_ids, X_train_mask = encode_list(train_df.sequence.tolist())
y_train = train_df.label.values
X_test_ids, X_test_mask   = encode_list(test_df.sequence.tolist())
y_test  = test_df.label.values

print("✅ Pre-encoding completed:", X_train_ids.shape, X_train_mask.shape)

# ── 8. Build tf.data.Dataset ────────────────────────────────
def make_dataset(ids, mask, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((ids, mask), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=RND_SEED)
    return ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# ── 9. Cross-validation training & evaluation ───────────────
def cv_train():
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
    records = []
    out_spec = OutputSpec(OutputType(False, 'binary'), [0,1])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_ids, y_train), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        train_ids, train_mask = X_train_ids[tr_idx], X_train_mask[tr_idx]
        val_ids, val_mask     = X_train_ids[va_idx], X_train_mask[va_idx]
        train_lbl, val_lbl    = y_train[tr_idx], y_train[va_idx]

        ds_tr = make_dataset(train_ids, train_mask, train_lbl, shuffle=True)
        ds_va = make_dataset(val_ids, val_mask, val_lbl, shuffle=False)

        # Build model generator
        gen = FinetuningModelGenerator(
            pre_gen, out_spec,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROPOUT_RATE
        )
        # Frozen stage
        model = gen.create_model(seq_len=SEQ_LEN)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(FROZEN_LR),
            loss='binary_crossentropy', metrics=['accuracy']
        )
        model.fit(
            ds_tr, validation_data=ds_va,
            epochs=EPOCHS_FROZEN,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
            ]
        )

        # Unfreeze stage: reinstantiate generator to load new weights and unfreeze
        gen = FinetuningModelGenerator(
            pre_gen, out_spec,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROPOUT_RATE
        )
        model = gen.create_model(seq_len=SEQ_LEN)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(UNFROZEN_LR),
            loss='binary_crossentropy', metrics=['accuracy']
        )
        model.fit(
            ds_tr, validation_data=ds_va,
            epochs=EPOCHS_UNFROZEN,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
            ]
        )

        # Validation set evaluation
        y_prob = model.predict(ds_va)
        y_pred = (y_prob >= 0.5).astype(int).reshape(-1)
        m = {
            'fold': fold,
            'auc':  roc_auc_score(val_lbl, y_prob),
            'acc':  accuracy_score(val_lbl, y_pred),
            'prec': precision_score(val_lbl, y_pred, zero_division=0),
            'rec':  recall_score(val_lbl, y_pred, zero_division=0),
            'f1':   f1_score(val_lbl, y_pred, zero_division=0),
            'mcc':  matthews_corrcoef(val_lbl, y_pred)
        }
        print("→", m)
        records.append(m)

        # Save model
        os.makedirs(f"{OUTPUT_ROOT}/fold_{fold}", exist_ok=True)
        model.save(f"{OUTPUT_ROOT}/fold_{fold}/best_model")

    return pd.DataFrame(records)

df_cv = cv_train()
print("\n===== CV Means =====")
print(df_cv.mean().apply(lambda x: f"{x:.4f}"))
df_cv.to_csv(f"{OUTPUT_ROOT}/cv_metrics.csv", index=False)

# ── 10. Final full-data training & test evaluation ───────────
def final_train_and_eval():
    ds_tr = make_dataset(X_train_ids, X_train_mask, y_train, shuffle=True)
    ds_va = make_dataset(X_train_ids, X_train_mask, y_train, shuffle=False)

    out_spec = OutputSpec(OutputType(False, 'binary'), [0,1])

    # Freeze
    gen = FinetuningModelGenerator(
        pre_gen, out_spec,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=DROPOUT_RATE
    )
    model = gen.create_model(seq_len=SEQ_LEN)
    model.compile(optimizer=tf.keras.optimizers.Adam(FROZEN_LR),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_FROZEN,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True)])

    # Unfreeze
    gen = FinetuningModelGenerator(
        pre_gen, out_spec,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=DROPOUT_RATE
    )
    model = gen.create_model(seq_len=SEQ_LEN)
    model.compile(optimizer=tf.keras.optimizers.Adam(UNFROZEN_LR),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_UNFROZEN,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True)])

    # Test set evaluation
    ds_test = make_dataset(X_test_ids, X_test_mask, y_test, shuffle=False)
    y_prob = model.predict(ds_test)
    y_pred = (y_prob >= 0.5).astype(int).reshape(-1)
    print("\n=== Test Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    os.makedirs(f"{OUTPUT_ROOT}/final_model", exist_ok=True)
    model.save(f"{OUTPUT_ROOT}/final_model/best_model")
    pd.DataFrame([{
        'auc':  roc_auc_score(y_test, y_prob),
        'acc':  accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred, zero_division=0),
        'rec':  recall_score(y_test, y_pred, zero_division=0),
        'f1':   f1_score(y_test, y_pred, zero_division=0),
        'mcc':  matthews_corrcoef(y_test, y_pred)
    }]).to_csv(f"{OUTPUT_ROOT}/final_model/test_metrics.csv", index=False)

final_train_and_eval()

print("\n🎉 All done! Results saved in", OUTPUT_ROOT)
