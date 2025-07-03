#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProteinBERT_4.py

- On script start, detect and test GPU;
- Enable XLA JIT and mixed precision;
- Batch Size=32 to maximize RTX 4090 utilization;
- Pre-truncate sequences to avoid encode_X dimension mismatch errors;
- 5-fold CV + final full-data training & test evaluation.
"""

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report
)

# ── 0. XLA JIT + threading parallelism ─────────────────────────
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# ── 1. GPU detection & basic test ───────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("❌ No GPU detected. Please verify drivers and CUDA setup.")
    sys.exit(1)
print("✅ Detected GPU devices:", gpus)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# Basic computation test on GPU
tf_device = "/GPU:0"
with tf.device(tf_device):
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.reduce_sum(a)
print("✔ GPU test computation (1+2+3) =", b.numpy())

# ── 2. Mixed precision ─────────────────────────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
print("🔧 Mixed precision policy enabled:", mixed_precision.global_policy())

# ── 3. Paths & hyperparameters ─────────────────────────────────
PROTEINBERT_SRC = "/root/protein_bert"
PAMP_PATH       = "/root/autodl-tmp/PBPAMP.txt"
OUTPUT_ROOT     = "/root/autodl-tmp/ProteinBERT_4"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
sys.path.append(PROTEINBERT_SRC)

SEQ_LEN         = 128
TEST_RATIO      = 1/6
N_SPLITS        = 5
RND_SEED        = 42
PATIENCE        = 2

BATCH_SIZE      = 32      # Increased to 32
FROZEN_LR       = 1e-3
UNFROZEN_LR     = 5e-5
EPOCHS_FROZEN   = 5
EPOCHS_UNFROZEN = 2
DROPOUT_RATE    = 0.3

# ── 4. Import ProteinBERT ─────────────────────────────────────
from proteinbert import (
    load_pretrained_model,
    OutputType, OutputSpec,
    FinetuningModelGenerator, finetune
)
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# ── 5. Load PAMP.txt → DataFrame ──────────────────────────────
def load_pamp(path):
    sequences, labels = [], []
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        sequences.append(lines[i+1])
    return pd.DataFrame({"sequence": sequences, "label": labels})

# Prepare train/test splits
df = load_pamp(PAMP_PATH)
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO,
    stratify=df.label, random_state=RND_SEED
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"🚀 Dataset: Train={len(train_df)} (+{train_df.label.sum()} positives)  Test={len(test_df)} (+{test_df.label.sum()} positives)")

# ── 6. 5-Fold CV + training/evaluation ──────────────────────────
def cv_train(train_df, output_root):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
    records = []

    pretrained_gen, encoder = load_pretrained_model()
    output_spec = OutputSpec(OutputType(False, "binary"), [0,1])
    max_raw = SEQ_LEN - 2

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df.sequence, train_df.label), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        # Pre-truncate sequences to avoid encode_X shape mismatch
        X_tr = [seq[:max_raw] for seq in train_df.sequence.iloc[tr_idx]]
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_va = [seq[:max_raw] for seq in train_df.sequence.iloc[va_idx]]
        y_va = train_df.label.iloc[va_idx].tolist()

        # Instantiate a new fine-tuning generator for each fold
        gen = FinetuningModelGenerator(
            pretrained_gen, output_spec,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROPOUT_RATE
        )
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", patience=1, factor=0.25, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping   ("val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
        ]

        # Stage1: Freeze pretrained layers
        print(" Stage1: freeze pretrained layers")
        finetune(
            gen, encoder, output_spec,
            X_tr, y_tr, X_va, y_va,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            max_epochs_per_stage=EPOCHS_FROZEN,
            lr=FROZEN_LR,
            begin_with_frozen_pretrained_layers=True,
            lr_with_frozen_pretrained_layers=FROZEN_LR,
            n_final_epochs=0,
            callbacks=callbacks
        )

        # Stage2: Unfreeze all layers
        print(" Stage2: unfreeze all layers")
        finetune(
            gen, encoder, output_spec,
            X_tr, y_tr, X_va, y_va,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            max_epochs_per_stage=0,
            lr=UNFROZEN_LR,
            begin_with_frozen_pretrained_layers=False,
            lr_with_frozen_pretrained_layers=UNFROZEN_LR,
            n_final_epochs=EPOCHS_UNFROZEN,
            callbacks=callbacks
        )

        # Evaluation: encode_X then predict
        print(" Evaluating on validation set...")
        ids_va, mask_va = encoder.encode_X(X_va, seq_len=SEQ_LEN)
        model = gen.create_model(seq_len=SEQ_LEN)
        y_prob = model.predict([
            np.array(ids_va, dtype=np.int32),
            np.array(mask_va, dtype=np.int32)
        ], batch_size=BATCH_SIZE)
        y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

        metrics = {
            "fold": fold,
            "auc":  roc_auc_score(y_va, y_prob),
            "acc":  accuracy_score(y_va, y_pred),
            "prec": precision_score(y_va, y_pred, zero_division=0),
            "rec":  recall_score(y_va, y_pred, zero_division=0),
            "f1":   f1_score(y_va, y_pred, zero_division=0),
            "mcc":  matthews_corrcoef(y_va, y_pred)
        }
        print(metrics)
        records.append(metrics)

        # Save best model for this fold
        fold_dir = os.path.join(output_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save(os.path.join(fold_dir, "best_model"))

    return pd.DataFrame(records)

# Run CV
cv_df = cv_train(train_df, OUTPUT_ROOT)
print("\nCV results:\n", cv_df)
print("\nCV mean:\n", cv_df.mean().apply(lambda x: f"{x:.4f}"))
cv_df.to_csv(os.path.join(OUTPUT_ROOT, "cv_metrics.csv"), index=False)

# ── 7. Final full-data training & test evaluation ─────────────
def final_train_and_eval(train_df, test_df, output_root):
    pretrained_gen, encoder = load_pretrained_model()
    output_spec = OutputSpec(OutputType(False, "binary"), [0,1])
    gen = FinetuningModelGenerator(
        pretrained_gen, output_spec,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=DROPOUT_RATE
    )
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", patience=1, factor=0.25, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping   ("val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
    ]

    max_raw = SEQ_LEN - 2
    X_tr = [seq[:max_raw] for seq in train_df.sequence]
    y_tr = train_df.label.tolist()
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=RND_SEED
    )

    # Stage1
    finetune(
        gen, encoder, output_spec,
        X_tr2, y_tr2, X_val2, y_val2,
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        max_epochs_per_stage=EPOCHS_FROZEN,
        lr=FROZEN_LR, begin_with_frozen_pretrained_layers=True,
        lr_with_frozen_pretrained_layers=FROZEN_LR,
        n_final_epochs=0, callbacks=callbacks
    )
    # Stage2
    finetune(
        gen, encoder, output_spec,
        X_tr2, y_tr2, X_val2, y_val2,
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        max_epochs_per_stage=0,
        lr=UNFROZEN_LR, begin_with_frozen_pretrained_layers=False,
        lr_with_frozen_pretrained_layers=UNFROZEN_LR,
        n_final_epochs=EPOCHS_UNFROZEN, callbacks=callbacks
    )

    print("\n=== Evaluating on test set ===")
    ids_te, mask_te = encoder.encode_X(
        [seq[:max_raw] for seq in test_df.sequence], seq_len=SEQ_LEN
    )
    model = gen.create_model(seq_len=SEQ_LEN)
    y_prob = model.predict([
        np.array(ids_te, dtype=np.int32),
        np.array(mask_te, dtype=np.int32)
    ], batch_size=BATCH_SIZE)
    y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

    print(classification_report(test_df.label, y_pred, digits=4))
    df_test = pd.DataFrame([{
        "auc":  roc_auc_score(test_df.label, y_prob),
        "acc":  accuracy_score(test_df.label, y_pred),
        "prec": precision_score(test_df.label, y_pred, zero_division=0),
        "rec":  recall_score(test_df.label, y_pred, zero_division=0),
        "f1":   f1_score(test_df.label, y_pred, zero_division=0),
        "mcc":  matthews_corrcoef(test_df.label, y_pred)
    }])
    final_model_dir = os.path.join(output_root, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save(os.path.join(final_model_dir, "best_model"))
    df_test.to_csv(os.path.join(final_model_dir, "test_metrics.csv"), index=False)

# Execute final training and evaluation
final_train_and_eval(train_df, test_df, OUTPUT_ROOT)
print("\n🎉 All done. Outputs in", OUTPUT_ROOT)
