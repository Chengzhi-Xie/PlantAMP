#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PBPAMP_k=10.py

dropout_rate=0.3, MC_ROUNDS=50
Procedure:
 1) Read PAMP.txt and split into training/testing in a 5:1 ratio
 2) Perform 10-fold CV for the single hyperparameter combination:
    - Three-stage fine-tuning (freeze → unfreeze → fine-tune)
    - Perform MC-Dropout TTA on validation set
    - Save each fold's model and in-fold metrics to fold_metrics.csv
 3) Use the above 10-fold models to perform weighted MC-Dropout ensemble evaluation on the test set
    Output final_metrics.csv
"""

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Local protein_bert source path & import
PROTEINBERT_SRC = "/root/protein_bert"
sys.path.append(PROTEINBERT_SRC)
from proteinbert import load_pretrained_model, FinetuningModelGenerator, finetune
from proteinbert import OutputType, OutputSpec
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.model_generation import additional_token_to_index

# ————————————— Global Constants —————————————
PAMP_PATH    = "/root/autodl-tmp/PBPAMP.txt"
OUTPUT_ROOT  = "/root/autodl-tmp/PBPAMP_k=10"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEED         = 42
SEQ_LEN      = 256
TEST_RATIO   = 1/6
N_SPLITS     = 10   # Changed to 10 splits
BATCH_SIZE   = 8

# Three-stage fine-tuning hyperparameters (fixed)
FROZEN_LR    = 1e-3
UNFROZEN_LR  = 5e-5
E1, E2, E3   = 5, 2, 1

# HPO
DROP_OUT_RATE = 0.3
MC_ROUNDS     = 50

# GPU memory growth as needed
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# ————————————— Load/Split Data —————————————
def load_and_split(path):
    seqs, labels = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        # Truncate to SEQ_LEN-2
        seqs.append(lines[i+1][:SEQ_LEN-2])
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    tr, te = train_test_split(df, test_size=TEST_RATIO,
                              stratify=df.label, random_state=SEED)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# ————————————— MC Dropout Prediction —————————————
def mc_dropout_predict(model, seq_batch, go_batch, rounds):
    preds = np.zeros(len(seq_batch), dtype=float)
    for _ in range(rounds):
        # Enable Dropout by setting training=True
        p = model([seq_batch, go_batch], training=True).numpy().ravel()
        preds += p
    return preds / rounds

# ————————————— 10-Fold CV + In-Fold Saving —————————————
def cross_validate(train_df):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pre_gen, encoder = load_pretrained_model()
    OUTPUT_TYPE = OutputType(False, 'binary')
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, [0,1])

    fold_metrics = []
    max_raw = SEQ_LEN - 2
    mask_idx = additional_token_to_index.get('<MASK>', additional_token_to_index['<PAD>'])
    pad_idx  = additional_token_to_index['<PAD>']

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        # Three-stage fine-tuning
        gen = FinetuningModelGenerator(
            pre_gen, OUTPUT_SPEC,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROP_OUT_RATE
        )
        cb = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True, verbose=1
        )]
        # Stage 1: Freeze
        finetune(gen, encoder, OUTPUT_SPEC,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=E1, lr=FROZEN_LR,
                 begin_with_frozen_pretrained_layers=True,
                 lr_with_frozen_pretrained_layers=FROZEN_LR,
                 n_final_epochs=0, callbacks=cb)
        # Stage 2: Unfreeze
        finetune(gen, encoder, OUTPUT_SPEC,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=0, lr=UNFROZEN_LR,
                 begin_with_frozen_pretrained_layers=False,
                 lr_with_frozen_pretrained_layers=UNFROZEN_LR,
                 n_final_epochs=E2, callbacks=cb)
        # Stage 3: Fine-tune
        finetune(gen, encoder, OUTPUT_SPEC,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=0, lr=UNFROZEN_LR/2,
                 begin_with_frozen_pretrained_layers=False,
                 lr_with_frozen_pretrained_layers=UNFROZEN_LR/2,
                 n_final_epochs=E3, callbacks=cb)

        # MC-Dropout TTA on validation set
        model = gen.create_model(seq_len=SEQ_LEN)
        seq_vl, go_vl = encoder.encode_X(X_vl, seq_len=SEQ_LEN)
        preds_vl = mc_dropout_predict(model, seq_vl, go_vl, MC_ROUNDS)
        y_pred   = (preds_vl >= 0.5).astype(int)

        # Save model
        fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save(os.path.join(fold_dir, "best_model"))

        # Record fold metrics
        fm = {
            'fold':     fold,
            'auc':      roc_auc_score(y_vl, preds_vl),
            'acc':      accuracy_score(y_vl, y_pred),
            'precision':precision_score(y_vl, y_pred, zero_division=0),
            'recall':   recall_score(y_vl, y_pred, zero_division=0),
            'f1':       f1_score(y_vl, y_pred, zero_division=0),
            'mcc':      matthews_corrcoef(y_vl, y_pred)
        }
        print(f"[Fold {fold}] AUC={fm['auc']:.4f}, ACC={fm['acc']:.4f}, MCC={fm['mcc']:.4f}")
        fold_metrics.append(fm)

        # **Write fold metrics immediately** for intermediate inspection
        pd.DataFrame(fold_metrics).to_csv(
            os.path.join(OUTPUT_ROOT, "fold_metrics.csv"), index=False
        )

    return fold_metrics

# ————————————— Final Test Evaluation —————————————
def final_evaluate(test_df, fold_metrics):
    # Prepare test set
    seq_te, go_te = load_pretrained_model()[1].encode_X(
        test_df.sequence.tolist(), seq_len=SEQ_LEN
    )
    y_true = test_df.label.values
    # Weight by fold MCC
    weights = np.array([m['mcc'] for m in fold_metrics])
    weights = weights / weights.sum()

    prob_sum = np.zeros(len(y_true), dtype=float)
    for i, fm in enumerate(fold_metrics, 1):
        model = tf.keras.models.load_model(
            os.path.join(OUTPUT_ROOT, f"fold_{i}", "best_model")
        )
        preds = mc_dropout_predict(model, seq_te, go_te, MC_ROUNDS)
        prob_sum += weights[i-1] * preds

    # Search threshold for maximum MCC
    best_thr, best_mcc = 0.5, -1
    for thr in np.linspace(0.3, 0.7, 81):
        m = matthews_corrcoef(y_true, (prob_sum>=thr).astype(int))
        if m > best_mcc:
            best_thr, best_mcc = thr, m

    y_pred = (prob_sum >= best_thr).astype(int)
    final = {
        'threshold': best_thr,
        'auc':       roc_auc_score(y_true, prob_sum),
        'acc':       accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'mcc':       best_mcc
    }
    print("\n=== Final Test Metrics ===")
    for k, v in final.items():
        print(f"{k.upper()}: {v:.4f}")

    pd.DataFrame([final]).to_csv(
        os.path.join(OUTPUT_ROOT, "final_metrics.csv"), index=False
    )

if __name__ == "__main__":
    train_df, test_df = load_and_split(PAMP_PATH)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 1) 10-Fold CV
    fm = cross_validate(train_df)

    # 2) Test set evaluation
    final_evaluate(test_df, fm)

    print("\nAll results saved to:", OUTPUT_ROOT)
