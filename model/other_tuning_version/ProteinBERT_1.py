#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProteinBERT_1.py

Complete offline version based on local protein_bert:
  - Split training/testing in a 5:1 ratio
  - 5-fold cross-validation (three-stage freeze → unfreeze → fine-tune)
  - Record 6 metrics per fold (AUC/ACC/Precision/Recall/F1/MCC)
  - Ensemble on test set (average probabilities), record 6 metrics
"""

import os, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 1) Local protein_bert source path & import
PROTEINBERT_SRC = "/root/protein_bert"
sys.path.append(PROTEINBERT_SRC)
from proteinbert import load_pretrained_model, FinetuningModelGenerator, finetune
from proteinbert import OutputType, OutputSpec
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# 2) Global configuration
PAMP_PATH    = "/root/autodl-tmp/PBPAMP.txt"
OUTPUT_ROOT  = "/root/autodl-tmp/ProteinBERT_1"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEED         = 42
SEQ_LEN      = 128       # including <CLS> & <SEP>
TEST_RATIO   = 1/6
N_SPLITS     = 5
BATCH_SIZE   = 8
PATIENCE     = 2

# Three-stage fine-tuning hyperparameters (HPO optimal)
FROZEN_LR    = 1e-3
UNFROZEN_LR  = 5e-5
E1, E2, E3   = 5, 2, 1
DROPOUT_RATE = 0.3

# GPU memory growth as needed
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)


def load_and_split(path):
    """Read PAMP.txt and split into train/test in a 5:1 ratio"""
    seqs, labels = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        seqs.append(lines[i+1])
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    tr, te = train_test_split(
        df, test_size=TEST_RATIO,
        stratify=df.label, random_state=SEED
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def cross_validate(train_df):
    """5-fold cross-validation; return a DataFrame containing 6 metrics per fold"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pretrained_gen, encoder = load_pretrained_model()
    OUTPUT_TYPE = OutputType(False, 'binary')
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, [0, 1])

    records = []
    max_raw = SEQ_LEN - 2

    for fold, (ti, vi) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        # Construct sequence and label lists
        Xtr = train_df.sequence.iloc[ti].map(lambda s: s[:max_raw]).tolist()
        ytr = train_df.label.iloc[ti].tolist()
        Xvl = train_df.sequence.iloc[vi].map(lambda s: s[:max_raw]).tolist()
        yvl = train_df.label.iloc[vi].tolist()

        # Create generator
        gen = FinetuningModelGenerator(
            pretrained_gen, OUTPUT_SPEC,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROPOUT_RATE
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1,
                restore_best_weights=True, verbose=1
            )
        ]

        # Stage 1: Freeze pretrained layers, train only the head
        finetune(
            gen, encoder, OUTPUT_SPEC,
            Xtr, ytr, Xvl, yvl,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            max_epochs_per_stage=E1, lr=FROZEN_LR,
            begin_with_frozen_pretrained_layers=True,
            lr_with_frozen_pretrained_layers=FROZEN_LR,
            n_final_epochs=0,
            callbacks=callbacks
        )

        # Stage 2: Unfreeze all network layers
        finetune(
            gen, encoder, OUTPUT_SPEC,
            Xtr, ytr, Xvl, yvl,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            max_epochs_per_stage=0, lr=UNFROZEN_LR,
            begin_with_frozen_pretrained_layers=False,
            lr_with_frozen_pretrained_layers=UNFROZEN_LR,
            n_final_epochs=E2,
            callbacks=callbacks
        )

        # Stage 3: Fully unfreeze, fine-tune for few epochs
        finetune(
            gen, encoder, OUTPUT_SPEC,
            Xtr, ytr, Xvl, yvl,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            max_epochs_per_stage=0, lr=UNFROZEN_LR / 2,
            begin_with_frozen_pretrained_layers=False,
            lr_with_frozen_pretrained_layers=UNFROZEN_LR / 2,
            n_final_epochs=E3,
            callbacks=callbacks
        )

        # Evaluate on validation set
        model = gen.create_model(seq_len=SEQ_LEN)
        Xv_enc = encoder.encode_X(Xvl, seq_len=SEQ_LEN)
        y_prob = model.predict(Xv_enc, batch_size=BATCH_SIZE).ravel()

        # Grid search for best threshold
        best = {'thr': 0.5, 'mcc': -1}
        for thr in np.linspace(0.3, 0.7, 41):
            y_pred = (y_prob >= thr).astype(int)
            m = matthews_corrcoef(yvl, y_pred)
            if m > best['mcc']:
                best = {'thr': thr, 'mcc': m}

        y_pred = (y_prob >= best['thr']).astype(int)
        rec = {
            'fold':      fold,
            'threshold': best['thr'],
            'auc':       roc_auc_score(yvl, y_prob),
            'acc':       accuracy_score(yvl, y_pred),
            'precision': precision_score(yvl, y_pred, zero_division=0),
            'recall':    recall_score(yvl, y_pred, zero_division=0),
            'f1':        f1_score(yvl, y_pred, zero_division=0),
            'mcc':       best['mcc']
        }
        print(f"[Fold {fold}] AUC={rec['auc']:.4f}, ACC={rec['acc']:.4f}, MCC={rec['mcc']:.4f}")
        records.append(rec)

        # Save model
        fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save(os.path.join(fold_dir, "best_model"))

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(os.path.join(OUTPUT_ROOT, "fold_metrics.csv"), index=False)
    return df_rec


def ensemble_and_evaluate(test_df, cv_df):
    """Load each fold's model, average test set probabilities, then evaluate six metrics"""
    max_raw = SEQ_LEN - 2
    Xte = test_df.sequence.map(lambda s: s[:max_raw]).tolist()
    yte = test_df.label.values

    _, encoder = load_pretrained_model()
    Xte_enc = encoder.encode_X(Xte, seq_len=SEQ_LEN)

    prob_sum = np.zeros(len(yte))
    for fold in cv_df.fold:
        m = tf.keras.models.load_model(
            os.path.join(OUTPUT_ROOT, f"fold_{fold}", "best_model")
        )
        prob_sum += m.predict(Xte_enc, batch_size=BATCH_SIZE).ravel()

    prob_avg = prob_sum / N_SPLITS
    thr = cv_df.threshold.mean()
    y_pred = (prob_avg >= thr).astype(int)

    final = {
        'threshold': thr,
        'auc':       roc_auc_score(yte, prob_avg),
        'acc':       accuracy_score(yte, y_pred),
        'precision': precision_score(yte, y_pred, zero_division=0),
        'recall':    recall_score(yte, y_pred, zero_division=0),
        'f1':        f1_score(yte, y_pred, zero_division=0),
        'mcc':       matthews_corrcoef(yte, y_pred)
    }
    print("\n=== Final Ensemble Metrics ===")
    for k, v in final.items():
        print(f"{k.upper()}: {v:.4f}")

    pd.DataFrame([final]).to_csv(
        os.path.join(OUTPUT_ROOT, "final_metrics.csv"), index=False
    )


if __name__ == "__main__":
    # 1. Split
    train_df, test_df = load_and_split(PAMP_PATH)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 2. CV
    cv_df = cross_validate(train_df)
    print("\nFold metrics:\n", cv_df)

    # 3. Ensemble & test evaluation
    ensemble_and_evaluate(test_df, cv_df)
    print("\nDone. Results in:", OUTPUT_ROOT)
