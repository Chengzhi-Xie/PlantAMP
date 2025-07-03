#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProteinBERT_3.py

Using local protein_bert code on PBPAMP.txt:

1) Split training/testing in a 5:1 ratio
2) 5-fold cross-validation (three-stage fine-tuning)
3) Validation set threshold search
4) Final full-data training + test set evaluation

Fully offline: no dependency on online HuggingFace models or tokenizer.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 1) Local protein_bert source path
PROTEINBERT_SRC = "/root/protein_bert"
sys.path.append(PROTEINBERT_SRC)

# 2) Import local model interfaces
from proteinbert import load_pretrained_model, FinetuningModelGenerator, finetune
from proteinbert import OutputType, OutputSpec
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# 3) Global configuration
PAMP_PATH   = "/root/autodl-tmp/PBPAMP.txt"
OUTPUT_ROOT = "/root/autodl-tmp/ProteinBERT_3"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

RND_SEED    = 42
SEQ_LEN     = 128        # includes <CLS> and <SEP>
TEST_RATIO  = 1/6
N_SPLITS    = 5
BATCH_SIZE  = 8
PATIENCE    = 2

# Optimal hyperparameters (HPO results)
FROZEN_LR        = 1e-3
UNFROZEN_LR      = 5e-5
EPOCHS_STAGE1    = 5
EPOCHS_STAGE2    = 2
DROPOUT_RATE     = 0.3

# GPU memory growth as needed
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 4) Load and split data
def load_and_split(path):
    seqs, labels = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        seqs.append(lines[i+1])
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    tr, te = train_test_split(
        df, test_size=TEST_RATIO,
        stratify=df['label'], random_state=RND_SEED
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# 5) 5-Fold CV + three-stage fine-tuning + threshold search
def cross_validate(train_df):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
    pretrained_gen, encoder = load_pretrained_model()
    OUTPUT_TYPE = OutputType(False, 'binary')
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, [0,1])

    records = []
    max_raw = SEQ_LEN - 2

    for fold, (ti, vi) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        Xtr = train_df.sequence.iloc[ti].map(lambda s: s[:max_raw]).tolist()
        ytr = train_df.label.iloc[ti].tolist()
        Xvl = train_df.sequence.iloc[vi].map(lambda s: s[:max_raw]).tolist()
        yvl = train_df.label.iloc[vi].tolist()

        # Encode
        Xt = encoder.encode_X(Xtr, seq_len=SEQ_LEN)
        Xv = encoder.encode_X(Xvl, seq_len=SEQ_LEN)

        # Instantiate a new generator for each fold
        model_gen = FinetuningModelGenerator(
            pretrained_gen, OUTPUT_SPEC,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROPOUT_RATE
        )

        # Callbacks
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1,
                restore_best_weights=True, verbose=1
            )
        ]

        # Stage 1: Freeze
        finetune(
            model_gen, encoder, OUTPUT_SPEC,
            Xtr, ytr, Xvl, yvl,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            max_epochs_per_stage=EPOCHS_STAGE1,
            lr=FROZEN_LR,
            begin_with_frozen_pretrained_layers=True,
            lr_with_frozen_pretrained_layers=FROZEN_LR,
            n_final_epochs=0,
            callbacks=cb
        )

        # Stage 2: Unfreeze all
        finetune(
            model_gen, encoder, OUTPUT_SPEC,
            Xtr, ytr, Xvl, yvl,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            max_epochs_per_stage=0,
            lr=UNFROZEN_LR,
            begin_with_frozen_pretrained_layers=False,
            lr_with_frozen_pretrained_layers=UNFROZEN_LR,
            n_final_epochs=EPOCHS_STAGE2,
            callbacks=cb
        )

        # Validation set prediction & threshold search
        model = model_gen.create_model(seq_len=SEQ_LEN)
        prob = model.predict(Xv, batch_size=BATCH_SIZE).ravel()

        best = {'thr':0.5, 'mcc':-1.0}
        for thr in np.linspace(0.3,0.7,41):
            pred = (prob >= thr).astype(int)
            m = matthews_corrcoef(yvl, pred)
            if m > best['mcc']:
                best = {'thr':thr, 'mcc':m}

        y_pred = (prob >= best['thr']).astype(int)
        rec = {
            'fold': fold,
            'threshold': best['thr'],
            'auc': roc_auc_score(yvl, prob),
            'acc': accuracy_score(yvl, y_pred),
            'prec': precision_score(yvl, y_pred, zero_division=0),
            'recall': recall_score(yvl, y_pred, zero_division=0),
            'f1': f1_score(yvl, y_pred, zero_division=0),
            'mcc': best['mcc']
        }
        print(f"[Fold {fold}] thr={best['thr']:.3f}, AUC={rec['auc']:.4f}, ACC={rec['acc']:.4f}, MCC={rec['mcc']:.4f}")

        # Save model
        d = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        os.makedirs(d, exist_ok=True)
        model.save(os.path.join(d, "best_model"))

        records.append(rec)

    return pd.DataFrame(records)

# 6) Final full-data training & test set evaluation
def final_train_and_evaluate(train_df, test_df, thr):
    pretrained_gen, encoder = load_pretrained_model()
    OUTPUT_TYPE = OutputType(False, 'binary')
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, [0,1])

    max_raw = SEQ_LEN - 2
    Xtr = train_df.sequence.map(lambda s:s[:max_raw]).tolist()
    ytr = train_df.label.tolist()
    Xte = test_df.sequence.map(lambda s:s[:max_raw]).tolist()
    yte = test_df.label.tolist()

    Xt = encoder.encode_X(Xtr, seq_len=SEQ_LEN)
    Xv = encoder.encode_X(Xte, seq_len=SEQ_LEN)

    model_gen = FinetuningModelGenerator(
        pretrained_gen, OUTPUT_SPEC,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=DROPOUT_RATE
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1
        )
    ]

    # Stage 1 + Stage 2
    finetune(
        model_gen, encoder, OUTPUT_SPEC,
        Xtr, ytr, Xte, yte,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        max_epochs_per_stage=EPOCHS_STAGE1,
        lr=FROZEN_LR,
        begin_with_frozen_pretrained_layers=True,
        lr_with_frozen_pretrained_layers=FROZEN_LR,
        n_final_epochs=0,
        callbacks=cb
    )
    finetune(
        model_gen, encoder, OUTPUT_SPEC,
        Xtr, ytr, Xte, yte,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        max_epochs_per_stage=0,
        lr=UNFROZEN_LR,
        begin_with_frozen_pretrained_layers=False,
        lr_with_frozen_pretrained_layers=UNFROZEN_LR,
        n_final_epochs=EPOCHS_STAGE2,
        callbacks=cb
    )

    model = model_gen.create_model(seq_len=SEQ_LEN)
    prob = model.predict(Xv, batch_size=BATCH_SIZE).ravel()
    pred = (prob >= thr).astype(int)

    print("\n=== Final Test Metrics ===")
    print("AUC:", roc_auc_score(yte, prob))
    print("ACC:", accuracy_score(yte, pred))
    print("MCC:", matthews_corrcoef(yte, pred))
    print("Confusion:\n", confusion_matrix(yte, pred))

    d = os.path.join(OUTPUT_ROOT, "final_model")
    os.makedirs(d, exist_ok=True)
    model.save(os.path.join(d, "best_model"))
    pd.DataFrame([{
        'threshold': thr,
        'auc': roc_auc_score(yte, prob),
        'acc': accuracy_score(yte, pred),
        'mcc': matthews_corrcoef(yte, pred)
    }]).to_csv(os.path.join(d, "test_metrics.csv"), index=False)

if __name__ == "__main__":
    tr, te = load_and_split(PAMP_PATH)
    cv_df = cross_validate(tr)
    print("\nCV summary:\n", cv_df)
    print("CV mean:\n", cv_df.mean(numeric_only=True))
    global_thr = cv_df.threshold.mean()
    print(f"\n▶ Using threshold={global_thr:.3f} for final test\n")
    final_train_and_evaluate(tr, te, global_thr)
    print("\nAll done. Outputs in:", OUTPUT_ROOT)
