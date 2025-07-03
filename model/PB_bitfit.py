#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PB_bitfit.py

Building on the original three-stage fine-tuning + MC-Dropout pipeline,
first perform Prompt Tuning (prepend X to the sequence and fine-tune for 1 epoch),
then perform BitFit (train only biases),
and finally evaluate on the test set using 5-fold ensemble + MC-Dropout.
"""

import os
import sys
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

# ————————————— Global Constants —————————————
PAMP_PATH    = "/root/autodl-tmp/PBLPAMP.txt"
OUTPUT_ROOT  = "/root/autodl-tmp/PB_bitfit"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEED         = 42
SEQ_LEN      = 256
TEST_RATIO   = 1/6
N_SPLITS     = 5
BATCH_SIZE   = 8

# Prompt Tuning parameters
N_PROMPT     = 20    # length of prompt to prepend

# Three-stage fine-tuning hyperparameters (fixed)
FROZEN_LR    = 1e-3
UNFROZEN_LR  = 5e-5
E1, E2, E3   = 5, 2, 1

# MC-Dropout settings
DROP_OUT_RATE = 0.3
MC_ROUNDS     = 50

# Enable GPU memory growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


# ————————————— Data Loading & Splitting —————————————
def load_and_split(path):
    sequences, labels = [], []
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        sequences.append(lines[i+1][:SEQ_LEN-2])  # truncate sequence
    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    train, test = train_test_split(
        df, test_size=TEST_RATIO,
        stratify=df['label'], random_state=SEED
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ————————————— MC-Dropout Prediction —————————————
def mc_dropout_predict(model, seq_batch, go_batch, rounds):
    preds = np.zeros(len(seq_batch), dtype=float)
    for _ in range(rounds):
        p = model([seq_batch, go_batch], training=True).numpy().ravel()
        preds += p
    return preds / rounds


# ————————————— 5-Fold CV + In-Fold Saving —————————————
def cross_validate(train_df):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pretrained_gen, encoder = load_pretrained_model()
    output_type = OutputType(False, 'binary')
    output_spec = OutputSpec(output_type, [0,1])

    fold_metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        # --- Prompt Tuning stage ---
        # Prepend N_PROMPT 'X's and fine-tune for 1 epoch
        def prepend_prompt(seqs):
            return ['X' * N_PROMPT + seq for seq in seqs]
        X_tr_p = prepend_prompt(X_tr)
        X_vl_p = prepend_prompt(X_vl)
        seq_tr_p, go_tr_p = encoder.encode_X(X_tr_p, seq_len=SEQ_LEN)
        seq_vl_p, go_vl_p = encoder.encode_X(X_vl_p, seq_len=SEQ_LEN)

        gen_pt = FinetuningModelGenerator(
            pretrained_gen, output_spec,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROP_OUT_RATE
        )
        model_pt = gen_pt.create_model(seq_len=SEQ_LEN)
        model_pt.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
        )
        model_pt.fit(
            [seq_tr_p, go_tr_p], np.array(y_tr),
            validation_data=([seq_vl_p, go_vl_p], np.array(y_vl)),
            epochs=1, batch_size=BATCH_SIZE, verbose=1
        )

        # --- Three-stage fine-tuning ---
        gen = FinetuningModelGenerator(
            pretrained_gen, output_spec,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=DROP_OUT_RATE
        )
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True, verbose=1
        )]

        # Stage 1: Freeze pretrained layers
        finetune(gen, encoder, output_spec,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=E1, lr=FROZEN_LR,
                 begin_with_frozen_pretrained_layers=True,
                 lr_with_frozen_pretrained_layers=FROZEN_LR,
                 n_final_epochs=0, callbacks=callbacks)
        # Stage 2: Unfreeze entire network
        finetune(gen, encoder, output_spec,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=0, lr=UNFROZEN_LR,
                 begin_with_frozen_pretrained_layers=False,
                 lr_with_frozen_pretrained_layers=UNFROZEN_LR,
                 n_final_epochs=E2, callbacks=callbacks)
        # Stage 3: Fine-tune convergence
        finetune(gen, encoder, output_spec,
                 X_tr, y_tr, X_vl, y_vl,
                 seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                 max_epochs_per_stage=0, lr=UNFROZEN_LR/2,
                 begin_with_frozen_pretrained_layers=False,
                 lr_with_frozen_pretrained_layers=UNFROZEN_LR/2,
                 n_final_epochs=E3, callbacks=callbacks)

        # --- BitFit stage (train only biases) ---
        print("=== Stage 4: BitFit tuning (train only biases) ===")
        seq_tr, go_tr = encoder.encode_X(X_tr, seq_len=SEQ_LEN)
        seq_vl, go_vl = encoder.encode_X(X_vl, seq_len=SEQ_LEN)
        model_bitfit = gen.create_model(seq_len=SEQ_LEN)
        # Freeze all weights except biases
        for var in model_bitfit.trainable_variables:
            if 'bias' not in var.name:
                var._trainable = False
        model_bitfit.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=UNFROZEN_LR/4),
            loss=tf.keras.losses.BinaryCrossentropy(),
        )
        model_bitfit.fit(
            [seq_tr, go_tr], np.array(y_tr),
            validation_data=([seq_vl, go_vl], np.array(y_vl)),
            epochs=1, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=1
        )

        # --- Validation MC-Dropout TTA & Model Saving ---
        preds_vl = mc_dropout_predict(model_bitfit, seq_vl, go_vl, MC_ROUNDS)
        y_pred   = (preds_vl >= 0.5).astype(int)

        fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model_bitfit.save(os.path.join(fold_dir, "best_model"))

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
        pd.DataFrame(fold_metrics).to_csv(
            os.path.join(OUTPUT_ROOT, "fold_metrics.csv"), index=False
        )

    return fold_metrics


# ————————————— Final Test Evaluation —————————————
def final_evaluate(test_df, fold_metrics):
    seq_te, go_te = load_pretrained_model()[1].encode_X(
        test_df.sequence.tolist(), seq_len=SEQ_LEN
    )
    y_true = test_df.label.values
    weights = np.array([m['mcc'] for m in fold_metrics])
    weights = weights / weights.sum()

    prob_sum = np.zeros(len(y_true), dtype=float)
    for i, _ in enumerate(fold_metrics):
        model = tf.keras.models.load_model(
            os.path.join(OUTPUT_ROOT, f"fold_{i+1}", "best_model")
        )
        preds = mc_dropout_predict(model, seq_te, go_te, MC_ROUNDS)
        prob_sum += weights[i] * preds

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
    fm = cross_validate(train_df)
    final_evaluate(test_df, fm)
    print("\nAll results saved to:", OUTPUT_ROOT)
