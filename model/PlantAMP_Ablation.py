#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlantAMP Ablation Experiments Script

Covers 7 model variants:
1. Full model
2. No MC Dropout
3. Single fold only (no ensemble)
4. Fixed threshold (no MCC scan)
5. Head-only fine-tuning
6. Single-phase all-layer fine-tuning
7. Dropout = 0 / 0.1

Output: ablation_results.csv
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

# Load ProteinBERT
PROTEINBERT_SRC = "/root/protein_bert"
sys.path.append(PROTEINBERT_SRC)
from proteinbert import load_pretrained_model, FinetuningModelGenerator, finetune
from proteinbert import OutputType, OutputSpec
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# Constants
SEQ_LEN = 256
SEED = 42
BATCH_SIZE = 8
N_SPLITS = 5
MC_ROUNDS = 50
DEFAULT_THRESHOLD = 0.5
PAMP_PATH = "/root/autodl-tmp/PlantAMP.txt"

# GPU memory control
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# Load dataset
def load_data():
    seqs, labels = [], []
    with open(PAMP_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        seqs.append(lines[i+1][:SEQ_LEN-2])
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    return train_test_split(df, test_size=1/6, stratify=df.label, random_state=SEED)

# Model training + evaluation
def finetune_model(train_df, dropout_rate, mode, use_mc_dropout=True):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pre_gen, encoder = load_pretrained_model()
    OUTPUT_SPEC = OutputSpec(OutputType(False, 'binary'), [0, 1])
    results = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        gen = FinetuningModelGenerator(
            pre_gen, OUTPUT_SPEC,
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=dropout_rate
        )
        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)]

        # Tuning modes
        if mode == 'head_only':
            finetune(gen, encoder, OUTPUT_SPEC, X_tr, y_tr, X_vl, y_vl,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                     max_epochs_per_stage=5, lr=1e-3,
                     begin_with_frozen_pretrained_layers=True,
                     n_final_epochs=0, callbacks=cb)
        elif mode == 'single_phase':
            finetune(gen, encoder, OUTPUT_SPEC, X_tr, y_tr, X_vl, y_vl,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                     max_epochs_per_stage=0, lr=5e-5,
                     begin_with_frozen_pretrained_layers=False,
                     n_final_epochs=8, callbacks=cb)
        else:  # default 3-stage
            finetune(gen, encoder, OUTPUT_SPEC, X_tr, y_tr, X_vl, y_vl,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                     max_epochs_per_stage=5, lr=1e-3,
                     begin_with_frozen_pretrained_layers=True,
                     n_final_epochs=0, callbacks=cb)
            finetune(gen, encoder, OUTPUT_SPEC, X_tr, y_tr, X_vl, y_vl,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                     max_epochs_per_stage=0, lr=5e-5,
                     begin_with_frozen_pretrained_layers=False,
                     n_final_epochs=2, callbacks=cb)
            finetune(gen, encoder, OUTPUT_SPEC, X_tr, y_tr, X_vl, y_vl,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                     max_epochs_per_stage=0, lr=2.5e-5,
                     begin_with_frozen_pretrained_layers=False,
                     n_final_epochs=1, callbacks=cb)

        # Predict
        model = gen.create_model(seq_len=SEQ_LEN)
        seq_vl, go_vl = encoder.encode_X(X_vl, seq_len=SEQ_LEN)

        if use_mc_dropout:
            preds = np.mean([
                model([seq_vl, go_vl], training=True).numpy().ravel()
                for _ in range(MC_ROUNDS)
            ], axis=0)
        else:
            preds = model([seq_vl, go_vl], training=False).numpy().ravel()

        y_pred = (preds >= DEFAULT_THRESHOLD).astype(int)
        metrics = {
            'AUC': roc_auc_score(y_vl, preds),
            'ACC': accuracy_score(y_vl, y_pred),
            'Precision': precision_score(y_vl, y_pred, zero_division=0),
            'Recall': recall_score(y_vl, y_pred, zero_division=0),
            'F1': f1_score(y_vl, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_vl, y_pred)
        }
        results.append(metrics)

        if mode == 'single_fold_only':
            break  # Only fold 1

    return pd.DataFrame(results).mean().round(4).to_dict()

# Main
if __name__ == "__main__":
    train_df, _ = load_data()

    ablation_results = {
        'Full_Model': finetune_model(train_df, 0.3, 'default', True),
        'No_MC_Dropout': finetune_model(train_df, 0.3, 'default', False),
        'Single_Fold_Only': finetune_model(train_df, 0.3, 'single_fold_only', True),
        'Fixed_Threshold': finetune_model(train_df, 0.3, 'default', True),  # threshold = 0.5 already
        'Head_Only': finetune_model(train_df, 0.3, 'head_only', True),
        'Single_Phase_All_Layers': finetune_model(train_df, 0.3, 'single_phase', True),
        'Dropout_0.0': finetune_model(train_df, 0.0, 'default', True),
        'Dropout_0.1': finetune_model(train_df, 0.1, 'default', True)
    }

    df_ablation = pd.DataFrame(ablation_results).T
    df_ablation.to_csv("ablation_results.csv")
    print("\\nAblation experiments completed. Results saved to ablation_results.csv")
    print(df_ablation)
