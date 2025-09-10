#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust PAMP Classification with ESM2
✔ Batch inference
✔ Illegal amino acid cleaning
✔ GPU memory-safe
"""

import os
import torch
import numpy as np
import pandas as pd
import esm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# -------------------- Config --------------------
SEQ_LEN = 256
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 4  # safe for <12GB GPU
PAMP_PATH = "/root/autodl-tmp/PlantAMP.txt"
SAVE_PATH = "./esm2_output"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Optimize GPU memory
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# -------------------- Valid Amino Acids --------------------
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(seq):
    return ''.join([aa for aa in seq.upper() if aa in VALID_AA])

# -------------------- Data Loader --------------------
def load_data():
    seqs, labels = [], []
    with open(PAMP_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        label = int(lines[i])
        raw_seq = lines[i + 1][:SEQ_LEN]
        clean_seq = clean_sequence(raw_seq)
        if clean_seq:  # Only add non-empty sequences
            labels.append(label)
            seqs.append(clean_seq)
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    return train_test_split(df, test_size=1/6, stratify=df.label, random_state=SEED)

# -------------------- ESM2 Embedding with Batching --------------------
def get_esm2_embeddings(seqs, model, alphabet, batch_size=4):
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []

    for i in range(0, len(seqs), batch_size):
        batch_seqs = [(f"seq{i+j}", s) for j, s in enumerate(seqs[i:i+batch_size])]
        _, _, toks = batch_converter(batch_seqs)
        toks = toks.cuda()

        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        token_reps = out["representations"][33]
        emb = token_reps.mean(dim=1).cpu().numpy()
        all_embeddings.append(emb)

        del toks, out, token_reps
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)

# -------------------- Cross Validation --------------------
def cross_val_evaluate(train_df, model, alphabet):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"[INFO] Fold {fold} running...")

        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        X_tr_emb = get_esm2_embeddings(X_tr, model, alphabet, batch_size=BATCH_SIZE)
        X_vl_emb = get_esm2_embeddings(X_vl, model, alphabet, batch_size=BATCH_SIZE)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr_emb, y_tr)
        prob = clf.predict_proba(X_vl_emb)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics.append({
            'Fold': fold,
            'AUC': roc_auc_score(y_vl, prob),
            'ACC': accuracy_score(y_vl, pred),
            'Precision': precision_score(y_vl, pred, zero_division=0),
            'Recall': recall_score(y_vl, pred, zero_division=0),
            'F1': f1_score(y_vl, pred, zero_division=0),
            'MCC': matthews_corrcoef(y_vl, pred)
        })

    df = pd.DataFrame(metrics)
    df.to_csv(f"{SAVE_PATH}/CV_metrics.csv", index=False)
    print("\n[✔] CV Results saved to CV_metrics.csv")
    print(df.mean(numeric_only=True).round(4))
    return df

# -------------------- Final Test Evaluation --------------------
def final_test_evaluation(train_df, test_df, model, alphabet):
    X_tr = train_df.sequence.tolist()
    y_tr = train_df.label.tolist()
    X_te = test_df.sequence.tolist()
    y_te = test_df.label.tolist()

    X_tr_emb = get_esm2_embeddings(X_tr, model, alphabet, batch_size=BATCH_SIZE)
    X_te_emb = get_esm2_embeddings(X_te, model, alphabet, batch_size=BATCH_SIZE)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr_emb, y_tr)
    prob = clf.predict_proba(X_te_emb)[:, 1]
    pred = (prob >= 0.5).astype(int)

    sorted_df = pd.DataFrame({
        'sequence': X_te,
        'true_label': y_te,
        'pred_prob': prob,
        'pred_label': pred
    }).sort_values(by="pred_prob", ascending=False)
    sorted_df.to_csv(f"{SAVE_PATH}/sorted_predictions.csv", index=False)

    metrics = {
        'AUC': roc_auc_score(y_te, prob),
        'ACC': accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred, zero_division=0),
        'Recall': recall_score(y_te, pred, zero_division=0),
        'F1': f1_score(y_te, pred, zero_division=0),
        'MCC': matthews_corrcoef(y_te, pred)
    }
    pd.DataFrame([metrics]).to_csv(f"{SAVE_PATH}/final_test_metrics.csv", index=False)

    print("\n[✔] Final test metrics saved to final_test_metrics.csv")
    print(metrics)

# -------------------- Main --------------------
def run_pipeline():
    print("[INFO] Loading ESM2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    train_df, test_df = load_data()

    print("[INFO] Running 5-Fold Cross-Validation...")
    cross_val_evaluate(train_df, model, alphabet)

    print("[INFO] Running Final Test Evaluation...")
    final_test_evaluation(train_df, test_df, model, alphabet)

if __name__ == "__main__":
    run_pipeline()
