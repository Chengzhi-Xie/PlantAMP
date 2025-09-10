#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PAMP Classification with ProtT5 Embeddings (Batch + Safe)
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# -------------------- Config --------------------
SEQ_LEN = 256
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 2  # ProtT5 big，recom 2
PAMP_PATH = "/root/autodl-tmp/PlantAMP.txt"
SAVE_PATH = "./prott5_output"
os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------- delete the non-stan AA --------------------
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(seq):
    return ' '.join([aa for aa in seq.upper() if aa in VALID_AA])

# -------------------- Load data --------------------
def load_data():
    seqs, labels = [], []
    with open(PAMP_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        label = int(lines[i])
        raw_seq = lines[i + 1][:SEQ_LEN]
        clean_seq = clean_sequence(raw_seq)
        if clean_seq:
            labels.append(label)
            seqs.append(clean_seq)
    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    return train_test_split(df, test_size=1/6, stratify=df.label, random_state=SEED)

# -------------------- ProtT5 embeding sele --------------------
def get_embeddings(seqs, tokenizer, model, batch_size=2):
    model.eval().cuda()
    embeddings = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        last_hidden = output.last_hidden_state  # (B, L, D)
        mean_emb = last_hidden.mean(dim=1).cpu().numpy()  # (B, D)
        embeddings.append(mean_emb)

        # release the GPU
        del tokens, output, last_hidden
        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)

# -------------------- train & validation --------------------
def run_ablation():
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    train_df, _ = load_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"[Fold {fold}]")

        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        X_tr_emb = get_embeddings(X_tr, tokenizer, model, batch_size=BATCH_SIZE)
        X_vl_emb = get_embeddings(X_vl, tokenizer, model, batch_size=BATCH_SIZE)

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
    df.to_csv(f"{SAVE_PATH}/ProtT5_ablation_results.csv", index=False)
    print("\n✔ Results saved to ProtT5_ablation_results.csv")
    print(df.mean(numeric_only=True).round(4))

# -------------------- main function --------------------
if __name__ == "__main__":
    run_ablation()
