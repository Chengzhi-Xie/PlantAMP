# PBPAMP
PBPAMP is a fine‐tuning model built on the pretrained protein large language model (LLM) `ProteinBERT`, designed for the precise recognition and identification of plant antimicrobial peptides (PAMPs).

## Structure
![Fig  1_1](https://github.com/user-attachments/assets/938e2bee-d409-4300-903b-d484223f025e)
Schematic Overview of the PBPAMP Workflow. The framework comprises four key stages: data aggregation, ProteinBERT invocation, model fine-tuning, and performance evaluation. (A) Architecture of ProteinBERT: adapted from the original BERT design, it processes sequences via two parallel branches—local (left) and global (right). Six Transformer-style blocks alternately handle these representations through fully connected and convolutional sublayers, with residual connections and layer normalization. Global Attention modules enable local features to inform the global embedding. (B) PAMP Dataset Assembly: experimentally validated PAMP sequences were compiled from published studies and public AMP databases. After merging and deduplication, 2,497 unique PAMPs were retained as positive samples, alongside 2,677 non-PAMPs as negative controls. (C) Model Development and Evaluation: starting from the pretrained ProteinBERT encoder, we (1) froze the backbone and fine-tuned the classification head for 5 epochs at lr=1×10^(-3), (2) unfroze all layers for 2 epochs at lr=5×10^(-5), and (3) performed a final tuning epoch at lr=2.5×10^(-5)—applying Dropout (0.3) and Early Stopping (patience=1) at each stage. For inference, we conducted 50 stochastic forward passes with MC Dropout(rate=0.3) within a 5-fold cross-validation framework, then ensembled predictions by weighting each fold’s output by its validation MCC. Finally, we scanned classification thresholds from 0.30 to 0.70 (step=0.005) to maximize MCC, and reported test-set AUC, ACC, Precision, Recall, F1-score, and MCC.

## Deployments & Dependencies
The model was trained and evaluated on a server (`AutoDL` https://www.autodl.com/home) equipped with a single `NVIDIA` GeForce RTX 4090 GPU (24 GB VRAM), The software environment adopts python 3.10 (ubuntu22.04) and the framework PyTorch 2.1.2 (CUDA 11.8).

ProteinBERT download
git clone https://github.com/nadavbra/protein_bert.git

* python=3.8

* numpy==1.23.5
* pandas==1.5.3    
* scikit-learn==1.2.2  
* ensorflow==2.12.0
* tensorflow-addons==0.21.0
* h5py==3.8.0
* torch==2.0.1+cu118
* torchvision==0.15.2+cu118
* torchaudio==2.0.2+cu118
* pyfaidx==0.6.3
* lxml==4.9.1
* tqdm==4.65.0
* transformers==4.30.2
* datasets==2.12.0
* matplotlib==3.7.1
* seaborn==0.12.2

## Installation
1. Create a new environment: `conda` 
```conda create -n proteinbert_env python=3.8 -y```
2. Activate the environment:
```conda activate proteinbert_env```
3. Install proteinbert_env in the environment:
```pip install numpy==1.23.5```
...

## Datasets
PBPAMP.txt
2,497 unique plant-derived antimicrobial peptide (PAMP) sequences were compiled from six public databases, with non-standard residues and duplicates removed and clustering at 90% identity applied to ensure diversity.
2,677 plant-only non-PAMP sequences were selected from UniProt matched by length distribution, and the combined 5,174 sequences were split into training and test sets (5:1); comparable properties were confirmed, and five-fold cross-validation was applied to the training set.

## Codes
PBPAMP_k=5.py is the final model codes of PBPAMP. The other files in model file shows other versions and comparison codes.
In sequence_analysis file shows correlation analyses performed on PBPAMP.txt.

## Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue or contact me.



