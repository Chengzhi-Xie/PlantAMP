In this part, the main code for PlantAMP is PlantAMP_ROC_PRC_Confusion.py, which include final ROC, PRC and confusion matrix
PlantAMP_k=10.py and PlantAMP_k=5.py applied to contrastive cross-validation
PlantAMP_Ablation.py intended for model ablation experiment
PB_LoRA.py and PB_bitfit.py intended for other optimization comparisons
PAMP_ESM2.py and PAMP_ProtT5.py for performance comparisons of LLMs, refer to the environment configurations in plantamp_esm2_env.yml and plantamp_ProtT5_env.yml
In other_tuning_version fold including fine-tuning test for different version, the experimental results were all inferior to the final version

In addition, PBPAMP.txt in all of the code means PlantAMP.txt
