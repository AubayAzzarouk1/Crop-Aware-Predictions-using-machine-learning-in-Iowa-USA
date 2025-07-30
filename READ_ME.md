DS440W - Group 26

1) MMST-ViT-Baseline: Full implementation of how to verify and replicate the results to that of:
Parent Paper: "Open and Large-Scale Dataset for Multi-Modal Climate Change-aware Crop Yield Predictions" by Fudong Lin et. al (2022)

  1a) To run the baseline, you need to import (or reference class(es): 
    - build_config_soybean.py 
    - attention.py
    - contrastive_loss.py
    - main_finetune_mmst_vit.py
    - main_pretrain_mmst_vit.py
    - models_pvt.py
    - models_pvt_simclr.py (call "pvt_tiny" for light weight model)
    - models_mmst_vit.py
    *** reference: https://github.com/fudong03/MMST-ViT/tree/main for full details and steps to access modality data and Multi-Modal Spatial-Temporal Vision Transformer"

2) Accessing the data: https://huggingface.co/datasets/fudong03/Tiny-CropNet
    - contains IA, IL, MS, & LA county-level usda crop data (soybean/data).
    - daily and monthly high resolution meterological data (ex: min/max temp, vapor pressure...)
    - 4 quarterly Sentinel-2 satellite imagery (.hdf5 files) per year per state.
        ~ note, the tinycropnet sentinel-2 imagery seems to be either corrupted or improperly formatted by Fudong Lin during extraction.
        ~ To resolve this, you must manually access each huggingface url and use wget to download each sentinel-2 image individually to ensure integrity
        ~ refer to tinycropnet_setup_txt for full depth explanation
          
2a) It is recommended that you have at least a high capacity (> 100gb) of storage and disk space to accomdate for the enhanced
     satellite imagery and mmst_vit_model checkpoints (over 25mb in data, could not store in github).
      - each .hdf5 file contains 4gb worth of multi-spectral images (computationally expensive to run without enough storage)
      - it is recommended that you utilize an accelerated gpu (nvidia T4, A100, supercomputers...) to setup for optimal and ideal
        pre-processing speeds and tiny/cropnet data loading.
        
3) Main.py : central pipeline for our enhanced mmst-vit implementation/research on the TinyCropNet dataset
   Components:
     - configuration json file to read/load county-level data: soybean_val_outliers.json
     - Environment and path setup
     - custom dataloaders (hrrr,sentinel,usda)_datasets.
     - early initialization of pvtSimClr and MMST-ViT backbone classes imported from mmst_vit_baseline_imp.ipynb
     - EDA on each individual modality
     - Z-scores and outlier computations
     - Computed Acreage function to jointly measure and analyze soybean yields in contrast to production levels by county (measured in k/ac and bu/ac).
     - 2 seperate inferences on both 2 and 3 different modalities on outlier counties in the state of iowa (usda + hrrr & usda + hrrr + sentinel-2 )
       -  - mmstvit_real_outlier_predictions.csv - output inference values on 30 outlier county samples.
     - Evaluation and Validations (r2,rmse, corr, etc..) Metrics
    
