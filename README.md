## 🌾 Crop-Aware Crop Yield Predictions with a deep-learning based solution.
This project presents a **climate-aware deep learning solution** for forecasting soybean yields across U.S. counties by fusing satellite, weather, and agricultural data. At the core of this system is the **Multimodal Spatiotemporal Vision Transformer (MMST-ViT)** — a powerful transformer-based model designed to capture both spatial and temporal patterns in crop development.

---

## 🧠 Model Architecture: MMST-ViT

We adopted the **MMST-ViT** architecture for multimodal yield prediction at the county level. The model jointly learns from **visual patterns (Sentinel-2 imagery)**, **environmental variables (WRF-HRRR weather data)**, and **historical yields (USDA data)** to produce interpretable, region-specific forecasts.

### 1. 🎞️ Visual Backbone — PvT (Pyramid Vision Transformer)

- **Input**: Quarterly **Sentinel-2 RGB imagery**  
- **Preprocessing**: Image patches sized **384 × 384** were extracted per county, per quarter.
- **Model**: The PvT backbone captures hierarchical spatial features at each time step.
- **Output**: A sequence of **512-dimensional visual embeddings** representing vegetative development.

> ✅ Imagery was loaded via a custom `Sentinel_Dataset` class with quarterly consolidation. Corrupted `.h5` files were manually restored from Google Drive using `wget`.

### 🛰️ Sentinel-2 Imagery Visualization

Below is a grid of Sentinel-2 bands (B1–B8), including coastal, blue, green, red, red edge, and NIR. These image patches are sampled at 384×384 resolution and serve as visual input to the PvT backbone within MMST-ViT.

![Sentinel-2 Visualization](./sentinel-2%20visualization.png)

---

### 2. 🌦️ Context Features — USDA + WRF-HRRR Weather

- **USDA Yield Data**:
  - Used Ground truth values for county-level soybean yield (BU/acre)
- **WRF-HRRR Weather Features**:
  - Daily & monthly climate variables (temperature, wind speed, radiation, humidity, VPD, etc.)
  - Aggregated into a **540-dimensional context vector** per county-year:
    - `5 seasons × 12 months × 9 climate features`
- **Alignment**:
  - Weather and yield data were temporally aligned with Sentinel imagery using FIPS codes and year-stamps.
  - Missing data and duplicates (e.g., counties with multiple FIPS) were cleaned to ensure consistency.

> ✅ The context tensor was flattened and passed through a projection layer to match the model's fusion dimension (3072).

---

### 3. 🔗 Multimodal Fusion Layer

- A **two-layer transformer** module fuses the image features with the projected context vector.
- This allows the model to learn **interactions** between vegetation signals (e.g., NDVI proxies), seasonal weather trends, and past yield performance.

---

### 4. 🗺️ Spatial & 🕒 Temporal Transformers

- **Spatial Transformer** (4 layers):
  - Learns spatial dependencies across counties and image patches.
  - Helps identify region-level anomalies and common vegetative patterns.

- **Temporal Transformer** (4 layers):
  - Models progression across **6 months** (3–4 quarters).
  - Captures growing season trends such as canopy development, water stress, or dormancy.

---

### 5. 🎯 MLP Prediction Head

- Final **regression head** produces:
  - A scalar yield prediction (BU/acre)
  - A secondary uncertainty or standard deviation estimate
- Fully connected layers include **LayerNorm** for stability and **dropout** for regularization.

---

### 🔧 Implementation Details

| Component              | Shape / Output | Description |
|------------------------|----------------|-------------|
| Sentinel-2 input       | `[1, 6, 8, 3, 384, 384]` | RGB image sequences (quarterly × 2 counties × patches) |
| HRRR+USDA context      | `[1, 540]`     | Climate + yield features |
| Visual Embedding       | `[1, 6, 512]`  | Per-quarter extracted features |
| Fused Feature Vector   | `[1, 3072]`    | Concatenated projection from visual + context |
| Yield Prediction       | `[1, 2]`       | Output: `[predicted_yield, std_dev]` |

> ✅ Final inference was run using `checkpoint-198.pth`, achieving **R² ≈ 0.885** and **RMSE ≈ 2.7 BU/acre**.



- 📦 Performed ablation studies, inference, and regional error analysis

---

## 🗂️ Data Sources

| Modality       | Description                                 | Source |
|----------------|---------------------------------------------|--------|
| Sentinel-2     | Bi-quarterly RGB imagery, spatial patches   | Google Drive (manually curated `.h5` files) |
| WRF-HRRR       | Hourly weather variables (aggregated)       | [TinyCropNet Dataset on Hugging Face](https://huggingface.co/datasets/CropNet/CropNet) |
| USDA NASS      | County-level soybean yields (2017–2022)     | [USDA NASS](https://quickstats.nass.usda.gov/) |
| Parent Paper   | Research Paper                              | [An Open and Large-Scale Dataset for Multi-Modal Climate Change-aware Crop Yield Predictions](https://arxiv.org/abs/2406.06081) |



## 📄 Extended Research & Alternative Approaches

This repository accompanies the research project:

**"Assessing the Impacts of Climate Change on Soybean and Corn Yields and Their Broader Economic Consequences"**  
by Aubay Azzarouk, Penn State University (2025)

📥 [Download Full Report (PDF)](./Assessing%20the%20Impacts%20of%20Climate%20Change%20on%20Soybean%20and%20Corn%20Yields%20and%20Their%20Broader%20Economic%20Consequences.pdf)

---

### 🔬 Summary

This work investigates the impact of climate variability—including rising temperatures, shifting precipitation, and extreme weather—on soybean yield across Iowa’s primary agricultural zones. Using a modified version of the **MMST-ViT** model, we fuse Sentinel-2 imagery, HRRR reanalysis weather data, and USDA yield records to predict county-level yield outcomes from 2017–2022.

Highlights:
- ✅ Achieved **R² ≈ 0.885** and **RMSE ≈ 2.7 BU/acre**
- ✅ Identified spatial yield risks and temporal patterns
- ✅ Produced visual diagnostics of region-specific grid performance
- ✅ Flagged key outlier counties with high/low yield volatility

This pipeline enhances **evidence-based decision-making** for farmers and policymakers, with direct implications for precision agriculture, crop insurance design, and climate adaptation.

---

### 🧪 Parent Model Reference

This project draws inspiration from and extends upon the baseline proposed in:

**"A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction"**  
by Joshua Fan et al., Cornell University

📖 [Download GNN-RNN Paper (PDF)](./Alternative%20Approach%20Parent%20Paper.pdf)

Their GNN-RNN framework captures both spatial and temporal dependencies across counties in the U.S. and achieves state-of-the-art results in national-scale yield prediction. While their model emphasizes **geographic graph structure**, MMST-ViT explores **pixel-to-county multimodal fusion** with image tokens and transformer attention.

---

### 🔄 Ongoing Work: Region-Level MMST-ViT (Experimental Branch)

We are currently experimenting with a **region-level adaptation of MMST-ViT**, where yield predictions are grouped by **Agricultural Statistics Districts (ASDs)** instead of individual counties. This alternative pipeline aims to:

- 🌍 Reduce noise from small counties with sparse Sentinel-2 imagery
- 🔎 Emphasize broader spatial correlations (e.g., SW vs. NE Iowa)
- 🌱 Provide smoother, regionally interpretable predictions for stakeholders

Planned enhancements include:
- 🛰️ **NDVI integration** using Sentinel-2 NIR + Red bands  
- 📦 **MMST-ViT embeddings** reused in secondary classifiers  
- 🧠 **Model explainability** via attention weight visualization and risk maps

