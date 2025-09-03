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

![Sentinel-2 Bands Grid](images/sentinel_band_grid.png)

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




