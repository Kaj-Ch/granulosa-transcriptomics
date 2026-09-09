# Caprine Granulosa Cell scRNA-Seq Analysis Pipeline

Computational workflow for single-cell transcriptomics analysis of caprine granulosa cells across fertility phenotypes.

## 📌 Repository Structure
- `01_scvi_latent_and_clustering.py`: Deep generative model integration via scVI and Leiden resolution optimization.
- `02_density_shift_score.py`: Localized KNN graph label smoothing for Density Shift Score ($S_i$) computation and stratified DE testing.
- `requirements.txt`: Python package dependencies.

## 📦 Data Availability
Due to GitHub file size limits, processed single-cell AnnData (`.h5ad`) objects are archived on Figshare:
- **Datasets:** `staged_adata.h5ad` (~93 MB) and `merged_goat_fertility.h5ad` (~162 MB)
- **Figshare DOI:** [https://doi.org/10.6084/m9.figshare.33472399](https://doi.org/10.6084/m9.figshare.33472399)

## 🚀 Environment Setup & Execution

### 1. Setup Virtual Environment
```bash
git clone [https://github.com/caprine-single-cell/granulosa-transcriptomics.git](https://github.com/caprine-single-cell/granulosa-transcriptomics.git)
cd granulosa-transcriptomics

python -m venv scvi_env
source scvi_env/bin/activate  # On Windows: scvi_env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt