"""
Caprine Granulosa Cell Single-Cell RNA-Seq Analysis Pipeline
Module: scVI Integration, Latent Embedding, and Balanced Score Clustering Optimization

Author: Caprine Granulosa Research Team
Repository: https://github.com/caprine-single-cell/granulosa-transcriptomics
Data Repository: https://doi.org/10.6084/m9.figshare.33472399
License: MIT
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. GLOBAL REPRODUCIBILITY & ENVIRONMENT CONFIGURATION
# ==============================================================================
SEED = 42
scvi.settings.seed = SEED
sc.settings.seed = SEED
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DATA LOADING & FEATURE SELECTION
# ==============================================================================
def load_and_preprocess_data(file_path):
    """Loads input AnnData object, preserves raw counts, and selects 3,000 HVTs."""
    print(f"Loading pre-processed dataset from: {file_path}")
    adata = sc.read_h5ad(file_path)
    print(f"Dataset successfully loaded: {adata.n_obs} cells × {adata.n_vars} genes.")

    # Preserve raw counts layer for scVI
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    print("Selecting 3,000 Highly Variable Transcripts (HVTs)...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor="seurat_v3",
        batch_key="sample_id",
        layer="counts",
        subset=True
    )
    return adata


# ==============================================================================
# 3. DEEP GENERATIVE MODELING VIA scVI
# ==============================================================================
def run_scvi_latent_embedding(adata, n_latent=30, max_epochs=250):
    """Initializes and trains scVI VAE using raw counts layer."""
    print("Setting up scVI model architecture...")
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="sample_id"
    )

    vae = scvi.model.SCVI(
        adata,
        n_hidden=128,
        n_latent=n_latent,
        n_layers=2,
        dropout_rate=0.1,
        dispersion="gene",
        gene_likelihood="zinb"
    )

    print(f"Training scVI neural network across max {max_epochs} epochs...")
    vae.train(max_epochs=max_epochs, early_stopping=True, frequency=1)

    # Plot ELBO convergence
    plt.figure(figsize=(6, 4))
    plt.plot(vae.history["elbo_train"], label="Train ELBO", color="darkblue")
    plt.title("scVI Training Convergence Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (ELBO)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scvi_training_convergence.png", dpi=300)
    plt.close()

    adata.obsm["X_scVI"] = vae.get_latent_representation()
    
    print("Building KNN connectivity graph (k=30) in scVI latent space...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30)
    
    return adata, vae


# ==============================================================================
# 4. LEIDEN CLUSTERING OPTIMIZATION VIA BALANCED SCORE
# ==============================================================================
def optimize_clustering_balanced_score(
    adata,
    resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    min_cells_required=20,
    phenotype_key="fertility"
):
    latent_coords = adata.obsm["X_scVI"]
    results_list = []

    print(f"\nExecuting Leiden resolution sweep (Min size constraint: {min_cells_required} cells)...")
    print("-" * 100)

    for res in resolutions:
        cluster_key = f"leiden_res_{res:.2f}"
        sc.tl.leiden(adata, resolution=res, key_added=cluster_key, random_state=SEED)
        labels = adata.obs[cluster_key].astype(str)
        
        cluster_counts = adata.obs[cluster_key].value_counts()
        smallest_cluster_size = cluster_counts.min()
        num_clusters = len(cluster_counts)
        is_valid = smallest_cluster_size >= min_cells_required

        if adata.n_obs > 10000:
            idx = np.random.choice(adata.n_obs, size=10000, replace=False)
            raw_sil = silhouette_score(latent_coords[idx], labels[idx])
        else:
            raw_sil = silhouette_score(latent_coords, labels)
            
        norm_sil = (raw_sil + 1.0) / 2.0

        crosstab = pd.crosstab(adata.obs[cluster_key], adata.obs[phenotype_key], normalize="index")
        max_purity = crosstab.max(axis=1).mean()

        balanced_score = (norm_sil * max_purity) if is_valid else 0.0

        status = "PASS" if is_valid else f"FAIL (Min cluster: {smallest_cluster_size} cells)"
        print(
            f"Res: {res:<4.1f} | Clusters: {num_clusters:<2} | "
            f"Raw Sil: {raw_sil:.3f} | Norm Sil: {norm_sil:.3f} | "
            f"Max Purity: {max_purity:.1%} | Balanced: {balanced_score:.3f} | {status}"
        )

        results_list.append({
            "Resolution": res,
            "Num_Clusters": num_clusters,
            "Raw_Silhouette": raw_sil,
            "Normalized_Silhouette": norm_sil,
            "Max_Phenotype_Purity": max_purity,
            "Smallest_Cluster_Size": smallest_cluster_size,
            "Is_Valid_Size": is_valid,
            "Balanced_Score": balanced_score,
            "Cluster_Key": cluster_key
        })

    df_results = pd.DataFrame(results_list)
    output_table = TABLES_DIR / "clustering_optimization_balanced_score.csv"
    df_results.to_csv(output_table, index=False)
    print("-" * 100)

    df_valid = df_results[df_results["Is_Valid_Size"] == True]
    if df_valid.empty:
        raise ValueError("No resolution met the minimum cluster size requirement!")
        
    best_row = df_valid.loc[df_valid["Balanced_Score"].idxmax()]
    winning_res = best_row["Resolution"]
    winning_key = best_row["Cluster_Key"]

    print("\nOPTIMAL CLUSTER CONFIGURATION SELECTED:")
    print(f"  -> Best Resolution: {winning_res}")
    print(f"  -> Total Clusters: {int(best_row['Num_Clusters'])}")

    adata.obs["leiden"] = adata.obs[winning_key]
    return adata, df_results


# ==============================================================================
# 5. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    input_h5ad = DATA_DIR / "merged_goat_fertility.h5ad"

    adata = load_and_preprocess_data(input_h5ad)
    adata, vae_model = run_scvi_latent_embedding(adata, n_latent=30, max_epochs=250)
    adata, optimization_df = optimize_clustering_balanced_score(
        adata,
        resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        min_cells_required=20
    )

    output_h5ad = DATA_DIR / "staged_adata.h5ad"
    adata.write_h5ad(output_h5ad, compression="gzip")
    print(f"\nPipeline completed! Embedded AnnData saved to {output_h5ad}")