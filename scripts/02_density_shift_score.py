"""
Caprine Granulosa Cell Single-Cell RNA-Seq Analysis Pipeline
Module: Topological Density Shift Score & Stratified Phenotype Differential Expression

Author: Caprine Granulosa Research Team
Repository: https://github.com/caprine-single-cell/granulosa-transcriptomics
Data Repository: https://doi.org/10.6084/m9.figshare.33472399
License: MIT
"""

from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

SEED = 42
sc.settings.seed = SEED
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_staged_adata(input_path: Path) -> sc.AnnData:
    print(f"Loading staged dataset from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    adata = sc.read_h5ad(input_path)
    print(f"Successfully loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes.")
    return adata


def calculate_density_shift_score(
    adata: sc.AnnData,
    phenotype_key: str = "fertility",
    positive_target: str = "High fertility",
    cluster_key: str = "leiden",
) -> sc.AnnData:
    """Calculates local density shift score (S_i) via KNN graph label smoothing."""
    print("\n--- Step 3: Assessing Landscape Variations via Localized Proportions ---")

    props = pd.crosstab(adata.obs[cluster_key], adata.obs[phenotype_key], normalize="columns") * 100
    props.to_csv(TABLES_DIR / "Cluster_Proportions_By_Phenotype.csv")

    if "connectivities" not in adata.obsp:
        raise KeyError("KNN connectivities matrix missing from adata.obsp['connectivities'].")

    knn_graph = adata.obsp["connectivities"]
    is_positive_target = (adata.obs[phenotype_key] == positive_target).astype(float).values

    neighbor_counts = np.asarray(knn_graph.dot(is_positive_target)).ravel()
    total_neighbors = np.asarray(knn_graph.sum(axis=1)).ravel()

    total_neighbors[total_neighbors == 0] = 1.0

    adata.obs["High_Fertility_Local_Fraction"] = neighbor_counts / total_neighbors
    adata.obs["Density_Shift_Score"] = (adata.obs["High_Fertility_Local_Fraction"] - 0.5) * 2.0

    metrics_cols = [cluster_key, "literature_cell_type", phenotype_key, "Density_Shift_Score"]
    available_cols = [col for col in metrics_cols if col in adata.obs.columns]
    
    adata.obs[available_cols].to_csv(
        TABLES_DIR / "Cell_Abundance_Density_Scores.csv", index=True
    )
    print("-> Neighborhood density landscape score (S_i) calculated successfully!")
    return adata


def run_stratified_differential_expression(
    adata: sc.AnnData,
    cell_type_key: str = "literature_cell_type",
    phenotype_key: str = "fertility",
    min_cells_per_group: int = 10,
    top_n_genes: int = 50,
) -> pd.DataFrame:
    """Performs cell-type controlled differential expression between phenotypes."""
    print("\n--- Step 4: Starting Stratified Expression Isolation Loop ---")

    if cell_type_key not in adata.obs.columns:
        print(f"Warning: '{cell_type_key}' not found in adata.obs. Skipping DE.")
        return pd.DataFrame()

    all_diff_results = []

    for cell_type in adata.obs[cell_type_key].unique():
        adata_subset = adata[adata.obs[cell_type_key] == cell_type].copy()
        group_counts = adata_subset.obs[phenotype_key].value_counts()

        if len(group_counts) < 2 or group_counts.min() < min_cells_per_group:
            print(f" -> Skipping [{cell_type}]: Insufficient cell balance across conditions.")
            continue

        print(f" -> Testing phenotype variance inside: {cell_type}")
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby=phenotype_key,
            method="wilcoxon",
            key_added="fertility_diffs",
        )

        for g in group_counts.index:
            de_df = sc.get.rank_genes_groups_df(adata_subset, group=g, key="fertility_diffs")
            de_df = de_df.head(top_n_genes)

            for _, row in de_df.iterrows():
                tx_id = row["names"]
                readable_symbol = (
                    adata.var.loc[tx_id, "gene_symbol"]
                    if "gene_symbol" in adata.var.columns and tx_id in adata.var.index
                    else tx_id
                )

                all_diff_results.append({
                    "Cell_Type_Context": cell_type,
                    "Enriched_In_Phenotype": g,
                    "Transcript_ID": tx_id,
                    "Gene_Symbol": readable_symbol,
                    "Log2FoldChange": row["logfoldchanges"],
                    "Adjusted_p_val": row["pvals_adj"],
                })

    df_de_results = pd.DataFrame(all_diff_results)
    if not df_de_results.empty:
        de_output_path = TABLES_DIR / "Stratified_Phenotype_Differential_Expression.csv"
        df_de_results.to_csv(de_output_path, index=False)
        print(f"-> Stratified differential expression results saved to: {de_output_path}")

    return df_de_results


if __name__ == "__main__":
    input_file = DATA_DIR / "staged_adata.h5ad"
    output_file = DATA_DIR / "granulosa_density_analyzed.h5ad"

    adata = load_staged_adata(input_file)
    
    adata = calculate_density_shift_score(
        adata,
        phenotype_key="fertility",
        positive_target="High fertility",
        cluster_key="leiden"
    )

    de_results = run_stratified_differential_expression(
        adata,
        cell_type_key="literature_cell_type",
        phenotype_key="fertility",
        min_cells_per_group=10
    )

    print(f"\nWriting updated dataset to {output_file}...")
    adata.write_h5ad(output_file, compression="gzip")