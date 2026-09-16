import os
import gzip
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from scipy.io import mmread

def load_salmon_mtx_gz(sample_path: str) -> sc.AnnData:
    """Load Salmon Alevin MTX files into an AnnData object."""
    alevin_dir = os.path.join(sample_path, "alevin")
    matrix_file = os.path.join(alevin_dir, "quants_mat.mtx.gz")
    rows_file = os.path.join(alevin_dir, "quants_mat_rows.txt")
    cols_file = os.path.join(alevin_dir, "quants_mat_cols.txt")
    
    with gzip.open(matrix_file, 'rt') as f:
        X = mmread(f).tocsr()
    
    barcodes = pd.read_csv(rows_file, header=None)[0].astype(str).values
    genes = pd.read_csv(cols_file, header=None)[0].astype(str).values
    
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=genes)
    )
    return adata


def calculate_mad_outliers(adata: sc.AnnData, metric: str, nmads: float = 5.0) -> pd.Series:
    """
    Calculate outlier status for a metric using Median Absolute Deviation (MAD).
    
    Parameters:
        adata: AnnData object.
        metric: Column name in `adata.obs`.
        nmads: Number of MADs away from median to threshold.
    """
    data = adata.obs[metric]
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Calculate upper boundary (and lower if applicable)
    upper_bound = median + (nmads * mad)
    lower_bound = median - (nmads * mad)
    
    # Flag outliers (e.g., too high MT% or too low/high read counts)
    outliers = (data > upper_bound) | (data < lower_bound)
    return outliers


def main():
    # 1. Load Data
    sample_paths = {
        "SRR9945436": "./output/SRR9945436",
        "SRR9945437": "./output/SRR9945437"
    }
    
    adatas = {sample_id: load_salmon_mtx_gz(path) for sample_id, path in sample_paths.items()}
    
    # 2. Concatenate AnnData objects
    adata = ad.concat(adatas, label="sample_id", join="inner")
    adata.obs_names = adata.obs_names + "_" + adata.obs["sample_id"].astype(str)
    
    # 3. Add Metadata
    metadata_path = "SraRunTable.csv"
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)[["Run", "fertility"]]
        fertility_map = dict(zip(metadata["Run"], metadata["fertility"]))
        adata.obs["fertility"] = adata.obs["sample_id"].map(fertility_map)

    # 4. Identify Mitochondrial Genes from GTF
    gtf_path = "/mnt/vetdata2/Research/20260502_Goat_model/lib/Capra.gtf"
    mt_transcript_ids = set()
    
    if os.path.exists(gtf_path):
        with open(gtf_path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                fields = line.split("\t")
                if fields[0] in ["MT", "mt", "chrM", "M"]:
                    attributes = fields[8]
                    if 'transcript_id "' in attributes:
                        t_id = attributes.split('transcript_id "')[1].split('"')[0]
                        mt_transcript_ids.add(t_id)

    adata.var['mt'] = adata.var_names.isin(mt_transcript_ids)

    # 5. Compute Quality Control Metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # 6. Apply MAD-based Filtering
    # Detect outliers per sample to account for batch variance
    adata.obs['is_outlier_counts'] = False
    adata.obs['is_outlier_transcripts'] = False
    adata.obs['is_outlier_mt'] = False

    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        sample_adata = adata[sample_mask]

        adata.obs.loc[sample_mask, 'is_outlier_counts'] = calculate_mad_outliers(sample_adata, 'total_counts', nmads=5)
        adata.obs.loc[sample_mask, 'is_outlier_transcripts'] = calculate_mad_outliers(sample_adata, 'n_genes_by_counts', nmads=5)
        adata.obs.loc[sample_mask, 'is_outlier_mt'] = calculate_mad_outliers(sample_adata, 'pct_counts_mt', nmads=3)

    # Combine outlier markers
    adata.obs['outlier'] = (
        adata.obs['is_outlier_counts'] | 
        adata.obs['is_outlier_transcripts'] | 
        adata.obs['is_outlier_mt']
    )

    print(f"Total cells before filtering: {adata.n_obs}")
    filtered_adata = adata[~adata.obs['outlier']].copy()
    print(f"Total cells after MAD filtering: {filtered_adata.n_obs}")

    # 7. Save Processed Object
    filtered_adata.write("processed_adata.h5ad")


if __name__ == "__main__":
    main()