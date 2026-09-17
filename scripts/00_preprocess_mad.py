#!/usr/bin/env python3
"""
00_preprocess_mad.py
--------------------
Preprocessing script for goat granulosa cell transcriptomics.
Performs quality control using an adaptive Median Absolute Deviation (MAD) framework,
logs step-by-step filtering statistics per sample, exports a detailed CSV report,
and writes the clean AnnData object to disk.
"""

import os
import gzip
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.io import mmread
import anndata as ad

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def load_salmon_mtx_gz(sample_path):
    """Loads Salmon Alevin quantification outputs into an AnnData object."""
    alevin_dir = os.path.join(sample_path, "alevin")
    matrix_file = os.path.join(alevin_dir, "quants_mat.mtx.gz")
    rows_file = os.path.join(alevin_dir, "quants_mat_rows.txt")
    cols_file = os.path.join(alevin_dir, "quants_mat_cols.txt")
    
    with gzip.open(matrix_file, 'rt') as f:
        X = mmread(f).tocsr()
    
    barcodes = pd.read_csv(rows_file, header=None)[0].astype(str).values
    genes = pd.read_csv(cols_file, header=None)[0].astype(str).values
    
    return sc.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=genes))


def calc_mad_thresholds(data, n_mads=3, side="both"):
    """Calculates lower and/or upper cutoff thresholds based on Median Absolute Deviation."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    lower_limit = median - (n_mads * mad)
    upper_limit = median + (n_mads * mad)
    
    if side == "upper": 
        return upper_limit
    if side == "lower": 
        return lower_limit
    return lower_limit, upper_limit


def get_expressed_genes_count(adata_sub):
    """Counts non-zero expressed genes within a specific sample subset."""
    if hasattr(adata_sub.X, "tocsr"):
        return int((adata_sub.X.getnnz(axis=0) > 0).sum())
    else:
        return int((np.asarray(adata_sub.X > 0).sum(axis=0) > 0).sum())


# ==============================================================================
# 2. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    qc_report_list = []
    
    print("=" * 80)
    print("STARTING STEP 00: QUALITY CONTROL & MAD PREPROCESSING")
    print("=" * 80)
    
    # --- Step A: Read Native Salmon Datasets ---
    print("\n[1/5] Loading Salmon Alevin quantifications...")
    adata1 = load_salmon_mtx_gz("./output/SRR9945436")
    adata2 = load_salmon_mtx_gz("./output/SRR9945437")
    
    # --- Step B: Merge Datasets and Add Metadata ---
    print("[2/5] Merging samples and mapping metadata...")
    adata = ad.concat({"SRR9945436": adata1, "SRR9945437": adata2}, label="sample_id", join="inner")
    adata.obs_names = adata.obs_names + "_" + adata.obs["sample_id"].astype(str)

    metadata = pd.read_csv("SraRunTable.csv")[["Run", "fertility"]]
    fertility_map = dict(zip(metadata["Run"], metadata["fertility"]))
    adata.obs["fertility"] = adata.obs["sample_id"].map(fertility_map)

    # Flag Mitochondrial Genes using GTF
    gtf_path = "/mnt/vetdata2/Research/20260502_Goat_model/lib/Capra.gtf"
    mt_transcript_ids = set()
    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("#"): 
                continue
            fields = line.split("\t")
            if fields[0] in ["MT", "mt", "chrM", "M"]:
                attributes = fields[8]
                if 'transcript_id "' in attributes:
                    mt_transcript_ids.add(attributes.split('transcript_id "')[1].split('"')[0])

    adata.var['mt'] = adata.var_names.isin(mt_transcript_ids)

    # Compute Initial QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Log Stage 1: Raw Unfiltered Baseline
    for sample in adata.obs['sample_id'].unique():
        sub = adata[adata.obs['sample_id'] == sample]
        qc_report_list.append({
            'Stage': '1. Raw Unfiltered',
            'Sample_ID': sample,
            'Fertility': sub.obs['fertility'].iloc[0],
            'Applied_Cutoffs': 'None (Raw Data)',
            'Cell_Count': sub.n_obs,
            'Cells_Removed_This_Step': 0,
            'Pct_Cells_Removed': '0.0%',
            'Expressed_Genes': get_expressed_genes_count(sub),
            'Total_Matrix_Genes': sub.n_vars,
            'Genes_Removed_This_Step': 0,
            'Mean_UMIs': round(sub.obs['total_counts'].mean(), 2),
            'Mean_MT_Pct': round(sub.obs['pct_counts_mt'].mean(), 2)
        })

    # --- Step C: Compute Adaptive MAD Thresholds & Filter Cells ---
    print("[3/5] Computing adaptive MAD statistical cutoffs...")
    min_umi, max_umi = calc_mad_thresholds(adata.obs['total_counts'], n_mads=3, side="both")
    min_transcripts, max_transcripts = calc_mad_thresholds(adata.obs['n_genes_by_counts'], n_mads=3, side="both")
    min_transcripts = max(min_transcripts, 1000)  # Enforce strict baseline
    max_mt = calc_mad_thresholds(adata.obs['pct_counts_mt'], n_mads=5, side="upper")

    cutoff_str = f"UMIs: [{min_umi:.0f}, {max_umi:.0f}] | Genes: [{min_transcripts:.0f}, {max_transcripts:.0f}] | Max MT: {max_mt:.2f}%"
    print(f"    --> Thresholds applied: {cutoff_str}")

    # Save raw state in .raw slot
    adata.raw = adata

    # Filter cells
    adata_cell_filtered = adata[
        (adata.obs['total_counts'] >= min_umi) & 
        (adata.obs['total_counts'] <= max_umi) &
        (adata.obs['n_genes_by_counts'] >= min_transcripts) & 
        (adata.obs['n_genes_by_counts'] <= max_transcripts) & 
        (adata.obs['pct_counts_mt'] <= max_mt), 
        :
    ].copy()

    # Log Stage 2: Post Cell-Filtering
    stage1_df = pd.DataFrame(qc_report_list)
    for sample in adata_cell_filtered.obs['sample_id'].unique():
        sub = adata_cell_filtered[adata_cell_filtered.obs['sample_id'] == sample]
        prev_cells = stage1_df[stage1_df['Sample_ID'] == sample]['Cell_Count'].values[0]
        cells_removed = prev_cells - sub.n_obs
        pct_removed = (cells_removed / prev_cells) * 100 if prev_cells > 0 else 0
        
        qc_report_list.append({
            'Stage': '2. Post Cell-Filtering',
            'Sample_ID': sample,
            'Fertility': sub.obs['fertility'].iloc[0],
            'Applied_Cutoffs': cutoff_str,
            'Cell_Count': sub.n_obs,
            'Cells_Removed_This_Step': cells_removed,
            'Pct_Cells_Removed': f"{pct_removed:.2f}%",
            'Expressed_Genes': get_expressed_genes_count(sub),
            'Total_Matrix_Genes': sub.n_vars,
            'Genes_Removed_This_Step': 0,
            'Mean_UMIs': round(sub.obs['total_counts'].mean(), 2),
            'Mean_MT_Pct': round(sub.obs['pct_counts_mt'].mean(), 2)
        })

    # --- Step D: Filter Low-Expressed Genes ---
    print("[4/5] Removing unexpressed/lowly-expressed genes (min_cells < 3)...")
    adata_final = adata_cell_filtered.copy()
    sc.pp.filter_genes(adata_final, min_cells=3)

    # Log Stage 3: Post Gene-Filtering
    stage2_records = [r for r in qc_report_list if r['Stage'] == '2. Post Cell-Filtering']
    for sample in adata_final.obs['sample_id'].unique():
        sub = adata_final[adata_final.obs['sample_id'] == sample]
        prev_record = [r for r in stage2_records if r['Sample_ID'] == sample][0]
        
        prev_matrix_genes = prev_record['Total_Matrix_Genes']
        genes_removed = prev_matrix_genes - sub.n_vars
        
        qc_report_list.append({
            'Stage': '3. Post Gene-Filtering (Final)',
            'Sample_ID': sample,
            'Fertility': sub.obs['fertility'].iloc[0],
            'Applied_Cutoffs': 'min_cells >= 3',
            'Cell_Count': sub.n_obs,
            'Cells_Removed_This_Step': 0,
            'Pct_Cells_Removed': '0.0%',
            'Expressed_Genes': get_expressed_genes_count(sub),
            'Total_Matrix_Genes': sub.n_vars,
            'Genes_Removed_This_Step': genes_removed,
            'Mean_UMIs': round(sub.obs['total_counts'].mean(), 2),
            'Mean_MT_Pct': round(sub.obs['pct_counts_mt'].mean(), 2)
        })

    # --- Step E: Save Report & Output AnnData ---
    print("[5/5] Generating final summary report and writing files to disk...")
    qc_report_df = pd.DataFrame(qc_report_list)

    # Print Summary to Console
    print("\n" + "=" * 100)
    print("PREPROCESSING & QC REPORT SUMMARY")
    print("=" * 100)
    print(qc_report_df[['Stage', 'Sample_ID', 'Cell_Count', 'Cells_Removed_This_Step', 'Pct_Cells_Removed', 'Total_Matrix_Genes', 'Genes_Removed_This_Step']].to_string(index=False))
    print("=" * 100)

    # Write CSV Report
    report_csv = "sample_qc_filtering_report.csv"
    qc_report_df.to_csv(report_csv, index=False)
    print(f"-> Full report saved to: {report_csv}")

    # Write Processed AnnData
    output_h5ad = "merged_goat_fertility.h5ad"
    adata_final.write_h5ad(output_h5ad, compression="gzip")
    print(f"-> Filtered AnnData saved to: {output_h5ad}")
    print("\n[SUCCESS] Pipeline step 00 completed successfully.\n")

if __name__ == "__main__":
    main()