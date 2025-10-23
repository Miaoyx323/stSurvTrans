# data_preprocessing.py - Data preprocessing module
import os
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData

def load_single_cell_data(data_path):
    """
    Load single-cell data (supports h5ad/CSV/TSV formats)
    
    Parameters:
        data_path (str): Path to the data file (e.g., "data/scanpy_data.h5ad")
        data_format (str): Data format (h5ad=scanpy format, csv/tsv=tabular format)
        sep (str): Delimiter for tabular formats ("," for csv, "\t" for tsv)
    
    Returns:
        adata (sc.AnnData): scanpy data object (standard format for single-cell data)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")
    
    # 2. Automatically determine format and read data
    adata = None
    # Case 1: Path is a directory → read Visium spatial transcriptomics data
    if os.path.isdir(data_path):
        print(f"Directory path detected, reading data in Visium format: {data_path}")
        adata = sc.read_visium(data_path)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
    
    # Case 2: Path is a file → determine format by suffix
    elif os.path.isfile(data_path):
        # Get file extension (case-insensitive to handle uppercase suffixes like ".H5AD", ".CSV")
        file_ext = os.path.splitext(data_path)[-1].lower()
        
        # Subcase 2.1: h5ad format
        if file_ext == ".h5ad":
            print(f".h5ad file detected, reading data in h5ad format: {data_path}")
            adata = sc.read_h5ad(data_path)
            adata.obs_names_make_unique()
            adata.var_names_make_unique()
        
        # Subcase 2.2: csv format (default delimiter is comma, conforming to CSV standard)
        elif file_ext == ".csv":
            print(f".csv file detected, reading data in tabular format: {data_path}")
            # Tabular format defaults to "rows=cells, columns=genes", using the first column as index (cell names)
            df = pd.read_csv(data_path, index_col=0)
            # Convert to AnnData object (X=expression matrix, obs=cell information, var=gene information)
            adata = sc.AnnData(
                X=df.values,
                obs=pd.DataFrame(index=df.index),
                var=pd.DataFrame(index=df.columns)
            )
        
        # Subcase 2.3: Unsupported file format
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext} (only .h5ad/.csv files or Visium directories are supported)\n"
                f"Input file path: {data_path}"
            )
    
    # 3. Final data validation
    if adata is None:
        raise RuntimeError(f"Data reading failed, unknown path type: {data_path}")
    
    print(f"Successfully loaded data: {adata.n_obs} cells, {adata.n_vars} genes.")
    return adata

def clean_single_cell_data(
        adata, 
        min_genes_per_cell=200, 
        min_cells_per_gene=3, 
        filter_mt=True,
        mt_prefix="MT-",
        pct_counts_mt_threshold=20
):
    """
    Clean single-cell data (filter low-quality cells/genes, mark mitochondrial genes)
    
    Parameters:
        adata (sc.AnnData): Original scanpy data object
        min_genes_per_cell (int): Minimum number of genes per cell (filters empty cells)
        min_cells_per_gene (int): Minimum number of cells each gene appears in (filters low-expression genes)
        filter_mt (bool): Whether to filter mitochondrial genes
    
    Returns:
        adata_clean (sc.AnnData): Cleaned scanpy data object
    """
    adata_clean = adata.copy()
    
    # 1. Calculate mitochondrial gene proportion (mark mitochondrial genes: gene names starting with "MT-", may need adjustment based on species)
    if filter_mt:
        adata_clean.var["mt"] = adata_clean.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(adata_clean, qc_vars=["mt"], inplace=True)
    
    # 2. Filter low-quality cells (too few genes/high mitochondrial proportion)
    sc.pp.filter_cells(adata_clean, min_genes=min_genes_per_cell)
    if filter_mt:
        adata_clean = adata_clean[adata_clean.obs["pct_counts_mt"] < pct_counts_mt_threshold, :]
    
    # 3. Filter low-expression genes
    sc.pp.filter_genes(adata_clean, min_cells=min_cells_per_gene)
    
    # Output comparison before and after cleaning
    print(f"Data cleaning completed:")
    print(f"  - Number of cells: {adata.n_obs} → {adata_clean.n_obs}")
    print(f"  - Number of genes: {adata.n_vars} → {adata_clean.n_vars}")
    return adata_clean

def filter_genes_single(
        adata: AnnData,
        cell_count_cutoff=15, 
        cell_percentage_cutoff2=0.05, 
        nonz_mean_cutoff=1.12
):
    adata.var['n_cells'] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var['nonz_mean'] = np.array(adata.X.sum(0)).flatten() / adata.var['n_cells']
    
    gene_selection = (
        np.array(adata.var['n_cells'] > cell_count_cutoff) &
        np.array(adata.var['n_cells'] > adata.n_obs * cell_percentage_cutoff2) &
        np.array(adata.var['nonz_mean'] > nonz_mean_cutoff)
    )

    gene_selection = adata.var_names[gene_selection]

    return gene_selection

def preprocess_data(
        adata_st: AnnData,
        adata_bulk: AnnData,
        cell_count_cutoff=15, 
        cell_percentage_cutoff2=0.05, 
        nonz_mean_cutoff=1.12
):
    gene_selection_st = filter_genes_single(
        adata=adata_st,
        cell_count_cutoff=cell_count_cutoff,
        cell_percentage_cutoff2=cell_percentage_cutoff2,
        nonz_mean_cutoff=nonz_mean_cutoff
    )

    gene_selection = list(set(adata_bulk.var_names).intersection(set(gene_selection_st)))
    adata_st = adata_st[:, gene_selection].copy()
    adata_bulk = adata_bulk[:, gene_selection].copy()

    print(f"After preprocessing, a total of {len(gene_selection)} genes are used for subsequent analysis.")

    return adata_st, adata_bulk