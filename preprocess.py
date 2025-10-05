#!/usr/bin/env python
# coding: utf-8

################################
###           HVG            ###
################################

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import harmonypy as hm

print("Scanpy version:", sc.__version__)

# ------------------------
# HVG Preprocessing Function
# ------------------------
def hvg_selection_and_pooling(exp_paths, n_top_genes=1000):
    # Input: List of paths to expression matrices
    # Output: Filtered HVG expression matrices for each dataset

    all_genes = None
    hvg_bools = []

    # Iterate through each dataset
    for d in exp_paths:
        adata = sio.mmread(d).toarray()
        print("Original matrix size:", adata.shape)

        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        if all_genes is None:
            all_genes = adata.var_names
        else:
            # Keep only genes common across datasets
            all_genes = all_genes.intersection(adata.var_names)

        # Normalize and log-transform
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # Save HVG boolean vector
        hvg_bools.append(adata.var['highly_variable'])

    # HVG union
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print("Current HVG count:", sum(hvg_union), "Next HVG count:", sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]

    print("Final HVG count:", hvg_union.sum())

    # Save HVG union
    np.save("../GSE240429_data/data/filtered_expression_matrices/hvg_union.npy", hvg_union.to_numpy())

    # Filter expression matrices
    filtered_exp_mtxs = []
    for d in exp_paths:
        adata = sio.mmread(d).toarray()
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        filtered_exp_mtxs.append(adata[:, hvg_union].X)

    return filtered_exp_mtxs


# Data paths
exp_paths = [
    "../GSE240429_data/data/filtered_expression_matrices/1/matrix.mtx",
    "../GSE240429_data/data/filtered_expression_matrices/2/matrix.mtx",
    "../GSE240429_data/data/filtered_expression_matrices/3/matrix.mtx",
    "../GSE240429_data/data/filtered_expression_matrices/4/matrix.mtx"
]

# Process and save HVG
filtered_mtx = hvg_selection_and_pooling(exp_paths)
for i in range(len(filtered_mtx)):
    np.save(f"../GSE240429_data/data/filtered_expression_matrices/{i+1}/hvg_matrix.npy", filtered_mtx[i].T)


# ------------------------
# HVG Harmony Batch Correction
# ------------------------
print("\n=== HVG Harmony Batch Correction ===")
d1 = np.load("../GSE240429_data/data/filtered_expression_matrices/1/hvg_matrix.npy")
d2 = np.load("../GSE240429_data/data/filtered_expression_matrices/2/hvg_matrix.npy")
d3 = np.load("../GSE240429_data/data/filtered_expression_matrices/3/hvg_matrix.npy")
d4 = np.load("../GSE240429_data/data/filtered_expression_matrices/4/hvg_matrix.npy")

# Concatenate datasets
d = np.concatenate((d1.T, d2.T, d3.T, d4.T), axis=0)
print("Concatenated matrix size:", d.shape)

data_sizes = [d1.shape[1], d2.shape[1], d3.shape[1], d4.shape[1]]
batch_labels = np.concatenate([
    np.zeros(data_sizes[0]),
    np.ones(data_sizes[1]),
    np.ones(data_sizes[2]) * 2,
    np.ones(data_sizes[3]) * 3
]).astype(str)

df = pd.DataFrame(batch_labels, columns=["dataset"])

# Run Harmony for batch correction
harmony = hm.run_harmony(d, meta_data=df, vars_use=["dataset"])
harmony_corrected = harmony.Z_corr.T

# Split back into subsets
d1 = harmony_corrected[:data_sizes[0]]
d2 = harmony_corrected[data_sizes[0]:data_sizes[0]+data_sizes[1]]
d3 = harmony_corrected[data_sizes[0]+data_sizes[1]:data_sizes[0]+data_sizes[1]+data_sizes[2]]
d4 = harmony_corrected[data_sizes[0]+data_sizes[1]+data_sizes[2]:]

# Save
np.save("../GSE240429_data/data/filtered_expression_matrices/1/harmony_hvg_matrix.npy", d1.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/2/harmony_hvg_matrix.npy", d2.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/3/harmony_hvg_matrix.npy", d3.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/4/harmony_hvg_matrix.npy", d4.T)


################################
###           HEG            ###
################################

def heg_selection_and_pooling(exp_paths, n_top_genes=3200):
    # Function to select and pool high-expressed genes (HEG) from multiple datasets
    all_genes = None
    adata_list = []
    heg_bools = []

    # Iterate through each dataset
    for d in exp_paths:
        adata = sio.mmread(d).toarray()
        print("Original matrix size:", adata.shape)

        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Remove zero genes and cells
        non_zero_genes = (adata.X.sum(axis=0) != 0)
        adata = adata[:, non_zero_genes]
        non_zero_cells = (adata.X.sum(axis=1) != 0)
        adata = adata[non_zero_cells, :]

        if all_genes is None:
            all_genes = set(adata.var_names)
        else:
            all_genes &= set(adata.var_names)

        adata_list.append(adata)

    all_genes = sorted(list(all_genes))

    # Align genes
    filtered_adata_list = []
    for adata in adata_list:
        adata = adata[:, all_genes]
        filtered_adata_list.append(adata)
    adata_list = filtered_adata_list

    # Select HEG
    for adata in adata_list:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        avg_exp = adata.X.mean(axis=0)

        heg_indices = np.argsort(avg_exp)[::-1][:n_top_genes]
        heg_mask = np.zeros(len(all_genes), dtype=bool)
        heg_mask[heg_indices] = True
        heg_bools.append(heg_mask)

    # HEG union
    heg_union = heg_bools[0]
    for i in range(1, len(heg_bools)):
        print("Current HEG count:", sum(heg_union), "Next HEG count:", sum(heg_bools[i]))
        heg_union = heg_union | heg_bools[i]

    print("Final HEG count:", heg_union.sum())
    np.save("../GSE240429_data/data/filtered_expression_matrices/heg_union.npy", heg_union)

    # Filter expression matrices
    filtered_exp_mtxs = []
    for adata in adata_list:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        filtered_expression = adata[:, heg_union].X
        non_zero_rows = (filtered_expression.sum(axis=1) != 0)
        filtered_exp_mtxs.append(filtered_expression[non_zero_rows, :])

    return filtered_exp_mtxs


# HEG processing and saving
filtered_mtx = heg_selection_and_pooling(exp_paths)
for i in range(len(filtered_mtx)):
    np.save(f"../GSE240429_data/data/filtered_expression_matrices/{i+1}/heg_matrix.npy", filtered_mtx[i].T)


# ------------------------
# HEG Harmony Batch Correction
# ------------------------
print("\n=== HEG Harmony Batch Correction ===")
d1 = np.load("../GSE240429_data/data/filtered_expression_matrices/1/heg_matrix.npy")
d2 = np.load("../GSE240429_data/data/filtered_expression_matrices/2/heg_matrix.npy")
d3 = np.load("../GSE240429_data/data/filtered_expression_matrices/3/heg_matrix.npy")
d4 = np.load("../GSE240429_data/data/filtered_expression_matrices/4/heg_matrix.npy")

# Concatenate datasets
d = np.concatenate((d1.T, d2.T, d3.T, d4.T), axis=0)
print("Concatenated matrix size:", d.shape)

data_sizes = [d1.shape[1], d2.shape[1], d3.shape[1], d4.shape[1]]
batch_labels = np.concatenate([
    np.zeros(data_sizes[0]),
    np.ones(data_sizes[1]),
    np.ones(data_sizes[2]) * 2,
    np.ones(data_sizes[3]) * 3
]).astype(str)

df = pd.DataFrame(batch_labels, columns=["dataset"])

# Run Harmony for batch correction
harmony = hm.run_harmony(d, meta_data=df, vars_use=["dataset"])
harmony_corrected = harmony.Z_corr.T

# Split back into subsets
d1 = harmony_corrected[:data_sizes[0]]
d2 = harmony_corrected[data_sizes[0]:data_sizes[0]+data_sizes[1]]
d3 = harmony_corrected[data_sizes[0]+data_sizes[1]:data_sizes[0]+data_sizes[1]+data_sizes[2]]
d4 = harmony_corrected[data_sizes[0]+data_sizes[1]+data_sizes[2]:]

# Save corrected data
np.save("../GSE240429_data/data/filtered_expression_matrices/1/harmony_heg_matrix.npy", d1.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/2/harmony_heg_matrix.npy", d2.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/3/harmony_heg_matrix.npy", d3.T)
np.save("../GSE240429_data/data/filtered_expression_matrices/4/harmony_heg_matrix.npy", d4.T)
