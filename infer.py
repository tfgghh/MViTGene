import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

import gseapy as gp

from models import CLIPModel, WeightingNetwork
from dataset import CLIPDataset

print(sc.__version__)


def visualize_umap_clusters(
    expr_matrix,
    preprocess=True,
    normalize_and_log=True,
    batch_idx=None,
    n_neighbors=150,
    n_top_genes=1024,
    legend_loc='on data',
    show=False,
    save=False,
    save_name='umap_clusters.png'
):
    if preprocess:
        adata = sc.AnnData(X=expr_matrix, dtype=expr_matrix.dtype)
        if batch_idx is not None:
            adata.obs['batch_idx'] = batch_idx
        if normalize_and_log:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        print("n_top_genes: ", adata.var['highly_variable'].sum())
    else:
        adata = sc.AnnData(X=expr_matrix, dtype=expr_matrix.dtype)
        if batch_idx is not None:
            adata.obs['batch_idx'] = batch_idx

    sc.pp.pca(adata, n_comps=50, use_highly_variable=preprocess)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50)

    sc.tl.umap(adata)
    print("Running Leiden clustering")
    sc.tl.leiden(adata)
    print("n_clusters: ", adata.obs['leiden'].nunique())
    print("Plotting UMAP clusters")

    if batch_idx is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc.pl.umap(adata, color='leiden', ax=ax, show=show, legend_loc=legend_loc)
        if save:
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
        return adata
    else:
        fig, axs = plt.subplots(ncols=2, figsize=(10,5))
        sc.pl.umap(adata, color='leiden', ax=axs[0], show=False, legend_loc=legend_loc)
        sc.pl.umap(adata, color='batch_idx', ax=axs[1], show=False, legend_loc=legend_loc)
        if save:
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
        return adata


def build_loaders_inference():
    print("Building loaders for inference")

    full_expr3 = np.load("./GSE240429_data/data/filtered_expression_matrices/3/expression.npy")
    gene_var = np.var(full_expr3, axis=0)
    vocab_size = 2000
    vocab_gene_indices = np.argsort(-gene_var)[:vocab_size]

    top_k = 128

    dataset1 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_A1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/1/expression.npy",
        vocab_gene_indices=vocab_gene_indices,
        top_k=top_k,
        augment=False
    )

    dataset2 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_B1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/2/expression.npy",
        vocab_gene_indices=vocab_gene_indices,
        top_k=top_k,
        augment=False
    )

    dataset3 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_C1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/3/expression.npy",
        vocab_gene_indices=vocab_gene_indices,
        top_k=top_k,
        augment=False
    )

    dataset4 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_D1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/4/expression.npy",
        vocab_gene_indices=vocab_gene_indices,
        top_k=top_k,
        augment=False
    )

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    print("Finished building inference loader")
    return test_loader


def get_image_embeddings(model_path, model):
    test_loader = build_loaders_inference()
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

    return torch.cat(test_image_embeddings)


def get_spot_embeddings(model_path, model):
    test_loader = build_loaders_inference()

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            spot_features = model.spot_projection(batch["reduced_expression"].cuda())
            spot_embeddings.append(spot_features)
    return torch.cat(spot_embeddings, dim=0)


def find_matches(spot_emb, query_emb, top_k=50):
    spot_emb = F.normalize(torch.tensor(spot_emb), p=2, dim=1).float()
    query_emb = F.normalize(torch.tensor(query_emb), p=2, dim=1).float()

    cos_sim = query_emb @ spot_emb.T
    euc_dist = torch.cdist(query_emb, spot_emb)
    man_dist = torch.cdist(query_emb, spot_emb, p=1)

    euc_dist = (euc_dist - euc_dist.min()) / (euc_dist.max() - euc_dist.min() + 1e-8)
    man_dist = (man_dist - man_dist.min()) / (man_dist.max() - man_dist.min() + 1e-8)

    weight_net = WeightingNetwork()
    weights = weight_net(cos_sim, euc_dist, man_dist)

    weighted_sim = cos_sim * weights

    _, indices = torch.topk(weighted_sim, k=top_k, dim=1)
    return indices.cpu().numpy()


# ===========================
# Main execution (load → match → compute → save)
# ===========================
datasize = [2378, 2349, 2277, 2265]
model_path = "clip/best.pt"
save_path = "result/embeddings/"   # 统一保存路径
model = CLIPModel(num_genes=36601).cuda()

img_embeddings_all = get_image_embeddings(model_path, model)
spot_embeddings_all = get_spot_embeddings(model_path, model)

img_embeddings_all = img_embeddings_all.cpu().numpy()
spot_embeddings_all = spot_embeddings_all.cpu().numpy()
print(img_embeddings_all.shape)
print(spot_embeddings_all.shape)

os.makedirs(save_path, exist_ok=True)

for i in range(4):
    index_start = sum(datasize[:i])
    index_end = sum(datasize[:i+1])
    image_embeddings = img_embeddings_all[index_start:index_end]
    spot_embeddings = spot_embeddings_all[index_start:index_end]
    print(image_embeddings.shape)
    print(spot_embeddings.shape)
    np.save(save_path + f"img_embeddings_{i+1}.npy", image_embeddings.T)
    np.save(save_path + f"spot_embeddings_{i+1}.npy", spot_embeddings.T)

# Load spot expression
spot_expression1 = np.load("./GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy")
spot_expression2 = np.load("./GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy")
spot_expression3 = np.load("./GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy")
spot_expression4 = np.load("./GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy")

spot_embeddings1 = np.load(save_path + "spot_embeddings_1.npy")
spot_embeddings2 = np.load(save_path + "spot_embeddings_2.npy")
spot_embeddings4 = np.load(save_path + "spot_embeddings_4.npy")
image_embeddings3 = np.load(save_path + "img_embeddings_3.npy")

# Query setup
image_query = image_embeddings3
expression_gt = spot_expression3
spot_key = np.concatenate([spot_embeddings1, spot_embeddings2, spot_embeddings4], axis=1)
expression_key = np.concatenate([spot_expression1, spot_expression2, spot_expression4], axis=1)

# Shape guards
if image_query.shape[1] != 256:
    image_query = image_query.T
    print("image query shape: ", image_query.shape)
if expression_gt.shape[0] != image_query.shape[0]:
    expression_gt = expression_gt.T
    print("expression_gt shape: ", expression_gt.shape)
if spot_key.shape[1] != 256:
    spot_key = spot_key.T
    print("spot_key shape: ", spot_key.shape)
if expression_key.shape[0] != spot_key.shape[0]:
    expression_key = expression_key.T
    print("expression_key shape: ", expression_key.shape)

# Matching and aggregation (average of top-50)
print("finding matches, using average of top 50 expressions")
indices = find_matches(spot_key, image_query, top_k=50)
matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
for i in range(indices.shape[0]):
    matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0)
    matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

# Correlation calculations
true = expression_gt
pred = matched_spot_expression_pred

print(pred.shape)
print(true.shape)

corr = np.zeros(pred.shape[0])
for i in range(pred.shape[0]):
    corr[i] = np.corrcoef(pred[i, :], true[i, :])[0, 1]
corr = corr[~np.isnan(corr)]
print("Mean correlation across cells: ", np.mean(corr))

corr = np.zeros(pred.shape[1])
for i in range(pred.shape[1]):
    corr[i] = np.corrcoef(pred[:, i], true[:, i])[0, 1]
corr = corr[~np.isnan(corr)]
print("number of non-zero genes: ", corr.shape[0])
print("max correlation: ", np.max(corr))
ind = np.argsort(np.sum(true, axis=0))[-50:]
print("mean correlation highly expressed genes: ", np.mean(corr[ind]))
ind = np.argsort(np.var(true, axis=0))[-50:]
print("mean correlation highly variable genes: ", np.mean(corr[ind]))

# Marker genes
marker_gene_list = ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]
gene_names = pd.read_csv("./GSE240429_data/data/filtered_expression_matrices/3/features.tsv", header=None, sep="\t").iloc[:, 1].values
hvg_b = np.load("./GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
marker_gene_ind = np.zeros(len(marker_gene_list))
for i in range(len(marker_gene_list)):
    marker_gene_ind[i] = np.where(gene_names[hvg_b] == marker_gene_list[i])[0]
print("mean correlation marker genes: ", np.mean(corr[marker_gene_ind.astype(int)]))

# Save predictions
np.save(save_path + "matched_spot_embeddings_pred.npy", matched_spot_embeddings_pred.T)
np.save(save_path + "matched_spot_expression_pred.npy", matched_spot_expression_pred.T)

# ---------------------------
# UMAP + Leiden sweep on predicted matrix
# ---------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

raw_gene_names = pd.read_csv(
    "./GSE240429_data/data/filtered_expression_matrices/3/features.tsv",
    header=None, sep="\t"
).iloc[:, 1].values
hvg_idx = np.load("./GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
gene_names_hvg = raw_gene_names[hvg_idx]

pred_mat = matched_spot_expression_pred
true_mat = expression_gt

if pred_mat.shape[1] != len(gene_names_hvg) and pred_mat.shape[0] == len(gene_names_hvg):
    pred_mat = pred_mat.T
if true_mat.shape[1] != len(gene_names_hvg) and true_mat.shape[0] == len(gene_names_hvg):
    true_mat = true_mat.T

assert pred_mat.shape[1] == len(gene_names_hvg)

good_rows = ~np.all(pred_mat == 0, axis=1)
good_cols = ~np.all(pred_mat == 0, axis=0)
pred_cluster = pred_mat[good_rows][:, good_cols]
selected_genes = gene_names_hvg[good_cols]

adata = sc.AnnData(X=pred_cluster, var=pd.DataFrame(index=selected_genes))
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1024)
adata = adata[:, ~np.isnan(adata.var['dispersions'])].copy()
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)

if 'X_umap' not in adata.obsm_keys():
    sc.tl.umap(adata)

resolutions = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
for res in resolutions:
    key = f'leiden_r{res}'
    if key not in adata.obs.columns:
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=key,
            flavor="igraph",
            directed=False,
            n_iterations=2
        )
    adata.obs[key] = adata.obs[key].astype("category")
    fig = sc.pl.umap(
        adata,
        color=key,
        size=12,
        alpha=0.9,
        legend_loc='right margin',
        frameon=True,
        return_fig=True,
        show=False
    )
    fig.set_size_inches(7, 5)
    plt.show()

# ---------------------------
# GO enrichment on top-50 variable genes (optional)
# ---------------------------
gene_vars = np.array(pred_cluster.var(axis=0)).flatten()
top50_var_idx = np.argsort(-gene_vars)[:50]
top50_var_genes = selected_genes[top50_var_idx]
print("Top-50 most variable genes:", top50_var_genes)

def run_go_enrichment_simple(gene_list, species='Human', outdir=None):
    enr = gp.enrichr(
        gene_list=gene_list.tolist(),
        gene_sets=['GO_Biological_Process_2021'],
        organism=species,
        outdir=outdir,
        cutoff=0.1,
        verbose=False
    )
    if enr.results.empty:
        print("Warning: no significant enrichment")
        return None
    return enr.results

def plot_go_barplot(go_results, title='GO Enriched Terms', top_n=10, save_path=None):
    go_results_sorted = go_results.sort_values('Adjusted P-value').head(top_n)
    terms = go_results_sorted['Term']
    pvals = go_results_sorted['Adjusted P-value']
    neg_log_pvals = -np.log10(pvals)

    plt.figure(figsize=(8, 5))
    plt.barh(terms, neg_log_pvals)
    plt.xlabel('-log10(Adjusted P-value)')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

go_var_res = run_go_enrichment_simple(top50_var_genes, outdir='go_high_variance')
if go_var_res is not None:
    print("\nTop-10 enriched GO terms (high-variance genes):")
    print(go_var_res[['Term', 'Adjusted P-value', 'Overlap']].head(10))
    plot_go_barplot(go_var_res, title='High Variance Genes GO Enrichment', save_path='go_high_variance.png')
