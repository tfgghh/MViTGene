import os
import numpy as np
import torch
import pandas as pd
from models import CLIPModel  # Assuming you have a model file similar to this


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE   = "./GSE240429_data/data/filtered_expression_matrices"
MODEL_PATH = "./clip/best.pt"
SLIDES = [1, 2, 3, 4]
K = 100


def load_cells_by_genes(path):
    arr = np.load(path)
    return arr.T if arr.shape[0] > arr.shape[1] else arr  # cells x genes

def infer_full_genes(state):
    cands=[]
    for k,v in state.items():
        kn=k.replace("module.","")
        if "pred_head" in kn and kn.endswith("weight"):
            cands.append(v)
    if not cands: raise KeyError("no pred_head.*.weight")
    return cands[-1].shape[0]

def cosine_sim(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T

def Hit_at_k(pred, true, k):
    num_samples = pred.shape[0]
    pred_top = np.argsort(pred, axis=1)[:, -k:]
    true_top = np.argsort(true, axis=1)[:, -k:]
    hits = 0
    for i in range(num_samples):
        if len(set(pred_top[i]).intersection(set(true_top[i]))):
            hits += 1
    return hits / num_samples


def main_like_heclip():
    print(f"Loading model from: {MODEL_PATH}")  # Add debug print to check path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return


    raw = torch.load(MODEL_PATH, map_location="cpu")
    full_G = infer_full_genes(raw)
    model = CLIPModel(num_genes=full_G)
    clean = {k.replace("module.","").replace("well","spot"): v for k,v in raw.items()}
    model.load_state_dict(clean, strict=True)
    model.to(DEVICE).eval()
    print("[INFO] model loaded on", DEVICE)


    per_slide = {}
    with torch.no_grad():
        for sid in SLIDES:
            feats_i = pd.read_csv(os.path.join(BASE, str(sid), "features.tsv"),
                                  sep="\t", header=None).iloc[:,1].astype(str).values
            Y_gt = load_cells_by_genes(os.path.join(BASE, str(sid), "hvg_matrix.npy"))
            X_enc = load_cells_by_genes(os.path.join(BASE, str(sid), "harmony_hvg_matrix.npy"))
            X_enc_t = torch.tensor(X_enc, dtype=torch.float32, device=DEVICE)
            H_t = model.spot_projection(X_enc_t)      # spot embedding
            H = H_t.detach().cpu().numpy()

            per_slide[sid] = dict(
                feats=feats_i,
                gt=Y_gt,
                H=H
            )
            del X_enc_t, H_t
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()


    allP, allT = [], []
    rows = []
    for qid in SLIDES:
        q = per_slide[qid]
        Y_gt_q = q["gt"]  # [Nq, Gq]
        H_q = q["H"]  # [Nq, d]
        Nq, Gq = Y_gt_q.shape


        H_ref_list, Y_ref_list = [], []
        for rid in SLIDES:
            if rid == qid: continue
            r = per_slide[rid]
            H_ref_list.append(r["H"])
            Y_ref_list.append(r["gt"])
        H_ref = np.vstack(H_ref_list)
        Y_ref = np.vstack(Y_ref_list)


        S = cosine_sim(H_q, H_ref)  # [Nq, Nr]
        Nr = S.shape[1]
        kk = min(K, Nr)
        top_idx = np.argpartition(S, kth=Nr - kk, axis=1)[:, -kk:]


        P_knn = np.zeros_like(Y_gt_q)
        for i in range(Nq):
            P_knn[i] = np.mean(Y_ref[top_idx[i]], axis=0)


        allP.append(P_knn)
        allT.append(Y_gt_q)


        hk1 = Hit_at_k(P_knn, Y_gt_q, 1)
        hk2 = Hit_at_k(P_knn, Y_gt_q, 2)
        hk3 = Hit_at_k(P_knn, Y_gt_q, 3)

        print(f"[Slide {qid}] N={Nq} G={Gq} | Hit@1/2/3 = {hk1:.4f}/{hk2:.4f}/{hk3:.4f}")
        rows.append({
            "slide": qid, "N": Nq, "G": Gq,
            "Hit@1": round(float(hk1), 4),
            "Hit@2": round(float(hk2), 4),
            "Hit@3": round(float(hk3), 4),
        })


    AP = np.vstack(allP)
    AT = np.vstack(allT)
    Hk1 = Hit_at_k(AP, AT, 1)
    Hk2 = Hit_at_k(AP, AT, 2)
    Hk3 = Hit_at_k(AP, AT, 3)
    print(f"[ALL] Hit@1/2/3 = {Hk1:.4f}/{Hk2:.4f}/{Hk3:.4f}")
    rows.append({
        "slide": "ALL", "N": AP.shape[0], "G": AP.shape[1],
        "Hit@1": round(float(Hk1), 4),
        "Hit@3": round(float(Hk2), 4),
        "Hit@5": round(float(Hk3), 4),
    })


    out_csv = "./hit_results_hvg.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

if __name__ == "__main__":
    main_like_heclip()
