import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CLIPDataset(Dataset):
    def __init__(self,
                 image_path,
                 spatial_pos_path,
                 barcode_path,
                 reduced_mtx_path,
                 full_expr_path=None,
                 vocab_gene_indices=None,
                 top_k=128,
                 augment=True):

        super().__init__()

        # Read image and spatial information
        self.whole_image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)

        # Reduced expression matrix (e.g. Harmony)
        self.reduced_matrix = np.load(reduced_mtx_path).T
        self.full_expression = np.load(full_expr_path) if full_expr_path else None

        self.vocab_gene_indices = vocab_gene_indices
        self.top_k = top_k
        self.augment = augment

        if vocab_gene_indices is not None:
            self.gene_id_to_token_id = {int(gid): i for i, gid in enumerate(vocab_gene_indices)}

        self.basic_transform = T.Compose([
            T.ToTensor()
        ])

        print(f"[INFO] Dataset initialized: {len(self.barcode_tsv)} samples.")

    def strong_transform(self, image):
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)

        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)

        color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        image = color_jitter(image)

        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        image_np = np.asarray(image).astype(np.float32)
        noise = np.random.normal(0, 10.0, size=image_np.shape)
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(image_np)

    def __getitem__(self, idx):
        barcode = self.barcode_tsv.values[idx, 0]
        match = self.spatial_pos_csv[self.spatial_pos_csv[0] == barcode]

        if match.empty:
            raise ValueError(f"Barcode {barcode} not found in spatial_pos_csv.")

        v1 = match.iloc[0, 4]
        v2 = match.iloc[0, 5]

        crop = self.whole_image[(v1 - 112):(v1 + 112), (v2 - 112):(v2 + 112)]
        if crop.shape[0] != 224 or crop.shape[1] != 224:
            crop = cv2.resize(crop, (224, 224))

        image = Image.fromarray(crop)

        if self.augment:
            image = self.strong_transform(image)

        image = self.basic_transform(image)

        item = {
            "image": image.float(),
            "reduced_expression": torch.tensor(self.reduced_matrix[idx], dtype=torch.float),
            "barcode": barcode,
            "spatial_coords": torch.tensor([v1, v2], dtype=torch.float)
        }

        if self.full_expression is not None:
            full_expr = self.full_expression[idx]  # shape: (G,)
            item["expression"] = torch.tensor(full_expr, dtype=torch.float)

            if self.vocab_gene_indices is not None:
                top_gene_indices = np.argsort(-full_expr)  # Prioritize high expression
                token_ids = []
                expr_values = []

                for gid in top_gene_indices:
                    if gid in self.gene_id_to_token_id:
                        token_ids.append(self.gene_id_to_token_id[gid])
                        expr_values.append(float(full_expr[gid]))
                    if len(token_ids) == self.top_k:
                        break

                if len(token_ids) < self.top_k:
                    pad_len = self.top_k - len(token_ids)
                    token_ids += [0] * pad_len
                    expr_values += [0.0] * pad_len

                item["expression_token"] = torch.tensor(token_ids, dtype=torch.long)
                item["expression_token_value"] = torch.tensor(expr_values, dtype=torch.float)

        return item

    def __len__(self):
        return len(self.barcode_tsv)
