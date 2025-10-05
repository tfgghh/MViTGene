import torch
from torch import nn
import torch.nn.functional as F
import timm
import config as CFG
from modules import ImageEncoder, ProjectionHead

from einops import rearrange

def pearson_loss(pred, target, eps=1e-8):
    """
    Pearson loss = 1 - mean(PCC across genes)
    pred: (batch, num_genes)
    target: (batch, num_genes)
    """
    vx = pred - pred.mean(dim=0, keepdim=True)
    vy = target - target.mean(dim=0, keepdim=True)

    corr = (vx * vy).sum(dim=0) / (
        torch.sqrt((vx ** 2).sum(dim=0) + eps) *
        torch.sqrt((vy ** 2).sum(dim=0) + eps)
    )

    return 1 - corr.mean()



class CLIPModel(nn.Module):
    def __init__(self, temperature=0.3, image_embedding=512, spot_embedding=3467, projection_dim=256, num_genes=3467):
        super().__init__()

        # Initialize parameters
        self.temperature = temperature

        # Image encoder and projection layers
        self.image_encoder = ImageEncoder(model_name='resnet50')  # Specify ResNet50 encoder
        self.image_projection = ProjectionHead(embedding_dim=2048, projection_dim=256)

        # Spot projection layer
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim)

        # Gene expression prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_genes)
        )

        # ViT encoder (MobileViT model)
        self.vit_encoder = timm.create_model('mobilevit_s', pretrained=True, num_classes=0)

        # Freeze ViT parameters (optional)
        for p in self.vit_encoder.parameters():
            p.requires_grad = True

        # ViT projection layer
        self.vit_projection = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, projection_dim)
        )

        # Identity mapping for common space alignment
        self.image_to_common = nn.Identity()
        self.spot_to_common = nn.Identity()

    def split_patches(self, image, grid=3):
        B, C, H, W = image.shape
        patch_h = H // grid
        patch_w = W // grid
        valid_h = patch_h * grid
        valid_w = patch_w * grid

        # Ensure image is divisible by grid
        if valid_h != H or valid_w != W:
            top = (H - valid_h) // 2
            left = (W - valid_w) // 2
            image = image[:, :, top:top + valid_h, left:left + valid_w]

        # Split image into patches
        patches = rearrange(image, 'b c (gh ph) (gw pw) -> (b gh gw) c ph pw', gh=grid, gw=grid, ph=patch_h, pw=patch_w)
        return patches, B

    def aggregate_patch_embeddings(self, embeddings, B, grid=3):
        # Aggregate patch embeddings by averaging
        emb = rearrange(embeddings, '(b gh gw) d -> b d (gh gw)', gh=grid, gw=grid)
        return emb.mean(dim=-1)

    def gaussian_sim(self, x, sigma=0.2):
        # Compute Gaussian similarity between embeddings
        dist = torch.cdist(x, x, p=2)
        sim = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        return sim

    def forward(self, batch, epoch=None):
        images = batch["image"]
        spot_features = batch["reduced_expression"]

        # Process image patches
        patch_images, B = self.split_patches(images, grid=3)
        patch_features = self.image_encoder(patch_images)
        patch_embeddings = self.image_projection(patch_features)
        image_embeddings = self.aggregate_patch_embeddings(patch_embeddings, B, grid=3)

        # Process spot features
        spot_embeddings = self.spot_projection(spot_features)

        # ViT processing
        vit_feats = self.vit_encoder(images)
        vit_embeddings = F.normalize(self.vit_projection(vit_feats), dim=-1)

        # Normalize embeddings to the common space
        img_emb = F.normalize(self.image_to_common(image_embeddings), dim=-1)
        spot_emb = F.normalize(self.spot_to_common(spot_embeddings), dim=-1)

        # Compute cosine similarity and loss between image and spot embeddings
        logits = (spot_emb @ img_emb.T) / self.temperature
        targets_hard = torch.arange(logits.size(0)).to(logits.device)

        # Compute Gaussian similarity
        sim_img = self.gaussian_sim(img_emb)
        sim_spot = self.gaussian_sim(spot_emb)

        # Generate soft targets
        targets_soft = F.softmax((sim_img + sim_spot) / 2, dim=-1)

        # Compute soft contrastive loss
        soft_loss = F.kl_div(F.log_softmax(logits, dim=-1), targets_soft, reduction='batchmean')

        # Compute hard contrastive loss
        hard_loss = F.cross_entropy(logits, targets_hard)

        # Compute ViT loss
        vit_logit = (spot_emb @ vit_embeddings.T) / self.temperature
        vit_loss = F.cross_entropy(vit_logit, targets_hard)

        # Expression prediction loss
        expr_loss = torch.tensor(0.0, device=img_emb.device)
        if "expression" in batch:
            full_expression = batch["expression"]
            pred = self.pred_head(img_emb)
            mse = F.mse_loss(pred, full_expression)
            pcc = pearson_loss(pred, full_expression)
            expr_loss = 0* mse + 0 * pcc  # The optimal configuration is dataset-dependent and should be determined empirically.

        # ====== Loss Weights ======
        # These weights control the contribution of each loss component.
        # Users should tune them based on their dataset and objectives.
        # These coefficients need to be tuned manually.
        expr_weight = 0
        soft_weight = 0
        hard_weight = 0
        vit_weight = 0

        # Total loss
        total_loss = (
                soft_weight * soft_loss +
                hard_weight * hard_loss +
                expr_weight * expr_loss +
                vit_weight * vit_loss
        )

        return total_loss


class WeightingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, cos_sim, euc_dist, man_dist):
        inputs = torch.stack([cos_sim, euc_dist, man_dist], dim=2)  # (Nq, Ns, 3)
        return self.fc(inputs).squeeze(-1)  # (Nq, Ns)


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")