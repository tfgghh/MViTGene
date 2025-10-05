import torch
print(torch.cuda.is_available())
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
import scipy
from scipy.sparse import csr_matrix
import numpy as np
import config as CFG
from dataset import CLIPDataset
from models import CLIPModel
from utils import AvgMeter
from torch.utils.data import DataLoader
import argparse
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--exp_name', type=str, default='clip', help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=1, help='')
parser.add_argument('--num_workers', type=int, default=8, help='Increase num_workers for parallel data loading')

# DDP-related arguments
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

def ensure_expression_npy(matrix_dir, force=False):
    npy_path = os.path.join(matrix_dir, "expression.npy")
    mtx_path = os.path.join(matrix_dir, "matrix.mtx")

    if not os.path.exists(npy_path) or force:
        print(f"[INFO] Generating expression.npy from {mtx_path}")
        matrix = scipy.io.mmread(mtx_path)
        matrix = csr_matrix(matrix).toarray().T  # 转置为 [spot x gene]
        np.save(npy_path, matrix)
        print(f"[INFO] Saved expression.npy to {npy_path}")
    else:
        print(f"[INFO] expression.npy already exists: {npy_path}")
    return npy_path

def build_loaders(args):
    print("Building loaders")
    ensure_expression_npy("./GSE240429_data/data/filtered_expression_matrices/1", force=True)
    ensure_expression_npy("./GSE240429_data/data/filtered_expression_matrices/2", force=True)
    ensure_expression_npy("./GSE240429_data/data/filtered_expression_matrices/4", force=True)

    expr1 = np.load("./GSE240429_data/data/filtered_expression_matrices/1/expression.npy")
    num_genes = expr1.shape[1]

    dataset = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_A1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/1/expression.npy"
    )

    dataset2 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_B1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/2/expression.npy"
    )

    dataset4 = CLIPDataset(
        image_path="./GSE240429_data/image/GEX_C73_D1_Merged.tiff",
        spatial_pos_path="./GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
        reduced_mtx_path="./GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
        barcode_path="./GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv",
        full_expr_path="./GSE240429_data/data/filtered_expression_matrices/4/expression.npy"
    )

    dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset4])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(len(train_dataset), len(test_dataset))
    print("train/test split completed")

    # Increase num_workers for parallel data loading
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  # Shuffle for train
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    print("Finished building loaders")
    return train_loader, test_loader, num_genes

def cleanup():
    dist.destroy_process_group()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(model, train_loader, optimizer, args, epoch, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}  # Move batch to GPU
        optimizer.zero_grad()

        # Use mixed precision training
        with autocast():
            loss = model(batch, epoch=epoch)

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter

def test_epoch(model, test_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}  # Move batch to GPU
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter

def main():
    print("Starting...")
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    # Check if running in SLURM environment
    if "SLURM_LOCALID" in os.environ and "SLURM_NODEID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_id = int(os.environ["SLURM_NODEID"])
        rank = node_id * ngpus_per_node + local_rank
    else:
        local_rank = 0
        rank = 0

    current_device = local_rank
    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=rank)
    print("Process group ready!")

    train_loader, test_loader, num_genes = build_loaders(args)

    print('From Rank: {}, ==> Making model..'.format(rank))
    model = CLIPModel(num_genes=num_genes).cuda(current_device)
    print("Image encoder is ResNet50")
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[current_device],
        find_unused_parameters=False  # Disabled to avoid extra graph traversals
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Cosine Annealing Learning Rate Scheduler
    T_max = args.max_epochs  # Total number of epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    # Store losses for plotting later
    train_losses = []
    test_losses = []

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")

        # Train the model
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, args, epoch)
        train_losses.append(train_loss.avg)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)
        test_losses.append(test_loss.avg)

        # Save the best model based on validation loss
        if test_loss.avg < best_loss and rank == 0:
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))
            best_loss = test_loss.avg
            best_epoch = epoch
            torch.save(model.state_dict(), str(args.exp_name) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

        # Update learning rate
        lr_scheduler.step()

    print("Done! Final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))

    # Plotting the loss curves
    plt.figure()
    plt.plot(range(1, args.max_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.max_epochs + 1), test_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig(f"{args.exp_name}/loss_curve.png")
    plt.show()

    cleanup()

if __name__ == "__main__":
    main()