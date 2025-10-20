# train.py

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from models.camo_xpert import CamoXpert
from dataset import COD10KDataset
from loss import CamoXpertLoss
from metrics import CODMetrics
from models.utils import count_parameters, save_checkpoint

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs, aux_loss = model(images)
        loss, _ = criterion(outputs, masks, aux_loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, metrics, device):
    model.eval()
    val_loss = 0
    all_metrics = {}
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs, aux_loss = model(images)
            loss, _ = criterion(outputs, masks, aux_loss)
            val_loss += loss.item()
            batch_metrics = metrics.compute_all(outputs, masks)
            for key, value in batch_metrics.items():
                all_metrics[key] = all_metrics.get(key, 0) + value
    avg_metrics = {k: v / len(dataloader) for k, v in all_metrics.items()}
    return val_loss / len(dataloader), avg_metrics

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = COD10KDataset(root_dir=args.dataset_path, split="train", img_size=args.img_size, augment=True)
    val_dataset = COD10KDataset(root_dir=args.dataset_path, split="val", img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = CamoXpert(in_channels=3, num_classes=1).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Loss, optimizer, and scheduler
    criterion = CamoXpertLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)

    # Metrics
    metric