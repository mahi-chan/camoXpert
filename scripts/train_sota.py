"""
CamoXpert SOTA Training for Kaggle
No transfer learning - Pure pretrained backbone training
Optimized for Kaggle GPU constraints
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
import cv2
import numpy as np
from pathlib import Path

# Kaggle paths
KAGGLE_INPUT = '/kaggle/input'
KAGGLE_WORKING = '/kaggle/working'

# Add project to path
sys.path.insert(0, KAGGLE_WORKING)

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from data.augmentations import AdvancedCODAugmentation
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from models.utils import count_parameters, set_seed


class AdvancedCOD10KDataset(COD10KDataset):
    """Dataset with advanced augmentations"""

    def __init__(self, root_dir, split='train', img_size=416, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment_flag = augment and split == 'train'

        super().__init__(root_dir, split, img_size, False)
        self.advanced_aug = AdvancedCODAugmentation(img_size, self.augment_flag)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
        base_name = os.path.splitext(img_name)[0]
        mask = None
        for ext in mask_extensions:
            mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            raise ValueError(f"Mask not found for: {img_name}")

        mask = (mask > 128).astype(np.float32)
        transformed = self.advanced_aug(image, mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=True):
    """Training with mixed precision support"""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", ncols=100)

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp:
            with torch.cuda.amp.autocast():
                pred, aux_loss, deep_outputs = model(images, return_deep_supervision=True)
                loss, loss_dict = criterion(pred, masks, aux_loss, deep_outputs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, aux_loss, deep_outputs = model(images, return_deep_supervision=True)
            loss, loss_dict = criterion(pred, masks, aux_loss, deep_outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics, device):
    """Validation function"""
    model.eval()
    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", ncols=100, leave=False):
        images, masks = images.to(device), masks.to(device)
        pred, _ = model(images)
        all_metrics.append(metrics.compute_all(pred, masks))
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}


def print_metrics(metrics_dict, stage=""):
    """Pretty print metrics"""
    if stage:
        print(f"\n{'=' * 70}\n{stage}\n{'=' * 70}")

    print(f"Accuracy:  Pixel={metrics_dict.get('Pixel_Accuracy', 0):.4f} | "
          f"Prec={metrics_dict.get('Precision', 0):.4f} | "
          f"Recall={metrics_dict.get('Recall', 0):.4f}")
    print(f"Segment:   IoU={metrics_dict.get('IoU', 0):.4f} | "
          f"Dice={metrics_dict.get('Dice_Score', 0):.4f} | "
          f"F1={metrics_dict.get('F-measure', 0):.4f}")
    print(f"COD:       S={metrics_dict.get('S-measure', 0):.4f} | "
          f"E={metrics_dict.get('E-measure', 0):.4f} | "
          f"MAE={metrics_dict.get('MAE', 0):.4f}")


def find_dataset_path():
    """Auto-detect dataset path in Kaggle"""
    possible_paths = [
        '/kaggle/input/cod10k-v3',
        '/kaggle/input/cod10k',
        '/kaggle/input/camouflaged-object-detection',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found dataset at: {path}")
            return path

    # List available datasets
    print("Available datasets:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            print(f"  - /kaggle/input/{item}")

    raise FileNotFoundError("Dataset not found. Please check Kaggle input datasets.")


def main():
    """Main training function optimized for Kaggle"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Auto-detect dataset
    dataset_path = find_dataset_path()

    # Kaggle configuration
    config = {
        'backbone': 'edgenext_base_usi',
        'batch_size': 4,  # Optimized for Kaggle GPU
        'img_size': 416,
        'epochs': 80,
        'lr': 0.0001,
        'weight_decay': 0.00005,
        'use_amp': True,  # Mixed precision for faster training
        'checkpoint_dir': os.path.join(KAGGLE_WORKING, 'checkpoints_sota')
    }

    print("\n" + "=" * 70)
    print("CAMOXPERT SOTA TRAINING - KAGGLE OPTIMIZED")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Backbone:    {config['backbone']}")
    print(f"Image Size:  {config['img_size']}px")
    print(f"Batch Size:  {config['batch_size']}")
    print(f"Epochs:      {config['epochs']}")
    print(f"Mixed Prec:  {config['use_amp']}")
    print(f"Dataset:     {dataset_path}")
    print("=" * 70 + "\n")

    # Load datasets
    print("Loading datasets with advanced augmentations...")
    train_data = AdvancedCOD10KDataset(dataset_path, 'train', config['img_size'], augment=True)
    val_data = AdvancedCOD10KDataset(dataset_path, 'val', config['img_size'], augment=False)

    train_loader = DataLoader(
        train_data, config['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, config['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Train: {len(train_data)} | Val: {len(val_data)}\n")

    # Create model (NO TRANSFER LEARNING - Pure pretrained)
    print(f"Creating {config['backbone']} with ImageNet pretrained weights...")
    model = CamoXpert(
        in_channels=3,
        num_classes=1,
        pretrained=True,  # Uses ImageNet pretrained weights
        backbone=config['backbone'],
        num_experts=4
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Model: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)\n")

    # Setup training
    criterion = AdvancedCODLoss()
    metrics = CODMetrics()
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    best_iou = 0
    best_dice = 0
    history = []

    # ========== STAGE 1: Decoder Fine-tuning (20 epochs) ==========
    print("=" * 70)
    print("STAGE 1: Decoder Fine-tuning - Frozen Backbone (20 epochs)")
    print("=" * 70)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)

    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}/20")
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, config['use_amp'])
        val_m = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {loss:.4f}")
        print_metrics(val_m)

        # Save best model
        if val_m['IoU'] > best_iou:
            best_iou = val_m['IoU']
            best_dice = val_m['Dice_Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_iou': best_iou,
                'best_dice': best_dice,
                'val_metrics': val_m
            }, os.path.join(config['checkpoint_dir'], 'best_model_sota.pth'))
            print(f"✓ New best! IoU: {best_iou:.4f} | Dice: {best_dice:.4f}")

        history.append({'epoch': epoch, 'stage': 1, 'train_loss': loss, **val_m})

    print(f"\nStage 1 Complete. Best IoU: {best_iou:.4f}\n")

    # ========== STAGE 2: Full Fine-tuning (60 epochs) ==========
    print("=" * 70)
    print("STAGE 2: Full Fine-tuning - All Layers (60 epochs)")
    print("=" * 70)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Differential learning rates
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr'] * 0.1},  # Slower for backbone
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)

    for epoch in range(20, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, config['use_amp'])
        val_m = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {loss:.4f}")
        print_metrics(val_m)

        if val_m['IoU'] > best_iou:
            best_iou = val_m['IoU']
            best_dice = val_m['Dice_Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_iou': best_iou,
                'best_dice': best_dice,
                'val_metrics': val_m
            }, os.path.join(config['checkpoint_dir'], 'best_model_sota.pth'))
            print(f"✓ New best! IoU: {best_iou:.4f} | Dice: {best_dice:.4f}")

        history.append({'epoch': epoch, 'stage': 2, 'train_loss': loss, **val_m})

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_m
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch + 1}.pth'))

    # Save training history
    with open(os.path.join(config['checkpoint_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best IoU:     {best_iou:.4f}")
    print(f"Best Dice:    {best_dice:.4f}")
    print(f"Target SOTA:  0.70-0.72")
    print(f"Status:       {'✓ SOTA ACHIEVED!' if best_iou >= 0.70 else '↗ Close to SOTA'}")
    print("=" * 70)

    print_metrics(val_m, "FINAL VALIDATION METRICS")

    return best_iou, best_dice


if __name__ == '__main__':
    best_iou, best_dice = main()
    print(f"\nTraining finished! Best IoU: {best_iou:.4f}")