import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
import cv2
import numpy as np

from models.camoxpert import CamoXpert
from models.transfer_learning import transfer_weights
from data.dataset import COD10KDataset
from data.augmentations import AdvancedCODAugmentation
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from utils.helpers import count_parameters, set_seed


class AdvancedCOD10KDataset(COD10KDataset):
    """Dataset with advanced augmentations"""

    def __init__(self, root_dir, split='train', img_size=416, augment=True):
        # Call parent init to set up paths
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment_flag = augment and split == 'train'

        # Setup dataset paths (same logic as parent)
        super().__init__(root_dir, split, img_size, False)  # Don't use parent's augmentation

        # Use advanced augmentation
        self.advanced_aug = AdvancedCODAugmentation(img_size, self.augment_flag)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_extensions = ['.png', '.jpg', '.jpeg']
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

        # Apply advanced augmentation
        transformed = self.advanced_aug(image, mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


def train_epoch(model, loader, criterion, optimizer, device, use_deep_sup=True):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", ncols=100)

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        if use_deep_sup:
            pred, aux_loss, deep_outputs = model(images, return_deep_supervision=True)
            loss, loss_dict = criterion(pred, masks, aux_loss, deep_outputs)
        else:
            pred, aux_loss = model(images)
            loss, loss_dict = criterion(pred, masks, aux_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics, device):
    model.eval()
    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", ncols=100, leave=False):
        images, masks = images.to(device), masks.to(device)
        pred, _ = model(images)
        all_metrics.append(metrics.compute_all(pred, masks))
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    print("=" * 70)
    print("CAMOXPERT SOTA TRAINING PIPELINE")
    print("=" * 70)
    print(f"Backbone:    {args.backbone}")
    print(f"Image Size:  {args.img_size}px")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"Device:      {device}")
    print("=" * 70 + "\n")

    # Load datasets with advanced augmentation
    print("Loading datasets with advanced augmentations...")
    train_data = AdvancedCOD10KDataset(args.dataset_path, 'train', args.img_size, augment=True)
    val_data = AdvancedCOD10KDataset(args.dataset_path, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Data: Train={len(train_data)}, Val={len(val_data)}\n")

    # Create model
    print(f"Creating {args.backbone} model...")
    model = CamoXpert(in_channels=3, num_classes=1, pretrained=True,
                      backbone=args.backbone, num_experts=4)

    # Transfer weights if small model provided
    if args.small_model_path and os.path.exists(args.small_model_path):
        print(f"\nTransferring weights from small model...")
        model = transfer_weights(args.small_model_path, model, device)

    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)\n")

    # Setup training
    criterion = AdvancedCODLoss()
    metrics = CODMetrics()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_iou = 0
    history = []

    # STAGE 1: Decoder fine-tuning
    print("=" * 70)
    print("STAGE 1: Decoder Fine-tuning (20 epochs)")
    print("=" * 70)

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)

    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}/20")
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {loss:.4f} | IoU: {val_m['IoU']:.4f} | "
              f"F1: {val_m['F-measure']:.4f} | MAE: {val_m['MAE']:.4f}")

        if val_m['IoU'] > best_iou:
            best_iou = val_m['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_m
            }, os.path.join(args.checkpoint_dir, 'best_model_sota.pth'))
            print(f"✓ New best: {best_iou:.4f}")

        history.append({'epoch': epoch, 'train_loss': loss, **val_m})

    print(f"\nStage 1 complete. Best IoU: {best_iou:.4f}\n")

    # STAGE 2: Full fine-tuning
    print("=" * 70)
    print(f"STAGE 2: Full Fine-tuning ({args.epochs - 20} epochs)")
    print("=" * 70)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)

    for epoch in range(20, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {loss:.4f} | IoU: {val_m['IoU']:.4f} | "
              f"F1: {val_m['F-measure']:.4f} | MAE: {val_m['MAE']:.4f}")

        if val_m['IoU'] > best_iou:
            best_iou = val_m['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_m
            }, os.path.join(args.checkpoint_dir, 'best_model_sota.pth'))
            print(f"✓ New best: {best_iou:.4f}")

        history.append({'epoch': epoch, 'train_loss': loss, **val_m})

    # Save history
    with open(os.path.join(args.checkpoint_dir, 'training_history_sota.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best IoU:        {best_iou:.4f}")
    print(f"Improvement:     +{(best_iou - 0.6048) * 100:.2f}%")
    print(f"Target (SOTA):   0.70-0.72")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CamoXpert SOTA Training")
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--small-model-path', type=str, default=None,
                        help='Path to trained small model for transfer learning')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_sota')
    parser.add_argument('--backbone', type=str, default='edgenext_base_usi')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=416)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.00005)

    args = parser.parse_args()
    main(args)