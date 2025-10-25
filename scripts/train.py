import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from datetime import datetime

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from losses.camoxpert_loss import CamoXpertLoss
from metrics.cod_metrics import CODMetrics
from models.utils import count_parameters, set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training", ncols=100):
        images, masks = images.to(device), masks.to(device)
        outputs, aux_loss = model(images)
        loss, _ = criterion(outputs, masks, aux_loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, metrics, device):
    model.eval()
    all_metrics = []
    for images, masks in tqdm(dataloader, desc="Validation", ncols=100, leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs, _ = model(images)
        batch_metrics = metrics.compute_all(outputs, masks)
        all_metrics.append(batch_metrics)
    return {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print("CamoXpert Training with Pretrained EdgeNeXt Backbone")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Stage 1 (Frozen Backbone): 15 epochs")
    print(f"Stage 2 (Full Fine-tuning): {args.epochs - 15} epochs")

    train_dataset = COD10KDataset(root_dir=args.dataset_path, split='train', img_size=args.img_size, augment=True)
    val_dataset = COD10KDataset(root_dir=args.dataset_path, split='val', img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    print(f"\nDatasets: Train={len(train_dataset)}, Val={len(val_dataset)}\n")

    model = CamoXpert(in_channels=3, num_classes=1, pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable\n")

    criterion = CamoXpertLoss()
    metrics = CODMetrics()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_iou = 0
    history = []

    print("STAGE 1: Training Decoder (Frozen Backbone)")

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    for epoch in range(15):
        print(f"\nEpoch {epoch + 1}/15")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {train_loss:.4f}")
        print(
            f"Val - MAE: {val_metrics['MAE']:.4f} | IoU: {val_metrics['IoU']:.4f} | F1: {val_metrics['F-measure']:.4f} | S: {val_metrics['S-measure']:.4f} | E: {val_metrics['E-measure']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'best_iou': best_iou, 'val_metrics': val_metrics}, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"✓ New best! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})

    print(f"\nStage 1 Complete. Best IoU: {best_iou:.4f}\n")

    print("STAGE 2: Full Model Fine-tuning")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 15, eta_min=1e-6)

    for epoch in range(15, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"Loss: {train_loss:.4f}")
        print(
            f"Val - MAE: {val_metrics['MAE']:.4f} | IoU: {val_metrics['IoU']:.4f} | F1: {val_metrics['F-measure']:.4f} | S: {val_metrics['S-measure']:.4f} | E: {val_metrics['E-measure']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'best_iou': best_iou, 'val_metrics': val_metrics}, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"✓ New best! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})

    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training Complete! Best IoU: {best_iou:.4f}")


parser = argparse.ArgumentParser(description="CamoXpert Training")
parser.add_argument("--dataset-path", type=str, required=True)
parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--img-size", type=int, default=352)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--t0", type=int, default=10)
parser.add_argument("--t-mult", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)