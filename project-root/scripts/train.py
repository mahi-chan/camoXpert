import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from losses.camoxpert_loss import CamoXpertLoss
from metrics.cod_metrics import CODMetrics
from models.utils import count_parameters, save_checkpoint, set_seed, load_config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs, aux_loss = model(images)
        loss, loss_dict = criterion(outputs, masks, aux_loss)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, metrics, device):
    model.eval()
    val_loss = 0
    all_metrics = {'MAE': 0, 'F-measure': 0, 'S-measure': 0, 'E-measure': 0, 'IoU': 0}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs, aux_loss = model(images)
            loss, _ = criterion(outputs, masks, aux_loss)
            val_loss += loss.item()

            batch_metrics = metrics.compute_all(outputs, masks)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

    avg_metrics = {k: v / len(dataloader) for k, v in all_metrics.items()}
    return val_loss / len(dataloader), avg_metrics


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(args.seed)

    # Load datasets
    train_dataset = COD10KDataset(root_dir=args.dataset_path, split='train', img_size=args.img_size, augment=True)
    val_dataset = COD10KDataset(root_dir=args.dataset_path, split='val', img_size=args.img_size, augment=False)
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
    metrics = CODMetrics()

    # Training loop
    best_iou = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")

        # Save checkpoint if best model
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_metrics
            }
            save_checkpoint(checkpoint, args.checkpoint_dir, f'best_model.pth')
            print(f"✅ New best model saved! IoU: {best_iou:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }
            save_checkpoint(checkpoint, args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CamoXpert Training Script")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=352, help="Image size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="Weight decay")
    parser.add_argument("--t0", type=int, default=10, help="T_0 for cosine annealing")
    parser.add_argument("--t-mult", type=int, default=2, help="T_mult for cosine annealing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on")
    args = parser.parse_args()

    main(args)