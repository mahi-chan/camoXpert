"""
Command-line training script with gradient accumulation & checkpointing
Includes gradient explosion protection with automatic detection and recovery.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import json
import glob
from tqdm import tqdm
import numpy as np
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert, count_parameters
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='CamoXpert SOTA Training with Gradient Explosion Protection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gradient Explosion Protection:
  - Gradient clipping: 0.5 (conservative)
  - Stage 2 LR: 50% of Stage 1 for stability
  - Conservative gradient scaler with slower growth
  - Automatic NaN/Inf detection with checkpoint recovery

Recovery from Gradient Explosion:
  If training crashes with gradient explosion, resume with:
    --resume ./checkpoints/best_model.pth --lr 0.000125
        """
    )

    parser.add_argument('command', type=str, choices=['train'])
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--backbone', type=str, default='edgenext_base')
    parser.add_argument('--num-experts', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--stage1-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--grad-clip', type=float, default=0.5,
                        help='Gradient clipping norm (default: 0.5, lower = more stable)')
    parser.add_argument('--deep-supervision', action='store_true', default=False)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--use-ema', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def enable_gradient_checkpointing(model):
    print("🔧 Enabling gradient checkpointing...")
    checkpointed = 0
    for name, module in model.named_modules():
        if 'moe' in name.lower() or 'sdta' in name.lower():
            if hasattr(module, 'forward'):
                original_forward = module.forward

                def checkpointed_forward(self, *args, **kwargs):
                    def custom_forward(*inputs):
                        return original_forward(*inputs, **kwargs)

                    return gradient_checkpoint(custom_forward, *args, use_reentrant=False)

                module.forward = checkpointed_forward.__get__(module, type(module))
                checkpointed += 1
    print(f"✓ Checkpointed {checkpointed} modules")
    return model


def train_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps, ema, epoch, total_epochs,
                use_deep_sup, max_grad_norm=0.5):
    """
    Training epoch with gradient explosion protection.

    Args:
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5 for stability)
    """
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    # Gradient norm tracking
    grad_norms = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, (images, masks) in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.cuda.amp.autocast('cuda'):
            pred, aux_loss, deep = model(images, return_deep_supervision=use_deep_sup)
            loss, _ = criterion(pred, masks, aux_loss, deep)

            # Check for NaN/Inf in loss
            if not torch.isfinite(loss):
                print(f"\n❌ NaN/Inf detected in loss at batch {batch_idx}!")
                print(f"   Loss value: {loss.item()}")
                raise ValueError("Training stopped due to NaN/Inf loss.")

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)

            # Compute gradient norm and check for NaN/Inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(grad_norm.item())

            # Check for gradient explosion
            if not torch.isfinite(grad_norm):
                print(f"\n❌ NaN/Inf detected in gradients at batch {batch_idx}!")
                print(f"   Gradient norm: {grad_norm.item()}")
                print(f"   This indicates gradient explosion.")
                print(f"   Please reduce learning rate and restart from last good checkpoint.")
                raise ValueError("Training stopped due to NaN/Inf gradients. See error message above.")

            # Warn on high gradient norms
            if grad_norm > max_grad_norm * 5:
                print(f"\n⚠️  High gradient norm detected: {grad_norm:.2f} (clipped to {max_grad_norm})")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

        epoch_loss += loss.item() * accumulation_steps

        # Show gradient norm in progress bar
        if grad_norms:
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'grad': f'{grad_norms[-1]:.3f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # Handle remaining batches
    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if not torch.isfinite(grad_norm):
            print(f"\n❌ NaN/Inf detected in final gradient update!")
            raise ValueError("Training stopped due to NaN/Inf gradients.")

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Report gradient statistics
    if grad_norms:
        avg_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)
        print(f"   Gradient norm - Avg: {avg_grad:.3f}, Max: {max_grad:.3f}")

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics):
    model.eval()
    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.cuda(), masks.cuda()
        pred, _, _ = model(images)
        pred = torch.sigmoid(pred)
        all_metrics.append(metrics.compute_all(pred, masks))
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    effective_batch = args.batch_size * args.accumulation_steps

    print("\n" + "=" * 70)
    print("CAMOXPERT ULTIMATE TRAINING")
    print("=" * 70)
    print(f"Backbone:         {args.backbone}")
    print(f"Experts:          {args.num_experts}")
    print(f"Resolution:       {args.img_size}px")
    print(f"Batch Size:       {args.batch_size} × {args.accumulation_steps} = {effective_batch} effective")
    print(f"Epochs:           {args.epochs}")
    print(f"Deep Supervision: {args.deep_supervision}")
    print(f"Grad Checkpoint:  {args.gradient_checkpointing}")
    print(f"EMA:              {args.use_ema}")
    print(f"\n🎯 Target: IoU ≥ 0.72")
    print("=" * 70 + "\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Datasets
    train_data = COD10KDataset(args.dataset_path, 'train', args.img_size, augment=True)
    val_data = COD10KDataset(args.dataset_path, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}\n")

    # Model
    model = CamoXpert(3, 1, pretrained=True, backbone=args.backbone, num_experts=args.num_experts).cuda()

    if args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)

    total, trainable = count_parameters(model)
    print(f"Model: {total / 1e6:.1f}M params\n")

    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)
    metrics = CODMetrics()
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model) if args.use_ema else None

    best_iou = 0.0
    history = []
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'best_iou' in checkpoint:
            best_iou = checkpoint['best_iou']
            print(f"   Resumed best IoU: {best_iou:.4f}")

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"   Resuming from epoch: {start_epoch}")

        if ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            ema.shadow = checkpoint['ema_state_dict']
            print(f"   Restored EMA weights")

        print(f"✓ Checkpoint loaded successfully\n")

    # Stage 1
    if start_epoch < args.stage1_epochs:
        print("=" * 70)
        print("STAGE 1: DECODER TRAINING")
        print("=" * 70)

        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

        total_steps = len(train_loader) * args.stage1_epochs // args.accumulation_steps
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

        # Skip already completed epochs if resuming
        for _ in range(start_epoch * len(train_loader) // args.accumulation_steps):
            scheduler.step()

    for epoch in range(max(start_epoch, 0), args.stage1_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 args.accumulation_steps, ema, epoch, args.stage1_epochs,
                                 args.deep_supervision, max_grad_norm=args.grad_clip)

        for _ in range(len(train_loader) // args.accumulation_steps):
            scheduler.step()

        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        if ema:
            ema.restore()

        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow if ema else None,
                'best_iou': best_iou,
                'args': vars(args)
            }, f"{args.checkpoint_dir}/best_model.pth")
            print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'stage': 1, 'train_loss': train_loss, **val_metrics})

    print(f"\n✓ Stage 1 Complete. Best IoU: {best_iou:.4f}\n")

    # Stage 2
    if start_epoch >= args.stage1_epochs or args.resume:
        print("\n🧹 Cleaning up memory before Stage 2...")
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved\n")

        print("=" * 70)
        print("STAGE 2: FULL FINE-TUNING")
        print("=" * 70)
        print("🔓 Unfreezing all parameters")

        for param in model.parameters():
            param.requires_grad = True

        total, trainable = count_parameters(model)
        print(f"   Trainable parameters: {trainable / 1e6:.1f}M\n")

        # Use lower learning rate for Stage 2 to prevent gradient explosion
        # Reduce max_lr by 50% for stability
        stage2_lr = args.lr * 0.5
        stage2_backbone_lr = stage2_lr * 0.1

        print(f"GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved")
        print(f"Stage 2 Learning Rate: {stage2_lr} (50% of Stage 1 for stability)")
        print(f"Backbone Learning Rate: {stage2_backbone_lr}\n")

        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': stage2_backbone_lr},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
        ], weight_decay=args.weight_decay)

        total_steps = len(train_loader) * (args.epochs - args.stage1_epochs) // args.accumulation_steps
        scheduler = OneCycleLR(optimizer, max_lr=stage2_lr, total_steps=total_steps, pct_start=0.1)

        # More conservative gradient scaler for Stage 2
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2.**10,  # Lower initial scale
            growth_factor=1.5,   # Slower growth
            backoff_factor=0.5,  # Faster backoff on overflow
            growth_interval=1000 # Less frequent scaling increases
        )

        # Skip already completed epochs if resuming
        if start_epoch > args.stage1_epochs:
            epochs_done = (start_epoch - args.stage1_epochs) * len(train_loader) // args.accumulation_steps
            for _ in range(epochs_done):
                scheduler.step()
            print(f"   Skipped {start_epoch - args.stage1_epochs} completed epochs\n")

    for epoch in range(max(start_epoch, args.stage1_epochs), args.epochs):
        # Save checkpoint before each epoch for recovery
        checkpoint_path = f"{args.checkpoint_dir}/epoch_{epoch}_checkpoint.pth"

        try:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                     args.accumulation_steps, ema, epoch, args.epochs,
                                     args.deep_supervision, max_grad_norm=args.grad_clip)

            for _ in range(len(train_loader) // args.accumulation_steps):
                scheduler.step()

            if ema:
                ema.apply_shadow()
            val_metrics = validate(model, val_loader, metrics)
            if ema:
                ema.restore()

            print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_state_dict': ema.shadow if ema else None,
                    'best_iou': best_iou,
                    'args': vars(args)
                }, f"{args.checkpoint_dir}/best_model.pth")
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

            # Save latest checkpoint (for recovery)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'ema_state_dict': ema.shadow if ema else None,
                'best_iou': best_iou,
                'args': vars(args)
            }, checkpoint_path)

            history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, **val_metrics})

            # Clean up old checkpoints (keep only last 3)
            checkpoints = sorted(glob.glob(f"{args.checkpoint_dir}/epoch_*_checkpoint.pth"))
            for old_ckpt in checkpoints[:-3]:
                os.remove(old_ckpt)

        except ValueError as e:
            if "NaN/Inf" in str(e):
                print(f"\n⚠️  Gradient explosion detected at epoch {epoch}!")
                print(f"💡 Recovery suggestions:")
                print(f"   1. Restart training from: {args.checkpoint_dir}/best_model.pth")
                print(f"   2. Use lower learning rate: --lr {args.lr * 0.5}")
                print(f"   3. Reduce batch size or use smaller model")
                print(f"\n📊 Training history saved to: {args.checkpoint_dir}/history.json")

                # Save what we have so far
                with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
                    json.dump(history, f, indent=2)
                raise
            else:
                raise

    with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Target:   0.72")
    print(f"Status:   {'✅ SOTA!' if best_iou >= 0.72 else f'Gap: {0.72 - best_iou:.4f}'}")
    print("=" * 70)


if __name__ == '__main__':
    args = parse_args()
    if args.command == 'train':
        train(args)