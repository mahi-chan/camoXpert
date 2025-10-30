"""
Command-line training script with gradient accumulation & checkpointing
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
from tqdm import tqdm
import numpy as np
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert, count_parameters
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='CamoXpert SOTA Training with Memory Optimization')

    parser.add_argument('command', type=str, choices=['train'])
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--backbone', type=str, default='edgenext_base')
    parser.add_argument('--num-experts', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--stage1-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
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


class ExpertUsageTracker:
    """Track which experts are being selected for images"""
    def __init__(self):
        self.expert_usage = defaultdict(int)
        self.total_selections = 0

    def update(self, routing_info):
        """Update with routing info from MoE layer"""
        if routing_info and 'expert_counts' in routing_info:
            counts = routing_info['expert_counts'].cpu().numpy()
            names = routing_info.get('expert_names', [f'Expert_{i}' for i in range(len(counts))])
            for name, count in zip(names, counts):
                self.expert_usage[name] += int(count)
                self.total_selections += int(count)

    def get_stats(self):
        """Get expert usage statistics"""
        if self.total_selections == 0:
            return {}
        stats = {}
        for name, count in self.expert_usage.items():
            stats[name] = {
                'count': count,
                'percentage': 100 * count / self.total_selections
            }
        return stats

    def print_stats(self):
        """Print expert usage statistics"""
        stats = self.get_stats()
        if not stats:
            return
        print("\n📊 Expert Usage Statistics:")
        sorted_experts = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for name, data in sorted_experts:
            bar = '█' * int(data['percentage'] / 2)
            print(f"  {name:25s}: {data['count']:5d} ({data['percentage']:5.1f}%) {bar}")

    def reset(self):
        """Reset statistics"""
        self.expert_usage.clear()
        self.total_selections = 0


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for MoE layers (memory-intensive)"""
    checkpointed = 0

    for name, module in model.named_modules():
        if 'moe' in name.lower() and hasattr(module, '__class__'):
            if module.__class__.__name__ == 'MoELayer':
                original_forward = module.forward

                def create_checkpoint_wrapper(orig_forward):
                    def wrapper(x):
                        return gradient_checkpoint(orig_forward, x, use_reentrant=False)
                    return wrapper

                module.forward = create_checkpoint_wrapper(original_forward)
                checkpointed += 1

    print(f"✓ Gradient checkpointing enabled ({checkpointed} MoE layers)")
    return model


def train_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps, ema, epoch, total_epochs,
                use_deep_sup, expert_tracker=None):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, (images, masks) in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            pred, aux_loss, deep = model(images, return_deep_supervision=use_deep_sup)
            loss, _ = criterion(pred, masks, aux_loss, deep)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

        epoch_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics, expert_tracker=None):
    model.eval()
    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.cuda(), masks.cuda()
        pred, _, _ = model(images)
        pred = torch.sigmoid(pred)
        all_metrics.append(metrics.compute_all(pred, masks))
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


@torch.no_grad()
def analyze_expert_routing(model, loader, num_batches=10):
    """Analyze which experts are being selected for validation images"""
    model.eval()
    tracker = ExpertUsageTracker()

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        images = images.cuda()

        # Forward pass through model to collect routing info
        # We need to access MoE layers directly
        features = model.backbone(images)
        for feat, sdta, moe in zip(features, model.sdta_blocks, model.moe_layers):
            feat = sdta(feat)
            _, _, routing_info = moe(feat)
            tracker.update(routing_info)

    tracker.print_stats()
    return tracker


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    effective_batch = args.batch_size * args.accumulation_steps

    print(f"\n{'='*70}")
    print(f"CamoXpert Training: {args.backbone} | {args.num_experts} experts | {args.img_size}px")
    print(f"Batch: {args.batch_size}×{args.accumulation_steps}={effective_batch} | Epochs: {args.epochs} | Target IoU: 0.72")
    print(f"{'='*70}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Datasets
    train_data = COD10KDataset(args.dataset_path, 'train', args.img_size, augment=True)
    val_data = COD10KDataset(args.dataset_path, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Model
    model = CamoXpert(3, 1, pretrained=True, backbone=args.backbone, num_experts=args.num_experts).cuda()

    if args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)

    total, trainable = count_parameters(model)
    print(f"Parameters: {total/1e6:.1f}M ({trainable/1e6:.1f}M trainable)\n")

    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)
    metrics = CODMetrics()
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model) if args.use_ema else None

    best_iou = 0.0
    history = []

    # Stage 1: Decoder Training
    print(f"\n{'='*70}")
    print("STAGE 1: Decoder Training (backbone frozen)")
    print(f"{'='*70}")

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.stage1_epochs // args.accumulation_steps
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    for epoch in range(args.stage1_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 args.accumulation_steps, ema, epoch, args.stage1_epochs, args.deep_supervision)

        for _ in range(len(train_loader) // args.accumulation_steps):
            scheduler.step()

        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        if ema:
            ema.restore()

        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

        # Analyze expert routing every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Expert Routing Analysis (Epoch {epoch + 1})")
            print(f"{'='*70}")
            analyze_expert_routing(model, val_loader, num_batches=20)
            print(f"{'='*70}\n")

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

    print(f"\n✓ Stage 1 complete | Best IoU: {best_iou:.4f}\n")

    # Stage 2: Full Fine-tuning
    print(f"{'='*70}")
    print("STAGE 2: Full Fine-tuning (end-to-end)")
    print(f"{'='*70}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * (args.epochs - args.stage1_epochs) // args.accumulation_steps
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    for epoch in range(args.stage1_epochs, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 args.accumulation_steps, ema, epoch, args.epochs, args.deep_supervision)

        for _ in range(len(train_loader) // args.accumulation_steps):
            scheduler.step()

        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        if ema:
            ema.restore()

        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

        # Analyze expert routing every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Expert Routing Analysis (Epoch {epoch + 1})")
            print(f"{'='*70}")
            analyze_expert_routing(model, val_loader, num_batches=20)
            print(f"{'='*70}\n")

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

        history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, **val_metrics})

    with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training Complete | Best IoU: {best_iou:.4f} | Target: 0.72")
    print(f"Status: {'✅ SOTA Achieved!' if best_iou >= 0.72 else f'❌ Gap: {0.72 - best_iou:.4f}'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    args = parse_args()
    if args.command == 'train':
        train(args)