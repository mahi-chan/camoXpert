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
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
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
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (default: 0.999, use 0.9999 for very stable training)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage2-batch-size', type=int, default=None,
                        help='Batch size for stage 2 (default: same as --batch-size)')
    parser.add_argument('--progressive-unfreeze', action='store_true', default=False,
                        help='Gradually unfreeze backbone layers in stage 2')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--skip-stage1', action='store_true', default=False,
                        help='Skip stage 1 and go directly to stage 2 (use with --resume-from)')
    parser.add_argument('--stage2-lr', type=float, default=None,
                        help='Learning rate for Stage 2 (default: same as --lr)')
    parser.add_argument('--scheduler', type=str, choices=['onecycle', 'cosine', 'cosine_restart', 'none'],
                        default='onecycle', help='Learning rate scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine schedulers')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs for Stage 2')
    parser.add_argument('--t-mult', type=int, default=2,
                        help='T_mult for cosine_restart scheduler')

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


def clear_gpu_memory():
    """Clear GPU memory cache and collect garbage"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved")


def enable_gradient_checkpointing(model):
    print("🔧 Enabling gradient checkpointing...")
    checkpointed = 0

    def make_checkpointed_forward(original_forward):
        """Factory function to create checkpointed forward with proper closure"""
        def checkpointed_forward(self, *args, **kwargs):
            def custom_forward(*inputs):
                return original_forward(*inputs, **kwargs)

            return gradient_checkpoint(custom_forward, *args, use_reentrant=False)

        return checkpointed_forward

    for name, module in model.named_modules():
        if 'moe' in name.lower() or 'sdta' in name.lower():
            if hasattr(module, 'forward'):
                original_forward = module.forward
                module.forward = make_checkpointed_forward(original_forward).__get__(module, type(module))
                checkpointed += 1

    print(f"✓ Checkpointed {checkpointed} modules")
    return model


def progressive_unfreeze_backbone(model, stage):
    """
    Progressively unfreeze backbone layers
    stage 0: freeze all
    stage 1: unfreeze last layer
    stage 2: unfreeze last 2 layers
    stage 3: unfreeze all
    """
    # First freeze everything
    for param in model.backbone.parameters():
        param.requires_grad = False

    if stage == 0:
        return

    # Get backbone layers
    backbone_children = list(model.backbone.children())
    num_layers = len(backbone_children)

    if stage >= 3:
        # Unfreeze all
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone: All layers unfrozen")
    else:
        # Unfreeze last 'stage' layers
        layers_to_unfreeze = backbone_children[-stage:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"✓ Backbone: Last {stage}/{num_layers} layers unfrozen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable/1e6:.1f}M")


def create_scheduler(optimizer, scheduler_type, total_steps=None, total_epochs=None,
                    max_lr=None, min_lr=1e-6, warmup_epochs=0, t_mult=2):
    """Create learning rate scheduler"""

    if scheduler_type == 'onecycle':
        if total_steps is None:
            raise ValueError("total_steps required for onecycle scheduler")
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.1)

    elif scheduler_type == 'cosine':
        if total_epochs is None:
            raise ValueError("total_epochs required for cosine scheduler")
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    elif scheduler_type == 'cosine_restart':
        if total_epochs is None:
            raise ValueError("total_epochs required for cosine_restart scheduler")
        return CosineAnnealingWarmRestarts(optimizer, T_0=total_epochs//4, T_mult=t_mult, eta_min=min_lr)

    else:  # 'none'
        return None


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Apply learning rate warmup"""
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor
        return True
    return False


def train_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps, ema, epoch, total_epochs,
                use_deep_sup):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, (images, masks) in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            pred, aux_loss, deep = model(images, return_deep_supervision=use_deep_sup)
            loss, _ = criterion(pred, masks, aux_loss, deep)
            loss = loss / accumulation_steps

        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            print(f"\n❌ NaN/Inf detected in loss at batch {batch_idx}!")
            print(f"   Loss value: {loss.item()}")
            print(f"   Skipping this batch and stopping training for safety.")
            print(f"   Please reduce learning rate and restart from last good checkpoint.")
            raise ValueError("Training stopped due to NaN/Inf loss. See error message above.")

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Check for NaN/Inf in gradients
            if not torch.isfinite(grad_norm):
                print(f"\n❌ NaN/Inf detected in gradients at batch {batch_idx}!")
                print(f"   Gradient norm: {grad_norm.item()}")
                print(f"   This indicates gradient explosion.")
                print(f"   Please reduce learning rate and restart from last good checkpoint.")
                raise ValueError("Training stopped due to NaN/Inf gradients. See error message above.")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

        epoch_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Check for NaN/Inf in final gradients
        if not torch.isfinite(grad_norm):
            print(f"\n❌ NaN/Inf detected in final gradients!")
            print(f"   Gradient norm: {grad_norm.item()}")
            raise ValueError("Training stopped due to NaN/Inf gradients in final step.")

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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
    # Enable PyTorch memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    set_seed(args.seed)
    device = torch.device(args.device)

    # Set stage 2 batch size (default to half of stage 1 for memory efficiency)
    if args.stage2_batch_size is None:
        args.stage2_batch_size = max(1, args.batch_size // 2)

    effective_batch_s1 = args.batch_size * args.accumulation_steps
    effective_batch_s2 = args.stage2_batch_size * args.accumulation_steps

    print("\n" + "=" * 70)
    print("CAMOXPERT ULTIMATE TRAINING")
    print("=" * 70)
    print(f"Backbone:         {args.backbone}")
    print(f"Experts:          {args.num_experts}")
    print(f"Resolution:       {args.img_size}px")
    print(f"Stage 1 Batch:    {args.batch_size} × {args.accumulation_steps} = {effective_batch_s1} effective")
    print(f"Stage 2 Batch:    {args.stage2_batch_size} × {args.accumulation_steps} = {effective_batch_s2} effective")
    print(f"Epochs:           {args.epochs}")
    print(f"Deep Supervision: {args.deep_supervision}")
    print(f"Grad Checkpoint:  {args.gradient_checkpointing}")
    print(f"Progressive:      {args.progressive_unfreeze}")
    print(f"EMA:              {args.use_ema}")
    print(f"\n🎯 Target: IoU ≥ 0.72")
    print("=" * 70 + "\n")

    print_gpu_memory()

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
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.use_ema:
        print(f"✨ EMA enabled with decay: {args.ema_decay}")

    best_iou = 0.0
    history = []
    start_epoch = 0

    # Load checkpoint if resuming
    if args.resume_from:
        print(f"\n{'='*70}")
        print(f"LOADING CHECKPOINT: {args.resume_from}")
        print(f"{'='*70}")
        if not os.path.exists(args.resume_from):
            print(f"❌ ERROR: Checkpoint not found at {args.resume_from}")
            return

        checkpoint = torch.load(args.resume_from, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if ema and checkpoint.get('ema_state_dict'):
            ema.shadow = checkpoint['ema_state_dict']

        best_iou = checkpoint.get('best_iou', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"✓ Best IoU so far: {best_iou:.4f}")
        print(f"✓ Resuming from epoch {start_epoch}")

        if args.skip_stage1 and start_epoch < args.stage1_epochs:
            print(f"⚠️  WARNING: Checkpoint is from epoch {checkpoint.get('epoch', 0)} (Stage 1)")
            print(f"   You requested --skip-stage1, jumping to epoch {args.stage1_epochs}")
            start_epoch = args.stage1_epochs

        print(f"{'='*70}\n")

    # Stage 1
    if start_epoch < args.stage1_epochs:
        print("=" * 70)
        print("STAGE 1: DECODER TRAINING")
        print("=" * 70)

        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

        # Create scheduler for Stage 1
        total_steps = len(train_loader) * args.stage1_epochs // args.accumulation_steps
        total_epochs = args.stage1_epochs
        scheduler = create_scheduler(optimizer, args.scheduler,
                                     total_steps=total_steps,
                                     total_epochs=total_epochs,
                                     max_lr=args.lr,
                                     min_lr=args.min_lr,
                                     t_mult=args.t_mult)

        for epoch in range(start_epoch, args.stage1_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                     args.accumulation_steps, ema, epoch, args.stage1_epochs, args.deep_supervision)

            # Step scheduler if it exists
            if scheduler is not None:
                if args.scheduler == 'onecycle':
                    for _ in range(len(train_loader) // args.accumulation_steps):
                        scheduler.step()
                else:
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
    else:
        print(f"\n⏩ Skipping Stage 1 (resuming from epoch {start_epoch})\n")

    # ========================================
    # MEMORY CLEANUP BEFORE STAGE 2
    # ========================================
    if start_epoch < args.stage1_epochs:
        # Only cleanup if we just finished Stage 1
        print("🧹 Cleaning up memory before Stage 2...")
        del optimizer, scheduler
        clear_gpu_memory()
        print_gpu_memory()

    # ========================================
    # Stage 2: Full Fine-tuning with reduced batch size
    # ========================================
    print("\n" + "=" * 70)
    print("STAGE 2: FULL FINE-TUNING")
    print("=" * 70)

    # Create new dataloader with reduced batch size for Stage 2
    if args.stage2_batch_size != args.batch_size:
        print(f"🔧 Reducing batch size: {args.batch_size} → {args.stage2_batch_size}")
        train_loader = DataLoader(train_data, args.stage2_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_data, args.stage2_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    if args.progressive_unfreeze:
        print("📈 Using progressive unfreezing strategy")
        progressive_unfreeze_backbone(model, stage=1)
    else:
        print("🔓 Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {trainable/1e6:.1f}M")

    print()
    print_gpu_memory()

    # Use stage2-specific LR if provided, otherwise use default
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr

    print(f"Stage 2 Learning Rate: {stage2_lr}")
    if args.warmup_epochs > 0:
        print(f"Warmup epochs: {args.warmup_epochs}")
    if args.scheduler != 'none':
        print(f"Scheduler: {args.scheduler}")
        if args.scheduler in ['cosine', 'cosine_restart']:
            print(f"  Min LR: {args.min_lr}")

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': stage2_lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
    ], weight_decay=args.weight_decay)

    # Create scheduler for Stage 2
    total_steps = len(train_loader) * (args.epochs - args.stage1_epochs) // args.accumulation_steps
    total_epochs = args.epochs - args.stage1_epochs
    scheduler = create_scheduler(optimizer, args.scheduler,
                                 total_steps=total_steps,
                                 total_epochs=total_epochs,
                                 max_lr=stage2_lr,
                                 min_lr=args.min_lr,
                                 t_mult=args.t_mult)

    stage2_start = max(start_epoch, args.stage1_epochs)
    if stage2_start > args.stage1_epochs:
        print(f"📍 Resuming Stage 2 from epoch {stage2_start}\n")

    for epoch in range(stage2_start, args.epochs):
        # Progressive unfreezing: gradually unfreeze more layers
        if args.progressive_unfreeze:
            stage2_progress = epoch - args.stage1_epochs
            total_stage2 = args.epochs - args.stage1_epochs
            if stage2_progress == total_stage2 // 3:
                print("\n📈 Progressive unfreeze: Stage 2/3")
                progressive_unfreeze_backbone(model, stage=2)
                # Recreate optimizer with new parameters
                optimizer = AdamW([
                    {'params': model.backbone.parameters(), 'lr': stage2_lr * 0.1},
                    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
                ], weight_decay=args.weight_decay)
                clear_gpu_memory()
                print_gpu_memory()
                print()
            elif stage2_progress == 2 * total_stage2 // 3:
                print("\n📈 Progressive unfreeze: Stage 3/3 (Full)")
                progressive_unfreeze_backbone(model, stage=3)
                # Recreate optimizer with new parameters
                optimizer = AdamW([
                    {'params': model.backbone.parameters(), 'lr': stage2_lr * 0.1},
                    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
                ], weight_decay=args.weight_decay)
                clear_gpu_memory()
                print_gpu_memory()
                print()

        # Apply warmup if in warmup period
        stage2_epoch = epoch - args.stage1_epochs
        in_warmup = warmup_lr(optimizer, stage2_epoch, args.warmup_epochs, stage2_lr)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 args.accumulation_steps, ema, epoch, args.epochs, args.deep_supervision)

        # Step scheduler if not in warmup and scheduler exists
        if not in_warmup and scheduler is not None:
            if args.scheduler == 'onecycle':
                for _ in range(len(train_loader) // args.accumulation_steps):
                    scheduler.step()
            else:
                scheduler.step()

        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        if ema:
            ema.restore()

        # Print training metrics with LR
        current_lr = optimizer.param_groups[0]['lr']
        lr_info = f" | LR: {current_lr:.6f}" if args.scheduler != 'none' or in_warmup else ""
        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}{lr_info}")

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