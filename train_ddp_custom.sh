#!/bin/bash
# High-Resolution DDP Training Script - Targeting IoU 0.75
# Run with: bash train_ddp_custom.sh

echo "=================================================="
echo "CAMOXPERT DDP TRAINING - TARGETING IoU 0.75"
echo "=================================================="
echo "GPUs: 2 × Tesla T4"
echo "Resolution: 416px (vs 352px baseline)"
echo "Total Batch: Stage 1 = 28, Stage 2 = 20"
echo "Effective Batch (w/ grad accumulation): 56 / 40"
echo "Mixed Precision: Enabled (AMP)"
echo "Gradient Checkpointing: Enabled"
echo "Epochs: 200 (40 Stage 1 + 160 Stage 2)"
echo "=================================================="
echo ""

torchrun --nproc_per_node=2 --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 14 \
    --stage2-batch-size 10 \
    --accumulation-steps 2 \
    --img-size 416 \
    --epochs 200 \
    --stage1-epochs 40 \
    --lr 0.0008 \
    --stage2-lr 0.0006 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --gradient-checkpointing \
    --num-workers 4 \
    --use-cod-specialized

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
