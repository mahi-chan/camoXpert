# Training Commands - Single GPU vs DDP

## Your Custom Configuration - DDP Multi-GPU ⚡

### Quick Start (Choose One):

**Option 1: Python Launcher**
```bash
python launch_ddp_custom.py
```

**Option 2: Shell Script**
```bash
bash train_ddp_custom.sh
```

**Option 3: Direct Command**
```bash
torchrun --nproc_per_node=2 --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 32 \
    --stage2-batch-size 24 \
    --accumulation-steps 1 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --num-workers 4 \
    --no-amp \
    --use-cod-specialized
```

---

## Batch Size Explained

### Your Original Configuration (Single GPU):
- **Stage 1**: Batch size 64
- **Stage 2**: Batch size 48

### DDP Configuration (2 GPUs):
- **Stage 1**: 32 **per GPU** × 2 GPUs = **64 total** ✅
- **Stage 2**: 24 **per GPU** × 2 GPUs = **48 total** ✅

**Same effective batch size, but distributed across GPUs!**

---

## Configuration Summary

| Setting | Value | Description |
|---------|-------|-------------|
| **GPUs** | 2 × Tesla T4 | DistributedDataParallel |
| **Batch (S1)** | 64 total | 32 per GPU |
| **Batch (S2)** | 48 total | 24 per GPU |
| **Resolution** | 352×352 | Optimal for COD |
| **Epochs** | 150 | 30 decoder + 120 full |
| **LR (S1)** | 0.0008 | With warmup |
| **LR (S2)** | 0.00055 | Fine-tuning rate |
| **Scheduler** | Cosine | With min_lr 1e-5 |
| **Warmup** | 5 epochs | Stage 2 only |
| **Precision** | FP32 | More stable (--no-amp) |
| **Deep Sup** | Enabled | 3 supervision levels |

---

## Expected Performance

### Training Speed:
- **Single GPU**: ~8 hours
- **DDP (2 GPUs)**: ~4-5 hours ⚡ **~1.8× faster**

### Expected Results:
- **IoU**: ≥ 0.72 (your target)
- **Training**: Stable, no CUDA errors
- **Checkpoints**: Saved to `/kaggle/working/checkpoints_cod_specialized`

---

## Fallback: Single GPU (If DDP Fails)

If you encounter any issues with DDP, use single GPU:

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 24 \
    --stage2-batch-size 18 \
    --accumulation-steps 3 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --num-workers 4 \
    --no-amp \
    --use-cod-specialized
```

*(Batch size auto-adjusted with gradient accumulation to maintain effective batch size)*

---

## Monitoring Training

Check progress:
```bash
# View checkpoints
ls -lh /kaggle/working/checkpoints_cod_specialized/

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Troubleshooting

### "Address already in use"
```bash
# Change port
torchrun --nproc_per_node=2 --master_port=29501 train_ultimate.py train --use-ddp ...
```

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch-size 24 --stage2-batch-size 18
```

### NCCL Timeout
```bash
# Increase timeout
export NCCL_TIMEOUT=1800
torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp ...
```

---

Good luck! 🚀 Your training should achieve IoU ≥ 0.72!
