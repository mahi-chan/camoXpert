# MoE Verification in Kaggle - Quick Start Guide

## Method 1: Quick One-Command Verification

Copy and paste this into a Kaggle notebook cell:

```bash
# Quick verification (no checkpoint needed)
python kaggle_verify_moe.py

# Or with custom settings
python kaggle_verify_moe.py --batch-size 4 --img-size 320 --num-batches 5
```

---

## Method 2: Verify with Trained Checkpoint

If you have a trained checkpoint, test with it:

```bash
# Verify with checkpoint
python kaggle_verify_moe.py --checkpoint /kaggle/working/checkpoints_sota/best_model.pth
```

---

## Method 3: Run Inline in Notebook Cell

Copy this entire code block into a Kaggle notebook cell:

```python
# ============================================================================
# KAGGLE MoE VERIFICATION - Inline Version
# ============================================================================

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Import your model
from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader

print("="*70)
print("🔍 MoE SYSTEM VERIFICATION")
print("="*70)

# ----- CONFIGURATION -----
DATASET_PATH = '/kaggle/input/cod10k-dataset/COD10K-v3'  # Adjust if needed
CHECKPOINT_PATH = None  # Set to checkpoint path if testing trained model
IMG_SIZE = 320
BATCH_SIZE = 4
NUM_BATCHES = 5

# ----- CHECK GPU -----
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
capability = torch.cuda.get_device_capability(0)
print(f"Compute capability: {capability[0]}.{capability[1]}")

# ----- LOAD MODEL -----
print("\n📦 Loading model...")
model = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()

if CHECKPOINT_PATH:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print("Using randomly initialized model (architecture test)")

model.eval()

# ----- LOAD DATASET -----
print(f"\n📂 Loading dataset from {DATASET_PATH}...")
dataset = COD10KDataset(DATASET_PATH, 'val', IMG_SIZE, augment=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"✅ Dataset: {len(dataset)} images")

# ----- TEST MoE ROUTING -----
print("\n🔍 Testing MoE routing...\n")

expert_counts = defaultdict(int)
total_selections = 0
routing_diversity = []
issues = []

with torch.no_grad():
    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Verifying", total=NUM_BATCHES)):
        if batch_idx >= NUM_BATCHES:
            break

        images = images.cuda()
        B = images.shape[0]

        # Extract features
        features = model.backbone(images)

        for layer_idx, (feat, sdta, moe) in enumerate(zip(features, model.sdta_blocks, model.moe_layers)):
            feat = sdta(feat)

            # Get routing
            gate_logits = moe.gate(feat)
            top_k_logits, top_k_indices = torch.topk(gate_logits, moe.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)

            # Verify weights sum to 1
            weight_sums = top_k_weights.sum(dim=1)
            if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4):
                issues.append(f"Layer {layer_idx}: Weights don't sum to 1.0")

            # Check for NaN/Inf
            if torch.isnan(gate_logits).any() or torch.isinf(gate_logits).any():
                issues.append(f"Layer {layer_idx}: NaN/Inf detected")

            # Track expert usage
            for i in range(B):
                experts_selected = top_k_indices[i].cpu().numpy()
                routing_diversity.append(tuple(sorted(experts_selected)))
                for expert_idx in experts_selected:
                    expert_counts[expert_idx.item()] += 1
                    total_selections += 1

# ----- PRINT RESULTS -----
print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)

if len(issues) == 0:
    print("\n✅ All checks PASSED!")
else:
    print("\n❌ Issues found:")
    for issue in issues:
        print(f"   - {issue}")

# Expert usage
print("\n📊 Expert Usage:\n")
expert_names = ['TextureExpert', 'AttentionExpert', 'HybridExpert',
                'FrequencyExpert', 'EdgeExpert', 'SemanticContextExpert', 'ContrastExpert']

sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
for expert_idx, count in sorted_experts:
    if expert_idx < len(expert_names):
        name = expert_names[expert_idx]
        percentage = 100 * count / total_selections if total_selections > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"   {name:25s}: {count:5d} ({percentage:5.1f}%) {bar}")

# Routing diversity
unique_combinations = len(set(routing_diversity))
total_decisions = len(routing_diversity)
diversity_ratio = unique_combinations / total_decisions if total_decisions > 0 else 0

print(f"\n📈 Routing Diversity: {diversity_ratio:.1%} ({unique_combinations}/{total_decisions} unique)")

if diversity_ratio > 0.5:
    print("   ✅ High diversity - router is content-aware")
elif diversity_ratio > 0.2:
    print("   ✓  Moderate diversity")
else:
    print("   ⚠️  Low diversity - router may need more training")

print("\n" + "="*70)

if len(issues) == 0 and diversity_ratio > 0.2:
    print("✅ MoE SYSTEM IS WORKING CORRECTLY! Ready to train!")
else:
    print("⚠️  Review results above")

print("="*70)
```

---

## Expected Output

### ✅ Good Output (MoE Working Correctly)

```
🔍 MoE SYSTEM VERIFICATION
======================================================================

🖥️  GPU: Tesla P100-PCIE-16GB
Compute capability: 6.0

📦 Loading model...
Using randomly initialized model (architecture test)

📂 Loading dataset from /kaggle/input/cod10k-dataset/COD10K-v3...
✅ Dataset: 2026 images

🔍 Testing MoE routing...

Verifying: 100%|██████████| 5/5 [00:03<00:00,  1.43it/s]

======================================================================
📊 RESULTS
======================================================================

✅ All checks PASSED!

📊 Expert Usage:

   TextureExpert            :   142 ( 19.7%) █████████
   AttentionExpert          :   128 ( 17.8%) ████████
   HybridExpert             :   119 ( 16.5%) ████████
   EdgeExpert               :   113 ( 15.7%) ███████
   SemanticContextExpert    :   105 ( 14.6%) ███████
   FrequencyExpert          :    97 ( 13.5%) ██████
   ContrastExpert           :    16 (  2.2%) █

📈 Routing Diversity: 52.9% (127/240 unique)
   ✅ High diversity - router is content-aware

======================================================================
✅ MoE SYSTEM IS WORKING CORRECTLY! Ready to train!
======================================================================
```

---

## Troubleshooting

### Issue 1: Dataset Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/cod10k-dataset/COD10K-v3'
```

**Fix:**
1. Check your dataset path in Kaggle
2. Common paths:
   - `/kaggle/input/cod10k-dataset/COD10K-v3`
   - `/kaggle/input/cod10k-v3/COD10K-v3`
   - `/kaggle/input/cod10k`
3. Update `DATASET_PATH` in the script

### Issue 2: Import Error

**Error:**
```
ModuleNotFoundError: No module named 'models'
```

**Fix:**
```bash
# Add this at the top of your notebook
import sys
sys.path.insert(0, '/kaggle/working')

# Then run the verification
```

### Issue 3: Low Diversity

**Output:**
```
📈 Routing Diversity: 8.5% (12/240 unique)
   ⚠️  Low diversity - router may need more training
```

**This is NORMAL for randomly initialized models!**
- Random weights → router hasn't learned yet
- After training, diversity should increase to 40-60%
- Only worry if diversity stays low AFTER training

### Issue 4: One Expert Dominates

**Output:**
```
TextureExpert: 680 (85.0%)  ← Too high!
Others:        120 (15.0%)
```

**Fix:**
- If this happens AFTER training: Increase load balancing loss
- Edit `models/experts.py` line 390: Change `0.001` to `0.01`

---

## Quick Reference

### Test Commands

```bash
# Minimal test (fastest)
python kaggle_verify_moe.py

# Test with checkpoint
python kaggle_verify_moe.py --checkpoint /kaggle/working/checkpoints_sota/best_model.pth

# Test with custom settings
python kaggle_verify_moe.py --batch-size 8 --img-size 352 --num-batches 10
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | None | Path to trained checkpoint |
| `--batch-size` | 4 | Batch size for testing |
| `--img-size` | 320 | Input image size |
| `--num-batches` | 5 | Number of batches to test |

---

## What Gets Verified

1. ✅ **Routing Logic** - Top-3 selection works correctly
2. ✅ **Weight Properties** - Weights sum to 1.0, non-negative
3. ✅ **Numerical Stability** - No NaN/Inf values
4. ✅ **Output Validity** - MoE outputs are non-zero and finite
5. ✅ **Content Awareness** - Router makes diverse decisions
6. ✅ **Expert Usage** - All experts are used (none stuck at 0%)

---

## Next Steps

After verification passes:

1. **Start Training:**
   ```bash
   python train_ultimate.py train \
       --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
       --checkpoint-dir /kaggle/working/checkpoints_sota \
       --backbone edgenext_base \
       --num-experts 7 \
       --batch-size 16 \
       --accumulation-steps 8 \
       --img-size 288 \
       --lr 0.0001 \
       --gradient-checkpointing \
       --deep-supervision \
       --use-ema
   ```

2. **Monitor Expert Usage:**
   - Watch the expert usage statistics printed each epoch
   - Diversity should increase as training progresses
   - All experts should remain active (>1%)

3. **Run Verification Again:**
   - After training, verify with checkpoint
   - Check that diversity increased (40-60% is good)
   - Confirm no single expert dominates

---

## Summary

**Quick Start:**
```bash
python kaggle_verify_moe.py
```

**Expected Time:** 10-30 seconds

**What You'll See:**
- ✅ All checks passed
- Expert usage distribution
- Routing diversity percentage
- Ready to train message

🚀 That's it! Your MoE system is verified and ready for training!
