# MoE System Verification Guide

## Overview

The CamoXpert model uses a **Mixture of Experts (MoE)** system with **content-aware routing** to select the best experts for each image. This document explains how the MoE system works and how to verify it's functioning correctly.

---

## How MoE Works in CamoXpert

### 1. **Content-Aware Routing**

The `ContentAwareGate` network analyzes image features from three perspectives:

```
Image Features → ContentAwareGate → Expert Selection Scores
                     ↓
              [Spatial Branch]  ← Detects edges, textures (local patterns)
              [Channel Branch]  ← Detects semantics (global context)
              [Max Branch]      ← Detects salient features (peaks)
                     ↓
                  Fusion → Gate Logits [B, 7]
```

**Example**: An image with strong edges will produce high scores for `EdgeExpert`, while a texture-rich image will favor `TextureExpert`.

### 2. **Top-K Expert Selection**

For each image, the router selects the **top-3 best experts** based on gate scores:

```python
gate_logits = gate(features)          # [B, 7] - scores for all 7 experts
top_k_logits, top_k_indices = torch.topk(gate_logits, k=3)  # Select top-3
top_k_weights = F.softmax(top_k_logits)  # Convert to weights [B, 3]
```

**Key properties**:
- Each image gets its own top-3 experts (content-aware)
- Weights sum to 1.0 per image
- Experts are selected based on learned patterns, not random

### 3. **Expert Specialization**

CamoXpert has **7 specialized experts**:

| Expert | Specialization | When Selected |
|--------|---------------|---------------|
| **TextureExpert** | Fine-grained textures | Camouflaged objects with texture patterns |
| **AttentionExpert** | Spatial attention | Objects requiring focus refinement |
| **EdgeExpert** | Boundary detection | Clear object boundaries |
| **FrequencyExpert** | Multi-scale patterns | Objects with frequency characteristics |
| **HybridExpert** | Combined approaches | Complex scenes |
| **SemanticContextExpert** | Global context | Scene understanding |
| **ContrastExpert** | Low-contrast objects | Subtle camouflage |

### 4. **Weighted Output Combination**

Expert outputs are combined using learned weights:

```python
output = w1 * Expert1(x) + w2 * Expert2(x) + w3 * Expert3(x)
```

where `w1 + w2 + w3 = 1.0` and weights are learned per-image.

---

## Verification Tools

### Built-in Debug Mode (Always Active)

The MoE system includes **runtime verification checks** enabled by default:

```python
# In models/experts.py - MoELayer.forward()
if self.debug_mode:  # True by default
    # Check 1: Gate outputs are valid
    assert not torch.isnan(gate_logits).any()
    assert not torch.isinf(gate_logits).any()

    # Check 2: Weights sum to 1.0
    assert torch.allclose(weights.sum(dim=1), torch.ones(B))

    # Check 3: Weights are non-negative
    assert (weights >= 0).all()

    # Check 4: Output is valid
    assert not torch.isnan(output).any()
    assert output.abs().sum() > 1e-6  # Not all zeros
```

**What this catches**:
- ❌ NaN or Inf values (numerical instability)
- ❌ Invalid expert indices
- ❌ Weights that don't sum to 1.0
- ❌ All-zero outputs (experts not firing)

If training crashes with an assertion error, it means the MoE system has a bug.

### Comprehensive Verification Script

Run the standalone verification script to thoroughly test the MoE system:

```bash
cd /home/user/camoXpert

# Test with randomly initialized model (architecture check)
python scripts/verify_moe.py \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --backbone edgenext_base \
    --num-experts 7 \
    --img-size 320 \
    --batch-size 4

# Test with trained checkpoint (routing quality check)
python scripts/verify_moe.py \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint /kaggle/working/checkpoints_sota/best_model.pth \
    --backbone edgenext_base \
    --num-experts 7 \
    --img-size 320 \
    --batch-size 4
```

**What it verifies**:

1. **[1/6] Routing Logic**
   - ✅ Top-3 selection works correctly
   - ✅ Indices are in valid range [0, 6]
   - ✅ No duplicate experts selected per image

2. **[2/6] Weight Properties**
   - ✅ Weights sum to 1.0 (within tolerance)
   - ✅ Weights are non-negative
   - ✅ Weights show diversity (not all equal)

3. **[3/6] Output Combination**
   - ✅ Output shapes match input shapes
   - ✅ Outputs are non-zero
   - ✅ Output magnitudes are reasonable

4. **[4/6] Numerical Stability**
   - ✅ No NaN or Inf in predictions
   - ✅ No extreme values (>1e6)

5. **[5/6] Content Awareness**
   - ✅ Router makes diverse decisions (not stuck on same experts)
   - ✅ Different images get different expert selections
   - 📊 Reports diversity ratio (higher = more content-aware)

6. **[6/6] Expert Specialization**
   - ✅ All experts are used (not completely ignored)
   - 📊 Shows expert usage distribution
   - ⚠️ Warns if one expert dominates (>80%)

**Example output**:

```
🔍 MoE SYSTEM VERIFICATION
======================================================================

[1/6] Verifying routing logic...
  ✅ Routing logic verified: Top-3 selection working correctly

[2/6] Checking weight properties...
  ✅ Weight properties verified: Sum to 1.0, non-negative

[3/6] Verifying output combination...
  ✅ Output combination verified: Shapes correct, non-zero outputs

[4/6] Checking for numerical issues...
  ✅ Numerical stability verified: No NaN/Inf detected

[5/6] Verifying content-aware routing...
  📊 Routing diversity: 127 unique combinations out of 240 decisions
     Diversity ratio: 52.92%
  ✅ High diversity - router is content-aware

[6/6] Analyzing expert specialization...

  📊 Expert Selection Frequency:
     TextureExpert            :   142 ( 19.7%) █████████
     AttentionExpert          :   128 ( 17.8%) ████████
     EdgeExpert               :   119 ( 16.5%) ████████
     HybridExpert             :   113 ( 15.7%) ███████
     SemanticContextExpert    :   105 ( 14.6%) ███████
     FrequencyExpert          :    97 ( 13.5%) ██████
     ContrastExpert           :    16 (  2.2%) █

  📈 Load balance: Min=2.2%, Max=19.7%
  ✅ Experts show specialization with balanced load

======================================================================
📋 VERIFICATION SUMMARY
======================================================================

✓ All verification tests completed!

Key findings:
  • Routing diversity: 52.9% (higher is better)
  • Average weight std: 0.1245 (>0.05 indicates learning)
  • Average output magnitude: 0.3421

======================================================================
✅ MoE system verification complete!
======================================================================
```

---

## What to Look For During Training

### 1. Expert Usage Statistics

The training script prints expert usage every epoch:

```
📊 Expert Usage Statistics:
  TextureExpert            :   234 ( 21.2%) ██████████
  AttentionExpert          :   198 ( 17.9%) ████████
  EdgeExpert               :   187 ( 16.9%) ████████
  HybridExpert             :   165 ( 14.9%) ███████
  ...
```

**Good signs**:
- ✅ All experts are used (>1% each)
- ✅ Usage varies between experts (specialization)
- ✅ No single expert dominates (>80%)

**Bad signs**:
- ⚠️ One expert has 0% usage → expert may be broken
- ⚠️ One expert has >80% usage → router not learning
- ⚠️ All experts have ~14% usage → perfectly uniform (router may be ignoring content)

### 2. Training Behavior

**Router is learning correctly if**:
- IoU improves steadily (e.g., 0.15 → 0.30 → 0.45 → ...)
- Loss decreases consistently
- Expert usage distribution changes over epochs (router adapting)

**Router is NOT learning if**:
- IoU stays at 0.0000 despite loss decreasing
- Expert usage stays perfectly uniform (14.3% each)
- Training crashes with assertion errors

---

## Troubleshooting

### Issue: Assertion Error During Training

**Error**: `AssertionError: ❌ Weights don't sum to 1.0`

**Cause**: Numerical instability in softmax or gate network

**Fix**:
1. Reduce learning rate (too high LR causes instability)
2. Check for NaN gradients (`torch.isnan(model.parameters())`)
3. Enable gradient clipping (already enabled in train_ultimate.py)

---

### Issue: One Expert Dominates (>80%)

**Symptom**:
```
TextureExpert: 850 (85.2%) ████████████████████
Others:        148 (14.8%)
```

**Cause**: Load balancing loss too weak (0.001)

**Fix**: Increase load balancing weight in `models/experts.py:391`:
```python
# Change from:
aux_loss = F.mse_loss(expert_freq, target_freq) * 0.001

# To:
aux_loss = F.mse_loss(expert_freq, target_freq) * 0.01
```

---

### Issue: All Experts Have Same Usage (~14.3%)

**Symptom**:
```
All experts:  142-145 (14.0-14.5%) ███████
```

**Cause**: Load balancing loss too strong OR gate not learning

**Fix**:
1. **Check if router is learning**: Print gate weight gradients
2. **Reduce load balancing**: Change `0.001` → `0.0001`
3. **Check gate learning rate**: Ensure gate network has sufficient LR

---

### Issue: IoU = 0.0000 Despite Training

**Symptom**: Loss decreasing but IoU stays at zero

**Cause**: Learning rate too high, model diverging

**Fix**: Reduce learning rate:
```bash
--lr 0.0001  # Instead of 0.00025
```

---

## Disabling Debug Mode

If debug checks slow down training (unlikely), disable them:

```python
# In models/camoxpert.py, when creating MoELayers:
self.moe_layers = nn.ModuleList([
    MoELayer(dim, num_experts=num_experts, top_k=top_k, debug_mode=False)
    for dim in self.feature_dims
])
```

**Warning**: Only disable if training speed is critical and MoE is verified working.

---

## Summary

The MoE system in CamoXpert:
1. ✅ Uses content-aware routing (analyzes features, not random)
2. ✅ Selects top-3 experts per image dynamically
3. ✅ Combines outputs with learned weights (sum to 1.0)
4. ✅ Has 7 specialized experts for different image characteristics
5. ✅ Includes runtime verification (catches bugs immediately)
6. ✅ Can be thoroughly tested with verification script

**To verify your model is working**:
```bash
# Run during or after training
python scripts/verify_moe.py \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint /kaggle/working/checkpoints_sota/best_model.pth \
    --num-experts 7 \
    --img-size 320
```

Look for:
- ✅ High routing diversity (>50%)
- ✅ Balanced expert usage (no expert <1% or >80%)
- ✅ All verification tests pass
