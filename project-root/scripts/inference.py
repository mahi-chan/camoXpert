# inference.py

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from models.utils import load_checkpoint
from models.segmentation_head import SegmentationHead
from models.fusion import BiLevelFusion
from models.utils import LayerNorm2d
from models.backbone import EdgeNeXtBackbone
from models.moe import CamoXpert  # Assuming CamoXpert is implemented in models.moe

def preprocess_image(image_path, img_size):
    """
    Preprocess the input image for inference.

    Args:
        image_path (str): Path to the input image.
        img_size (int): Target image size.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0  # Normalize to [0, 1]
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # ImageNet normalization
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

def postprocess_mask(mask, original_size):
    """
    Postprocess the output mask.

    Args:
        mask (torch.Tensor): Predicted mask tensor.
        original_size (tuple): Original image size (height, width).

    Returns:
        np.ndarray: Binary mask resized to the original size.
    """
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # Threshold to binary mask
    mask = cv2.resize(mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    return mask

def main(args):
    # Load the model
    model = CamoXpert(in_channels=3, num_classes=1)
    load_checkpoint(args.checkpoint, model)
    model = model.to(args.device)
    model.eval()

    # Preprocess the input image
    image = preprocess_image(args.image_path, args.img_size)
    original_size = cv2.imread(args.image_path).shape[:2]  # Original image size
    image = image.to(args.device)

    # Perform inference
    with torch.no_grad():
        output, _ = model(image)
        mask = output.sigmoid()  # Apply sigmoid activation

    # Postprocess the mask
    binary_mask = postprocess_mask(mask, original_size)

    # Save the output mask
    output_path = Path(args.output_dir) / f"{Path(args.image_path).stem}_mask.png"
    cv2.imwrite(str(output_path), binary_mask * 255)
    print(f"✅ Mask saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CamoXpert Inference Script")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save the output mask")
    parser.add_argument("--img-size", type=int, default=352, help="Image size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)