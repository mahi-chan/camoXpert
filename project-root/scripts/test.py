# test.py

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.camo_xpert import CamoXpert  # Assuming the model is implemented in camo_xpert.py
from dataset import COD10KDataset  # Assuming the dataset class is implemented in dataset.py
from metrics import CODMetrics  # Assuming CODMetrics is implemented in metrics.py
from models.utils import load_checkpoint, count_parameters

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    metrics = CODMetrics()
    all_metrics = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            predictions, _ = model(images)
            batch_metrics = metrics.compute_all(predictions, masks)
            all_metrics.append(batch_metrics)

    # Aggregate metrics
    aggregated_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
    return aggregated_metrics

def main(args):
    # Load dataset
    dataset = COD10KDataset(root_dir=args.dataset_path, split="test", img_size=args.img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = CamoXpert(in_channels=3, num_classes=1)
    load_checkpoint(args.checkpoint, model)
    model = model.to(args.device)

    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Evaluate model
    metrics = evaluate_model(model, dataloader, args.device)
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CamoXpert Test Script")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--img-size", type=int, default=352, help="Image size for testing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run testing on")
    args = parser.parse_args()

    main(args)