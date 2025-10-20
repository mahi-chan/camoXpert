
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.camo_xpert import CamoXpert
from dataset import COD10KDataset
from metrics import CODMetrics
from models.utils import load_checkpoint, count_parameters

def validate_model(model, dataloader, device):
    """
    Validate the model on the validation dataset and compute metrics.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the validation on.

    Returns:
        dict: Dictionary containing validation metrics.
    """
    model.eval()
    metrics = CODMetrics()
    all_metrics = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            predictions, _ = model(images)
            batch_metrics = metrics.compute_all(predictions, masks)
            all_metrics.append(batch_metrics)

    # Aggregate metrics
    aggregated_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
    return aggregated_metrics

def main(args):
    # Load dataset
    dataset = COD10KDataset(root_dir=args.dataset_path, split="val", img_size=args.img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = CamoXpert(in_channels=3, num_classes=1)
    load_checkpoint(args.checkpoint, model)
    model = model.to(args.device)

    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Validate model
    metrics = validate_model(model, dataloader, args.device)
    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CamoXpert Validation Script")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--img-size", type=int, default=352, help="Image size for validation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run validation on")
    args = parser.parse_args()

    main(args)