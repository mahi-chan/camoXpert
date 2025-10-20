# benchmarks.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.utils import count_parameters
from models.fusion import BiLevelFusion
from models.segmentation_head import SegmentationHead
from metrics import CODMetrics  # Assuming CODMetrics is implemented in a metrics.py file


def evaluate_model(model, dataset, batch_size, device):
    """
    Evaluate the model on a given dataset and compute benchmark metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (torch.utils.data.Dataset): The dataset for evaluation.
        batch_size (int): Batch size for evaluation.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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


if __name__ == "__main__":
    import argparse
    from models.camo_xpert import CamoXpert  # Assuming the main model is implemented in camo_xpert.py
    from dataset import COD10KDataset  # Assuming the dataset class is implemented in dataset.py

    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark CamoXpert on COD datasets")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=352, help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    args = parser.parse_args()

    # Load dataset
    dataset = COD10KDataset(root_dir=args.dataset_path, split="test", img_size=args.img_size, augment=False)

    # Load model
    model = CamoXpert(in_channels=3, num_classes=1)
    model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
    model = model.to(args.device)

    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Evaluate model
    metrics = evaluate_model(model, dataset, args.batch_size, args.device)
    print("\nBenchmark Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")