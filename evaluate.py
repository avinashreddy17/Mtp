# In evaluate.py

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from src.datasets import CelebADataset, RescaleAndCrop, ToTensor
from src.imm_model import IMM

def calculate_nme(predicted_kpts, true_kpts):
    """
    Calculates the Normalized Mean Error (NME).
    The error is normalized by the inter-ocular distance.
    """
    # In CelebA, the left eye is index 0 and the right eye is index 1
    interocular_distance = torch.sqrt(torch.sum((true_kpts[:, 0, :] - true_kpts[:, 1, :])**2, dim=1))
    
    # Calculate the L2 error for each landmark
    error_per_landmark = torch.sqrt(torch.sum((predicted_kpts - true_kpts)**2, dim=2))
    
    # Average the error across all landmarks
    mean_error = torch.mean(error_per_landmark, dim=1)
    
    # Normalize by the inter-ocular distance
    normalized_mean_error = mean_error / interocular_distance
    
    return normalized_mean_error.mean().item()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    composed_transforms = transforms.Compose([RescaleAndCrop(args.image_size), ToTensor()])
    dataset = CelebADataset(
        data_dir=args.dataset_path,
        # Use the MAFL test set as per the paper's evaluation protocol
        subset='test',
        dataset='mafl',
        transform=composed_transforms
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = IMM(n_landmarks=args.n_landmarks).to(device)
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    model.load_weights(args.checkpoint_path)
    model.eval()

    # --- Evaluation Loop ---
    total_nme = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            images = batch['image'].to(device)
            true_landmarks = batch['keypoints'].to(device)

            # Discover landmarks using our new method
            # Note: The discovery method itself is an optimization loop.
            # We're running it without tracking gradients for evaluation purposes.
            discovered_landmarks = model.discover_landmarks(images)
            
            nme = calculate_nme(discovered_landmarks, true_landmarks)
            total_nme += nme

    # --- Results ---
    avg_nme = total_nme / len(dataloader)
    print(f"\nEvaluation Finished!")
    print(f"Normalized Mean Error (NME) on the test set: {avg_nme:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Unsupervised Landmark Model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CelebA/MAFL dataset directory.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_landmarks', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation (can be smaller than training).')
    args = parser.parse_args()
    evaluate(args)