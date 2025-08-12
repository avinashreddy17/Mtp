import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import custom modules
from src.datasets import CelebADataset, RescaleAndCrop, ToTensor
from src.imm_model import IMM

def plot_landmarks(ax, image, landmarks, true_landmarks=None):
    """
    Plots the predicted and optional ground truth landmarks on an image.
    """
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=40, marker='.', c='r', label='Predicted')
    if true_landmarks is not None:
        ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], s=40, marker='.', c='g', label='Ground Truth')
    ax.axis('off')

def visualize_predictions(dataset, model, device, output_dir, num_images=16):
    """
    Saves a grid of images with predicted landmarks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    with torch.no_grad():
        for i in range(num_images):
            sample = dataset[i]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_landmarks = sample['keypoints'].numpy()

            # Handle single vs multi-GPU
            if isinstance(model, torch.nn.DataParallel):
                reconstructed_image, predicted_landmarks = model(image_tensor)
            else:
                reconstructed_image, predicted_landmarks = model(image_tensor)

            # Reshape and scale landmarks to image coordinates
            image_size = dataset.image_size[0]
            predicted_landmarks = (predicted_landmarks.squeeze(0).cpu().numpy() * 0.5 + 0.5) * image_size

            # Convert tensor back to a displayable image
            display_image = sample['image'].permute(1, 2, 0).numpy()
            display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())

            plot_landmarks(axes[i], display_image, predicted_landmarks, true_landmarks)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'landmark_predictions.png')
    plt.savefig(save_path)
    print("\nSaved visualization to {}".format(save_path))


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print("Using device: {}".format(device))

    # --- Dataset ---
    image_size_tuple = (args.image_size, args.image_size)
    composed_transforms = transforms.Compose([RescaleAndCrop(image_size_tuple), ToTensor()])
    dataset = CelebADataset(
        data_dir=args.dataset_path,
        subset='test', # Use the test split
        transform=composed_transforms,
        image_size=image_size_tuple
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size * max(1, num_gpus), shuffle=False, num_workers=4)

    # --- Model ---
    model = IMM(n_landmarks=args.n_landmarks)
    print("Loading model checkpoint from: {}".format(args.checkpoint_path))
    state_dict = torch.load(args.checkpoint_path, map_location=device)

    # Handle modules saved from DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if num_gpus > 1:
        print("Using {} GPUs for evaluation!".format(num_gpus))
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    # --- Evaluation Loop ---
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            images = batch['image'].to(device)
            # The full forward pass gets us the reconstructions and landmarks
            if isinstance(model, torch.nn.DataParallel):
                _, _ = model(images)
            else:
                _, _ = model(images)

    print("\nEvaluation Finished!")

    # --- Visualization ---
    if args.visualize:
        visualize_predictions(dataset, model, device, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Unsupervised Landmark Model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CelebA dataset directory.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save visualization images.')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_landmarks', type=int, default=10, help='Number of landmarks (must match trained model).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--visualize', action='store_true', help='Set this flag to save prediction images.')

    args = parser.parse_args()
    evaluate(args)