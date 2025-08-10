# In train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import our custom modules
from src.datasets import CelebADataset, RescaleAndCrop, ToTensor
from src.imm_model import IMM

def train(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # 1. Setup Dataset and DataLoader
    # ---------------------------------
    composed_transforms = transforms.Compose([
        RescaleAndCrop(args.image_size),
        ToTensor()
    ])

    dataset = CelebADataset(
        data_dir=args.dataset_path,
        subset='train',
        dataset=args.dataset,
        transform=composed_transforms
    )

    dataloader = DataLoader(
        dataset,
        # Increase batch size for multi-GPU training
        batch_size=args.batch_size * torch.cuda.device_count(),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Initialize Model and Optimizers
    # ---------------------------------
    model = IMM(
        n_landmarks=args.n_landmarks,
        lambda_perceptual=args.lambda_perceptual
    )

    # *** IMPORTANT: Wrap the model for multi-GPU training ***
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)


    # Optimizers still target the original model parameters
    optimizer_G = optim.Adam(
        list(model.module.encoder.parameters()) + list(model.module.generator.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model.module.discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    # Create directory for checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 3. The Training Loop
    # -------------------
    print("Starting training...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in progress_bar:
            source_image = batch['image'].to(device)
            target_landmarks = batch['keypoints'].to(device)
            target_image = source_image

            # --- Train the Discriminator ---
            optimizer_D.zero_grad()
            with torch.no_grad():
                reconstructed_image = model(source_image, target_landmarks)
            
            # Access loss calculation methods via model.module
            d_loss = model.module.calculate_discriminator_loss(target_image, reconstructed_image)
            d_loss.backward()
            optimizer_D.step()

            # --- Train the Generator ---
            optimizer_G.zero_grad()
            g_loss, _ = model.module.calculate_generator_loss(source_image, target_image, target_landmarks)
            g_loss.backward()
            optimizer_G.step()

            # --- Logging ---
            progress_bar.set_description(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"G Loss: {g_loss.item():.4f} | "
                f"D Loss: {d_loss.item():.4f}"
            )

            if args.debug and i >= 5: # Run for 5 batches in debug mode
                print("\nDebug mode: Ran for 5 iterations. Exiting.")
                break

        # --- Save Checkpoint ---
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            # Save the state_dict of the underlying model
            model.module.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if args.debug:
            break # Exit after the first epoch in debug mode

    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Unsupervised Landmark Model")
    
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CelebA dataset directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode for a few iterations.')

    parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba', 'mafl'])
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_-argument('--n_landmarks', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size *per GPU*.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading.')
    
    parser.add_argument('--lambda_perceptual', type=float, default=0.1)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    
    args = parser.parse_args()
    train(args)
