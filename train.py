# In train.py

import argparse
import os
import torch
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True # Speeds up data transfer to GPU
    )

    # 2. Initialize Model and Optimizers
    # ---------------------------------
    model = IMM(
        n_landmarks=args.n_landmarks,
        lambda_perceptual=args.lambda_perceptual
    ).to(device)

    # We need separate optimizers for the generator (encoder + generator) and the discriminator
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
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
            
            # The paper uses the source image as the target image for reconstruction
            target_image = source_image 

            # --- Train the Discriminator ---
            model.discriminator.train()
            optimizer_D.zero_grad()
            
            # Generate an image but don't track gradients for the generator here
            with torch.no_grad():
                encoded_source = model.encoder(source_image)
                reconstructed_image = model.generator(encoded_source, target_landmarks)

            d_loss = model.calculate_discriminator_loss(target_image, reconstructed_image)
            d_loss.backward()
            optimizer_D.step()

            # --- Train the Generator ---
            model.generator.train()
            model.encoder.train()
            optimizer_G.zero_grad()
            
            g_loss, generated_for_g_loss = model.calculate_generator_loss(source_image, target_image, target_landmarks)
            g_loss.backward()
            optimizer_G.step()

            # --- Logging ---
            progress_bar.set_description(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"G Loss: {g_loss.item():.4f} | "
                f"D Loss: {d_loss.item():.4f}"
            )

        # --- Save Checkpoint ---
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Unsupervised Landmark Model")
    
    # Paths and Directories
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CelebA dataset directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    
    # Model and Training Hyperparameters
    parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba', 'mafl'], help='Which dataset to use.')
    parser.add_argument('--image_size', type=int, default=128, help='Size to which images are resized.')
    parser.add_argument('--n_landmarks', type=int, default=5, help='Number of landmarks to detect.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to train.')
    
    # Loss Weights
    parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='Weight for the perceptual loss.')

    # Checkpoint settings
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save a checkpoint every N epochs.')
    
    args = parser.parse_args()
    
    train(args)