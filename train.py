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
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {num_gpus}")

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

    # Adjust batch size for the number of GPUs
    effective_batch_size = args.batch_size * max(1, num_gpus)

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
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

    # Wrap the model for multi-GPU training if more than one GPU is available
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # *** FIX: Access the underlying model correctly for both single/multi GPU ***
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer_G = optim.Adam(
        list(raw_model.encoder.parameters()) + list(raw_model.generator.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        raw_model.discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    
    # *** FIX: Define the loss criterion ***
    criterion = nn.BCEWithLogitsLoss()

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
            
            # *** FIX: Access discriminator via raw_model for multi-GPU compatibility ***
            # Train with REAL images
            real_output = raw_model.discriminator(target_image)
            d_loss_real = criterion(real_output, torch.ones_like(real_output))
            d_loss_real.backward()

            # Train with FAKE images
            # *** FIX: Access generator via raw_model ***
            encoded_features = raw_model.encoder(source_image)
            generated_image = raw_model.generator(encoded_features, target_landmarks)

            # *** FIX: Detach the generated images to stop gradients flowing to generator ***
            fake_output = raw_model.discriminator(generated_image.detach()) 
            d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
            d_loss_fake.backward()

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()

            # --- Train the Generator ---
            optimizer_G.zero_grad()
            
            # We pass the generated image to avoid re-computing it
            g_loss, _ = raw_model.calculate_generator_loss(
                source_image, 
                target_image, 
                target_landmarks, 
                reconstructed_image=generated_image # Pass the generated image
            )
            g_loss.backward()
            optimizer_G.step()

            # --- Logging ---
            progress_bar.set_description(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"G Loss: {g_loss.item():.4f} | "
                f"D Loss: {d_loss.item():.4f}"
            )
            
            if args.debug and i >= 5:
                print("\nDebug mode: Ran for 5 iterations. Exiting.")
                break
        
        if args.debug:
            break

        # --- Save Checkpoint ---
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            # *** FIX: Save the state_dict of the underlying model ***
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Unsupervised Landmark Model")
    
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CelebA dataset directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    
    parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba', 'mafl'])
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_landmarks', type=int, default=5, help='Number of landmarks to detect.')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size *per GPU*.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading.')
    
    parser.add_argument('--lambda_perceptual', type=float, default=0.1)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode for a few iterations.')
    
    args = parser.parse_args()
    
    train(args)