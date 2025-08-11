import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets import CelebADataset, RescaleAndCrop, ToTensor, create_heatmap_tensor
from src.imm_model import IMM

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, Number of GPUs: {num_gpus}")

    # Data
    image_size_tuple = (args.image_size, args.image_size)
    composed_transforms = transforms.Compose([RescaleAndCrop(image_size_tuple), ToTensor()])
    dataset = CelebADataset(data_dir=args.dataset_path, subset='train', transform=composed_transforms, image_size=image_size_tuple)
    dataloader = DataLoader(dataset, batch_size=args.batch_size * max(1, num_gpus), shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = IMM(n_landmarks=args.n_landmarks, lambda_perceptual=args.lambda_perceptual)
    raw_model = model
    if num_gpus > 1:
        model = nn.DataParallel(model)
        raw_model = model.module
    model.to(device)

    # Optimizers
    optimizer_G = optim.Adam(list(raw_model.encoder.parameters()) + list(raw_model.generator.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(raw_model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Starting Training...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            real_images = batch['image'].to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Train with Real Images
            d_real_pred = raw_model.discriminator(real_images)
            d_real_loss = criterion(d_real_pred, torch.ones_like(d_real_pred))
            
            # Generate fake images for discriminator training
            predicted_landmarks = raw_model.encoder(real_images)
            # Scale landmarks from [-1, 1] to [0, H] image coords
            landmarks_scaled = (predicted_landmarks.detach() * 0.5 + 0.5) * args.image_size
            heatmaps = create_heatmap_tensor(landmarks_scaled, image_size_tuple)
            fake_images = raw_model.generator(real_images, heatmaps)
            
            # Train with Fake Images
            d_fake_pred = raw_model.discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_pred, torch.zeros_like(d_fake_pred))
            
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator & Encoder ---
            optimizer_G.zero_grad()
            # We can re-use the fake_images from the discriminator step
            g_loss = raw_model.calculate_generator_loss(real_images, fake_images)
            g_loss.backward()
            optimizer_G.step()

            progress_bar.set_postfix(G_Loss=f"{g_loss.item():.4f}", D_Loss=f"{d_loss.item():.4f}")

        # Save Checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train IMM")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_landmarks', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lambda_perceptual', type=float, default=0.1)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    args = parser.parse_args()
    train(args)