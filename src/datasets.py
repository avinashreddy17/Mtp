import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def create_heatmap_tensor(coords_batch, image_size, gauss_std=0.1):
    """
    Creates a heatmap tensor from a batch of landmark coordinates.
    This is a direct and corrected PyTorch port of the original TensorFlow implementation.
    """
    # Create a grid of coordinates for a single image
    x = torch.arange(0, image_size[1], dtype=torch.float32, device=coords_batch.device)
    y = torch.arange(0, image_size[0], dtype=torch.float32, device=coords_batch.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Expand grid to match batch size
    # Shape: [batch_size, 1, H, W]
    xx = xx.unsqueeze(0).unsqueeze(1).repeat(coords_batch.size(0), 1, 1, 1)
    yy = yy.unsqueeze(0).unsqueeze(1).repeat(coords_batch.size(0), 1, 1, 1)

    # Separate and reshape landmark coordinates for broadcasting
    # Shape: [batch_size, n_landmarks, 1, 1]
    coords_x = coords_batch[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    coords_y = coords_batch[:, :, 1].unsqueeze(-1).unsqueeze(-1)

    # Calculate squared distance for all landmarks at once
    # This broadcasts each landmark's (x, y) over the entire grid
    dist_sq = (xx - coords_y)**2 + (yy - coords_x)**2

    # Generate heatmaps using Gaussian function.
    # The clamp prevents dist_sq from being too small, which can cause NaNs.
    dist_sq = torch.clamp(dist_sq, min=1e-5)
    heatmaps = torch.exp(-dist_sq / (2 * (image_size[0] * gauss_std)**2))

    # The original implementation sums the heatmaps for all landmarks
    # to create a single-channel heatmap per image in the batch.
    heatmap_sum = torch.sum(heatmaps, dim=1, keepdim=True)

    return heatmap_sum


class CelebADataset(Dataset):
    def __init__(self, data_dir, subset, dataset='celeba', transform=None, image_size=(128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        if dataset == 'celeba':
            self.landmarks_file = os.path.join(data_dir, 'Anno', 'list_landmarks_align_celeba.txt')
            self.partition_file = os.path.join(data_dir, 'Eval', 'list_eval_partition.txt')
            self.image_dir = os.path.join(data_dir, 'Img', 'img_align_celeba_hq')
        else: # MAFL
            self.landmarks_file = os.path.join(data_dir, 'MAFL', 'list_landmarks_mafl.txt')
            self.partition_file = os.path.join(data_dir, 'MAFL', 'list_eval_partition.txt')
            self.image_dir = os.path.join(data_dir, 'Img', 'img_align_celeba_hq')

        self.image_filenames = []
        self.landmarks = {}

        subset_map = {'train': '0', 'val': '1', 'test': '2'}
        target_partition = subset_map[subset]

        with open(self.partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1 and parts[1] == target_partition:
                    self.image_filenames.append(parts[0])

        with open(self.landmarks_file, 'r') as f:
            lines = f.readlines()[2:] # Skip header
            for line in lines:
                parts = line.strip().split()
                self.landmarks[parts[0]] = [int(p) for p in parts[1:]]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        keypoints = np.array(self.landmarks[img_name]).reshape(-1, 2)
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RescaleAndCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        rescaler = transforms.Resize(self.output_size)
        image = rescaler(image)
        return {'image': image, 'keypoints': keypoints}

class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        return {'image': image, 'keypoints': torch.from_numpy(keypoints)}