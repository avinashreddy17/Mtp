import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def create_heatmap_tensor(coords, image_size, gauss_std=0.1):
    """
    Creates a heatmap tensor from landmark coordinates.
    This is a direct PyTorch port of the original TensorFlow implementation.
    """
    # Create a grid of coordinates
    x = torch.arange(0, image_size[1], dtype=torch.float32)
    y = torch.arange(0, image_size[0], dtype=torch.float32)
    xx, yy = torch.meshgrid(y, x, indexing='ij')
    
    # Add batch and channel dimensions for broadcasting
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)

    # Prepare landmark coordinates
    coords = coords.float().unsqueeze(2).unsqueeze(3)
    
    # Calculate squared distance from each landmark
    dist_sq = (xx - coords[:, :, 0:1, :])**2 + (yy - coords[:, :, 1:2, :])
    
    # Generate heatmap using Gaussian function
    # The clamp prevents dist_sq from being too small, which can cause NaNs
    dist_sq = torch.clamp(dist_sq, min=1e-5)
    heatmaps = torch.exp(-dist_sq / (2 * gauss_std**2))
    
    # The original implementation sums the heatmaps for all landmarks
    heatmap_sum = torch.sum(heatmaps, dim=1)
    
    return heatmap_sum

class CelebADataset(Dataset):
    def __init__(self, data_dir, subset, dataset='celeba', transform=None, image_size=(128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        if dataset == 'celeba':
            self.landmarks_file = os.path.join(data_dir, 'Anno', 'list_landmarks_align_celeba.txt')
            self.partition_file = os.path.join(data_dir, 'Eval', 'list_eval_partition.txt')
            self.image_dir = os.path.join(data_dir, 'Img', 'img_align_celeba')
        else: # MAFL
            self.landmarks_file = os.path.join(data_dir, 'MAFL', 'list_landmarks_mafl.txt')
            self.partition_file = os.path.join(data_dir, 'MAFL', 'list_eval_partition.txt')
            self.image_dir = os.path.join(data_dir, 'Img', 'img_align_celeba') # MAFL images are also in the main image dir
            
        self.image_filenames = []
        self.landmarks = {}
        
        # Map subset names to partition IDs
        subset_map = {'train': '0', 'val': '1', 'test': '2'}
        target_partition = subset_map[subset]

        # Read partition file to get the list of images for the target subset
        with open(self.partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[1] == target_partition:
                    self.image_filenames.append(parts[0])

        # Read landmarks file
        with open(self.landmarks_file, 'r') as f:
            lines = f.readlines()
            # Skip the header lines
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                # Store landmarks as a list of integers
                self.landmarks[img_name] = [int(p) for p in parts[1:]]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get landmarks, reshape to (N, 2) format
        keypoints = np.array(self.landmarks[img_name]).reshape(-1, 2)
        
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

class RescaleAndCrop(object):
    """Rescale the image in a sample to a given size and crop."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # The original dataset is already aligned and cropped, so we just resize
        rescaler = transforms.Resize(self.output_size)
        image = rescaler(image)
        return {'image': image, 'keypoints': keypoints}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        # Swap color axis for PyTorch: H x W x C -> C x H x W
        image = transforms.ToTensor()(image)
        # Normalize to [-1, 1] range, as expected by the generator/discriminator
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        
        return {'image': image, 'keypoints': torch.from_numpy(keypoints)}