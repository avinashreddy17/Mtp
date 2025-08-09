import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CelebADataset(Dataset):
    LANDMARK_LABELS = {'left_eye': 0, 'right_eye': 1}
    N_LANDMARKS = 5

    def __init__(self, data_dir, subset, dataset='celeba', image_size=(128, 128), transform=None):
        self.image_dir, self.images, self.keypoints = self.load_dataset(data_dir, dataset, subset)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)
        keypoints = self.keypoints[idx].astype('float').reshape(-1, 2)

        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_dataset(self, data_root, dataset, subset):
        image_dir = os.path.join(data_root, 'Img', 'img_align_celeba_hq')

        with open(os.path.join(data_root, 'Anno', 'list_landmarks_align_celeba.txt'), 'r') as f:
            lines = f.read().splitlines()
        lines = lines[2:]
        image_files = []
        keypoints = []
        for line in lines:
            image_files.append(line.split()[0])
            keypoints.append([int(x) for x in line.split()[1:]])
        keypoints = np.array(keypoints, dtype=np.float32)

        with open(os.path.join(data_root, 'MAFL', 'training.txt'), 'r') as f:
            mafl_train = set(f.read().splitlines())
        mafl_train_overlap = [i for i, image_file in enumerate(image_files) if image_file in mafl_train]

        images_set = np.zeros(len(image_files), dtype=np.int32)

        if dataset == 'celeba':
            with open(os.path.join(data_root, 'Eval', 'list_eval_partition.txt'), 'r') as f:
                celeba_set = [int(line.split()[1]) for line in f.readlines()]
            images_set[:] = celeba_set
            images_set += 1
        elif dataset == 'mafl':
            images_set[mafl_train_overlap] = 1
        else:
            raise ValueError(f'Dataset = {dataset} not recognized.')

        with open(os.path.join(data_root, 'MAFL', 'testing.txt'), 'r') as f:
            mafl_test = set(f.read().splitlines())
        mafl_test_overlap = [i for i, image_file in enumerate(image_files) if image_file in mafl_test]
        images_set[mafl_test_overlap] = 4

        n_validation = int(round(0.1 * len(mafl_train_overlap)))
        mafl_validation = mafl_train_overlap[-n_validation:]
        images_set[mafl_validation] = 5

        if dataset == 'celeba':
            if subset == 'train':
                label = 1
            elif subset == 'val':
                label = 2
            else:
                raise ValueError(f'subset = {subset} for celeba dataset not recognized.')
        elif dataset == 'mafl':
            if subset == 'train':
                label = 1
            elif subset == 'test':
                label = 4
            elif subset == 'val':
                label = 5
            else:
                raise ValueError(f'subset = {subset} for mafl dataset not recognized.')

        image_files = np.array(image_files)
        images = image_files[images_set == label]
        keypoints = keypoints[images_set == label]

        keypoints = np.reshape(keypoints, [-1, 5, 2])

        return image_dir, images, keypoints

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(keypoints).float()}

class RescaleAndCrop(object):
    def __init__(self, output_size, crop_percent=0.8):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.crop_percent = crop_percent


    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        w, h = image.size
        
        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        resize_sz = int(round(new_h/self.crop_percent))
        margin = int(round((resize_sz-new_h)/2.0))
        
        image = transforms.Resize((resize_sz,resize_sz))(image)
        keypoints = keypoints * [resize_sz / w, resize_sz / h] - margin
        
        image = image.crop((margin, margin, margin+new_h, margin+new_w))

        return {'image': image, 'keypoints': keypoints}