import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COD10KDataset(Dataset):
    """
    COD10K Dataset for camouflaged object detection.
    """

    def __init__(self, root_dir, split='train', img_size=352, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        # Setup paths
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Train/Image')
            self.mask_dir = os.path.join(root_dir, 'Train/GT_Object')
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, 'Test/Image')
            self.mask_dir = os.path.join(root_dir, 'Test/GT_Object')
            # Take first 20% of test as validation
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[:val_size]
        else:  # test
            self.image_dir = os.path.join(root_dir, 'Test/Image')
            self.mask_dir = os.path.join(root_dir, 'Test/GT_Object')
            # Take last 80% of test as test
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[val_size:]

        if split == 'train':
            self.image_list = sorted(os.listdir(self.image_dir))

        # Define augmentations
        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.RGBShift(p=0.5)
                ], p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 128).astype(np.float32)

        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Add channel dimension to mask
        mask = mask.unsqueeze(0)

        return image, mask


class CAMODataset(COD10KDataset):
    """
    CAMO Dataset for camouflaged object detection.
    """

    def __init__(self, root_dir, split='train', img_size=352, augment=True):
        super().__init__(root_dir, split, img_size, augment)

        # Override paths for CAMO dataset structure
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Images/Train')
            self.mask_dir = os.path.join(root_dir, 'GT/Train')
        else:
            self.image_dir = os.path.join(root_dir, 'Images/Test')
            self.mask_dir = os.path.join(root_dir, 'GT/Test')

        self.image_list = sorted(os.listdir(self.image_dir))


class NC4KDataset(COD10KDataset):
    """
    NC4K Dataset for camouflaged object detection.
    """

    def __init__(self, root_dir, split='test', img_size=352, augment=False):
        super().__init__(root_dir, split, img_size, augment)

        # NC4K only has test split
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_list = sorted(os.listdir(self.image_dir))