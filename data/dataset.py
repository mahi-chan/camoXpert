import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=352, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Train/Image')
            self.mask_dir = os.path.join(root_dir, 'Train/GT_Object')
            if not os.path.exists(self.mask_dir):
                self.mask_dir = os.path.join(root_dir, 'Train/GT')
            self.image_list = sorted(os.listdir(self.image_dir))
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, 'Test/Image')
            self.mask_dir = os.path.join(root_dir, 'Test/GT_Object')
            if not os.path.exists(self.mask_dir):
                self.mask_dir = os.path.join(root_dir, 'Test/GT')
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[:val_size]
        else:
            self.image_dir = os.path.join(root_dir, 'Test/Image')
            self.mask_dir = os.path.join(root_dir, 'Test/GT_Object')
            if not os.path.exists(self.mask_dir):
                self.mask_dir = os.path.join(root_dir, 'Test/GT')
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[val_size:]

        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
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
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


class CAMODataset(COD10KDataset):
    def __init__(self, root_dir, split='train', img_size=352, augment=True):
        super().__init__(root_dir, split, img_size, augment)
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Images/Train')
            self.mask_dir = os.path.join(root_dir, 'GT/Train')
        else:
            self.image_dir = os.path.join(root_dir, 'Images/Test')
            self.mask_dir = os.path.join(root_dir, 'GT/Test')
        self.image_list = sorted(os.listdir(self.image_dir))


class NC4KDataset(COD10KDataset):
    def __init__(self, root_dir, split='test', img_size=352, augment=False):
        super().__init__(root_dir, split, img_size, augment)
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_list = sorted(os.listdir(self.image_dir))