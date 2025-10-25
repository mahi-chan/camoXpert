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

        # Try multiple possible directory structures
        possible_structures = [
            # Structure 1: Train/Image, Train/GT_Object
            {
                'train_img': 'Train/Image',
                'train_mask': 'Train/GT_Object',
                'test_img': 'Test/Image',
                'test_mask': 'Test/GT_Object'
            },
            # Structure 2: Train/Imgs, Train/GT
            {
                'train_img': 'Train/Imgs',
                'train_mask': 'Train/GT',
                'test_img': 'Test/Imgs',
                'test_mask': 'Test/GT'
            },
            # Structure 3: Direct Train/Test folders
            {
                'train_img': 'Train',
                'train_mask': 'TrainGT',
                'test_img': 'Test',
                'test_mask': 'TestGT'
            },
            # Structure 4: Flat structure with subfolders
            {
                'train_img': 'TrainDataset/Imgs',
                'train_mask': 'TrainDataset/GT',
                'test_img': 'TestDataset/Imgs',
                'test_mask': 'TestDataset/GT'
            }
        ]

        # Find the correct structure
        structure = None
        for struct in possible_structures:
            if split == 'train':
                img_path = os.path.join(root_dir, struct['train_img'])
                mask_path = os.path.join(root_dir, struct['train_mask'])
            else:
                img_path = os.path.join(root_dir, struct['test_img'])
                mask_path = os.path.join(root_dir, struct['test_mask'])

            if os.path.exists(img_path):
                structure = struct
                break

        if structure is None:
            # List available directories to help debug
            available = os.listdir(root_dir) if os.path.exists(root_dir) else []
            raise FileNotFoundError(
                f"Could not find COD10K dataset structure in {root_dir}\n"
                f"Available directories: {available}\n"
                f"Please check your dataset structure."
            )

        # Set paths based on found structure
        if split == 'train':
            self.image_dir = os.path.join(root_dir, structure['train_img'])
            self.mask_dir = os.path.join(root_dir, structure['train_mask'])
            self.image_list = sorted(os.listdir(self.image_dir))
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, structure['test_img'])
            self.mask_dir = os.path.join(root_dir, structure['test_mask'])
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[:val_size]
        else:  # test
            self.image_dir = os.path.join(root_dir, structure['test_img'])
            self.mask_dir = os.path.join(root_dir, structure['test_mask'])
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[val_size:]

        print(f"{split.upper()} dataset initialized: {len(self.image_list)} images")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Mask dir:  {self.mask_dir}")

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
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Try multiple mask extensions
        mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
        base_name = os.path.splitext(img_name)[0]

        mask = None
        for ext in mask_extensions:
            mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            raise ValueError(f"Failed to load mask for: {img_name}")

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