import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv
from src.config.configuration import *

class SemanticSegmentationDataLoader(Dataset):

    def __init__(self, root_dir: str = None, split: str= None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        print(self.root_dir,self.split)
        self.img_dir = os.path.join(root_dir, self.split, 'img')
        self.lbl_dir = os.path.join(root_dir, self.split, 'lbl')
        assert os.path.exists(self.img_dir), f'image directory @ {self.img_dir} does not exist'
        assert os.path.exists(self.lbl_dir), f'image directory @ {self.lbl_dir} does not exist'
        self.image_filenames = os.listdir(self.img_dir)
        self.mask_filenames = os.listdir(self.lbl_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load image and mask
        image_path = os.path.join(self.img_dir, self.image_filenames[index])
        mask_path = os.path.join(self.lbl_dir, self.mask_filenames[index])
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
        trans = transforms.Compose([
            transforms.ToTensor()])
        if self.transform is not None:
            # Apply transform if provided
            image = trans(image)

        return image, mask
