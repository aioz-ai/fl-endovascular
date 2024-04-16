import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class MedDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, domain: str , list_img=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        #self.list_img = list_img
        #self.mask_suffix = mask_suffix
        if list_img is None:
            self.ids = listdir(self.images_dir)# [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        else: 
            self.ids = list_img
        if self.ids is None:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        assert domain in ["phantom", "animal", "sim", "real"]
        self.domain = domain 
        if self.domain in ["animal", "phantom"]:
            self.mask_values =list([0,1,2]) 
        else:  #sim , real
            self.mask_values = list([0,1])
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(mask_values, pil_img,  is_mask):
        newW, newH = 256,256
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        

        if is_mask:
            img = np.where(img>128,1,img)
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        image_name = self.ids[idx]
        image_dir = join(self.images_dir, image_name)
        img = Image.open(image_dir)
        if self.domain== "animal" or self.domain == "phantom": 
            mask_name = image_name.replace("png", "npy")
            mask_dir = join(self.mask_dir,mask_name)
            mask = np.load(mask_dir)
            mask = Image.fromarray(mask)
            
        elif self.domain == 'sim': 
            mask_name = image_name.split(".")[0] + "_mask.png"
            mask_dir = join(self.mask_dir,mask_name)
            mask = Image.open(mask_dir)
            img = Image.open(image_dir).convert("RGB")
        else : # real
            mask_name = image_name
            mask_dir = join(self.mask_dir,mask_name)
            mask = Image.open(mask_dir)


        assert img.size == mask.size, \
            f'Image and mask {image_name} should be the same size, but are {img.size} and {mask.size}'
       # print(img.shape, mask.shape)
        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(self.mask_values, mask,  is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': image_name
        }


