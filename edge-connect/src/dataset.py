import os
import glob
import imageio
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except Exception as e:
            print(e)
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        if self.mask == 6:
            size = 256

        img = imageio.imread(self.data[index])

        if len(img.shape) < 3:
            img = gray2rgb(img)

        if size != 0:
            img = self.resize(img, size, size)
        
        img_gray = rgb2gray(img)

        path_to_img = self.data[index]
        filename = path_to_img.split('/')[-1]
        file_path = '/'.join(self.mask_data[0].split('/')[0:-1]) + '/' + filename
        mask = self.load_mask(img, index, f=file_path)
        edge = self.load_edge(img_gray, index, mask)

        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma
        mask = None

        if self.edge == 1:
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float64)

            if sigma == 0:
                sigma = random.randint(1, 4)
            return canny(img, sigma=sigma, mask=mask).astype(np.float64)
        else:
            imgh, imgw = img.shape[0:2]
            edge = imageio.imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)
            return edge

    def load_mask(self, img, index, f):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        if f is not None:
            mask = imageio.imread(f)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imageio.imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        if mask_type == 6:
            mask = imageio.imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = np.array(Image.fromarray(img).resize((width, height), Image.BICUBIC))
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
