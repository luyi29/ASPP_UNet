import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool):
        super(DriveDataset, self).__init__()
        img_path = os.path.join(root, 'MR_5')
        label_path = os.path.join(root, 'Mask_5')
        if train:
            self.img_names = [os.path.join(img_path,i) for i in os.listdir(img_path)[10:]]
            self.label_names = [os.path.join(label_path,i) for i in os.listdir(label_path)[10:]]
        else:
            self.img_names = [os.path.join(img_path,i) for i in os.listdir(img_path)[:10]]
            self.label_names = [os.path.join(label_path,i) for i in os.listdir(label_path)[:10]]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_names[idx],cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_names[idx],cv2.IMREAD_GRAYSCALE)
        # img = img.transpose(0,2).contiguous()
        # label = label.transpose(0,2).contiguous()
        return img, label

    def __len__(self):
        return len(self.img_names)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        
        images = torch.tensor(np.array(images))
        targets = torch.tensor(np.array(targets))
        # batched_imgs = cat_list(images, fill_value=0)
        # batched_targets = cat_list(targets, fill_value=255)
        return images, targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

