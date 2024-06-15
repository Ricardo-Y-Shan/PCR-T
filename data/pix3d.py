import json
import os
import cv2

import torch
import numpy as np

from torch.utils.data.dataset import Dataset

from global_config import Pix3D_ROOT


class Pix3D(Dataset):
    def __init__(self):
        with open(os.path.join(Pix3D_ROOT, 'pix3d.json'), 'r') as f:
            models_dict = json.load(f)
        self.item_list = []

        cats = ['chair', 'sofa', 'table']

        # Check for truncation and occlusion before adding a model to the evaluation list
        for d in models_dict:
            if d['category'] in cats:
                if not d['truncated'] and not d['occluded'] and not d['slightly_occluded']:
                    self.item_list.append(d)

    def __getitem__(self, index):
        item = self.item_list[index]
        id = item['img'].split('/')[-1].split('.')[0]
        img_path = os.path.join(Pix3D_ROOT, item['img'])
        mask_path = os.path.join(Pix3D_ROOT, item['mask'])

        ip_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path) != 0
        if ip_image.shape[-1] != 3:
            return {
                'category': item['category'],
                'image': torch.tensor(ip_image, dtype=torch.float32),
                'id': id,
            }
        ip_image = ip_image * mask_image

        # ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        current_shape = ip_image.shape[:2]
        ratio = 224.0 / max(current_shape)
        new_shape = tuple(round(x * ratio) for x in current_shape)
        ip_image = cv2.resize(ip_image, (new_shape[1], new_shape[0]))
        op_image = np.zeros((224, 224, 3))
        img_start = (224 - min(new_shape)) // 2
        img_end = img_start + min(new_shape)
        if new_shape[0] < 224:
            op_image[img_start:img_end, :, :] = ip_image
        elif new_shape[1] < 224:
            op_image[:, img_start:img_end, :] = ip_image
        op_image = op_image.astype('float16') / 255.0
        op_image[np.where(op_image == [0, 0, 0])] = 1.0

        return {
            'category': item['category'],
            'image': torch.tensor(op_image, dtype=torch.float32).permute(2, 0, 1),
            'id': id,
        }

    def __len__(self):
        return len(self.item_list)
