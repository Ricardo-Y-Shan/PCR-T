import json
import os
import cv2

import torch
import numpy as np
import pickle

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from global_config import ROOT_PATH, SHAPENET_PATH


class ShapeNetRenderings(Dataset):
    def __init__(self, data_type="train", view_list=None):
        # data_type should be either "test", "train"
        self.data_type = data_type
        self.data_root = os.path.join(ROOT_PATH, "data")

        self.view_list = view_list if view_list is not None else [0, 6, 9]

        with open(os.path.join(self.data_root, "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}

        txt_name = data_type + "_list.txt"
        self.file_name = os.path.join(self.data_root, txt_name)
        self.item_names: list[str] = []
        with open(self.file_name, "r") as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break
                self.item_names.append(line)

    def __getitem__(self, index):
        path = self.item_names[index]

        ids = path.split('_')
        category = ids[0]
        item_id = ids[1]
        img_path = os.path.join(SHAPENET_PATH["image_dir"], category, item_id, 'rendering')
        camera_meta_data = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))

        view_num = len(self.view_list)
        imgs = np.zeros((view_num, 224, 224, 3))
        poses = np.zeros((view_num, 5))
        for idx, view in enumerate(self.view_list):
            img = cv2.imread(os.path.join(img_path, str(view).zfill(2) + '.png'), cv2.IMREAD_UNCHANGED)
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float16') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[view]

        with open(os.path.join(SHAPENET_PATH["gt_dir"], "train" if self.data_type == "train" else "test",
                               path), 'rb') as pickle_file:
            pkl = pickle.load(pickle_file, encoding='bytes')
            if self.data_type != 'train':
                pkl = pkl[1]
            points_number, _ = pkl.shape

            points = pkl[:, :3]  # get the first 3 element of every list
            normals = pkl[:, 3:6]  # get the last 3 element of every list

        resample_file = os.path.join(SHAPENET_PATH["gt_dir"], "_".join([category, item_id, "resample.obj"]))
        points_resample = np.loadtxt(resample_file)

        return {
            "category_id": category,
            "label": self.labels_map[category],
            "item_id": item_id,
            "images": torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2),
            "cameras": torch.tensor(poses, dtype=torch.float32),
            "points": torch.tensor(points, dtype=torch.float32),
            "points_resample": torch.tensor(points_resample, dtype=torch.float32),
            "normals": torch.tensor(normals, dtype=torch.float32),
            "length": points_number,
        }

    def __len__(self):
        return len(self.item_names)


def shapenet_collate(batch):
    if len(batch) > 1:
        all_equal = True
        max_length = batch[0]["length"]
        for t in batch:
            if t["length"] != batch[0]["length"]:
                all_equal = False
                max_length = max(max_length, t["length"])
        points_orig, normals_orig = [], []
        if not all_equal:
            for t in batch:
                pts, normal = t["points"], t["normals"]
                length = pts.shape[0]
                choices = np.resize(np.random.permutation(length), max_length)
                t["points"], t["normals"] = pts[choices], normal[choices]
                points_orig.append(pts)
                normals_orig.append(normal)
            ret = default_collate(batch)
            ## ret["images"] = ret["images"].reshape(len(batch) * 3, 3, 224, 224)
            ret["points_orig"] = points_orig
            ret["normals_orig"] = normals_orig
            return ret
    ret = default_collate(batch)
    ## ret["images"] = ret["images"].squeeze(0)
    ret["points_orig"] = ret["points"]
    ret["normals_orig"] = ret["normals"]
    return ret
