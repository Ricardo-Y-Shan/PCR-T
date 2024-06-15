import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.shapenet import ShapeNetRenderings, shapenet_collate
from data.pix3d import Pix3D
from global_config import ROOT_PATH, PREDICT_DIR
from model.pcr_t import PCRModel
from model.psgn import PSGNModel


class Predictor(object):
    def __init__(self, options, data_type, ckpt_file, name, detail_record=False):
        self.options = options
        if self.options.target == "psgn":
            self.model = PSGNModel()
        else:
            self.model = PCRModel()
        checkpoint = torch.load(os.path.join(ROOT_PATH, ckpt_file))
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.cuda()

        if options.dataset.name == 'shapenet':
            print("ShapeNet: " + data_type)
            self.dataset = ShapeNetRenderings(data_type)
        else:
            assert self.options.target == "pcr", "The model must be PCRModel!"
            print("Pix3D")
            self.dataset = Pix3D()

        self.name = name
        self.detail_record = detail_record

    def predict_all(self):
        self.model.eval()
        print("Running predictions...")

        data_loader = DataLoader(self.dataset, batch_size=1, pin_memory=True, collate_fn=shapenet_collate)

        save_dir = PREDICT_DIR + "/" + self.name + "/"
        os.makedirs(save_dir, exist_ok=True)

        for element in tqdm(data_loader):
            with torch.no_grad():
                out = self.model(element["images"].cuda(), element["cameras"].cuda())
            if self.options.target == "psgn":
                pred_points = out["points"].squeeze().cpu().numpy()
            else:
                pred_points = out["points"][-1].squeeze().cpu().numpy()
            gt_points = element["points_resample"].squeeze().numpy()
            prefix = save_dir + element["category_id"][0] + "_" + element["item_id"][0]
            pred_file = prefix + "_final.obj"
            gt_file = prefix + "_gt.obj"
            np.savetxt(pred_file, pred_points)
            np.savetxt(gt_file, gt_points)
            # If needed, save intermediate results.
            # if self.detail_record:
            #     pred_points_before_mdn = out["points_before_gcn"].squeeze().cpu().numpy()
            #     mid_file = prefix + "_mid.obj"
            #     np.savetxt(mid_file, pred_points_before_mdn)

        print("Finish predictions.")

    def predict_pix3d(self):
        assert self.options.target == "pcr", "The model must be PCRModel!"

        self.model.eval()
        print("Running predictions...")

        data_loader = DataLoader(self.dataset, batch_size=1, pin_memory=True)

        save_dir = PREDICT_DIR + "/" + self.name + "/"
        os.makedirs(save_dir, exist_ok=True)

        for element in tqdm(data_loader):
            if element["image"].shape[1] != 3:
                continue
            prefix = save_dir + element["category"][0] + "_" + element["id"][0]
            pred_file = prefix + ".obj"
            if os.path.exists(pred_file):
                continue

            with torch.no_grad():
                out = self.model(element["image"].cuda(), None)
            pred_points = out["points"][-1].squeeze().cpu().numpy()
            np.savetxt(pred_file, pred_points)

        print("Finish predictions.")
