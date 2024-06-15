import os
import pickle
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.shapenet import ShapeNetRenderings, shapenet_collate
from global_config import ROOT_PATH
from model.pcr_t import PCRModel
from model.psgn import PSGNModel
from utils.average_meter import AverageMeter
from model.layer.chamfer_wrapper import ChamferDist


class Tester(object):
    def __init__(self, options, data_type='test_and_validation', model=None, state_file=None, detail_report=False):
        self.options = options
        self.detail_report = detail_report

        if torch.cuda.is_available():
            self.gpus = torch.device('cuda:0')
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            # print("CUDA visible devices is activated here")
            visible_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.gpus = list(range(len(visible_gpus)))
            print("CUDA is asking for " + str(visible_gpus) + ", PyTorch to doing a mapping, changing it to " + str(
                self.gpus))

        if model is None:
            if state_file is None:
                raise ValueError("state_file must be not None in case model is not provided!")
            print("Loading from checkpoint: " + state_file)
            if self.options.target == "psgn":
                self.model = PSGNModel()
            else:
                self.model = PCRModel()
            checkpoint = torch.load(os.path.join(ROOT_PATH, state_file), map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            self.bs_total = self.options.test.batch_size * len(self.gpus)
        else:
            self.model = model
            self.bs_total = sum(options.batch_list)

        self.dataset = ShapeNetRenderings(data_type)
        self.num_classes = self.options.dataset.num_classes
        self.weighted_mean = self.options.test.weighted_mean

        self.chamfer = ChamferDist()

        self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes + 1)]
        self.f1_tau = [AverageMeter() for _ in range(self.num_classes + 1)]
        self.f1_2tau = [AverageMeter() for _ in range(self.num_classes + 1)]

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            cd_result = np.mean(d1) + np.mean(d2)
            self.chamfer_distance[label].update(cd_result)
            self.chamfer_distance[-1].update(cd_result)
            f11_result = self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4)
            self.f1_tau[label].update(f11_result)
            self.f1_tau[-1].update(f11_result)
            f12_result = self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4)
            self.f1_2tau[label].update(f12_result)
            self.f1_2tau[-1].update(f12_result)

    def test_step(self, batch):
        with torch.no_grad():
            out = self.model(batch["images"], batch["cameras"])

            batch_size = int(batch["images"].shape[0])
            view_num = int(batch["cameras"].shape[1])
            gt_coord = batch["points"].float()

            if self.options.target == "psgn":
                if self.options.test.view_fusion:
                    pred_coord = out["points"]
                    labels = batch["label"]
                else:
                    gt_coord = gt_coord.unsqueeze(dim=1).expand(-1, view_num, -1, -1).reshape(batch_size * view_num, -1,
                                                                                              3)
                    pred_coord = out["points_all_views"]
                    labels = batch["label"].unsqueeze(dim=1).expand(-1, view_num).reshape(batch_size * view_num, -1)
                self.evaluate_chamfer_and_f1(pred_coord, gt_coord, labels)
            else:
                pred_coord = out["points"][-1]
                self.evaluate_chamfer_and_f1(pred_coord, gt_coord, batch["label"])

        return out

    def test(self):
        self.model.eval()
        print("Running evaluations...")

        data_loader = DataLoader(self.dataset,
                                 batch_size=self.bs_total,
                                 num_workers=self.options.num_workers,
                                 pin_memory=True,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=shapenet_collate)

        loop = tqdm(enumerate(data_loader), total=len(data_loader), dynamic_ncols=True)
        for step, batch in loop:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = self.test_step(batch)

            self.info_set(loop)

        self.final_report()

    def info_set(self, loop):
        loop.set_description('Test')
        info_dict = {}
        for k, v in self.get_result_summary().items():
            info_dict.update({k: v.str(k == 'cd')})
        loop.set_postfix(ordered_dict=info_dict)

    def average_of_average_meters(self, average_meters):
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        return weighted_avg

    def get_result_summary(self):
        return {
            "cd": self.chamfer_distance[-1],
            "f1_tau": self.f1_tau[-1],
            "f1_2tau": self.f1_2tau[-1],
        }

    def final_report(self):
        if self.detail_report:
            for i in range(self.num_classes):
                print(i, "cd:", self.chamfer_distance[i].str(True), "f1_tau:", self.f1_tau[i], "f1_2tau:",
                      self.f1_2tau[i])
        print("Test result: " + ", ".join(
            [key + " " + val.str(key == "cd")
             for key, val in self.get_result_summary().items()]))
