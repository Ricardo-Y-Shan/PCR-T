import os
import pickle
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.shapenet import ShapeNetRenderings, shapenet_collate
from global_config import ROOT_PATH
from model.pcr_t import PCRModel
from model.psgn import PSGNModel
from utils.average_meter import AverageMeter
from model.loss.pcr_loss import PCRLoss
from model.loss.psgn_loss import PSGNLoss
from functions.tester import Tester
from utils.tools import recursive_detach
from utils.data_parallel import AssignDataParallel


class Trainer(object):
    def __init__(self, options, state_file=None):
        self.options = options
        self.time_start = time.time()

        if torch.cuda.is_available():
            self.gpus = torch.device('cuda:0')
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            # print("CUDA visible devices is activated here")
            visible_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.gpus = list(range(len(visible_gpus)))
            print("CUDA is asking for " + str(visible_gpus) + ", PyTorch to doing a mapping, changing it to " + str(
                self.gpus))

        if self.options.target == "psgn":
            self.dataset = ShapeNetRenderings("train", view_list=list(range(24)))
            self.model = PSGNModel()
            self.loss_func = PSGNLoss(self.options)
        else:
            self.dataset = ShapeNetRenderings("train")
            self.model = PCRModel()
            self.loss_func = PCRLoss(self.options)

        self.bs_total = self.options.train.batch_size * len(self.gpus)

        if self.options.balance_gpu:
            self.model = AssignDataParallel(self.options.batch_list, self.model, device_ids=self.gpus).cuda()
            self.bs_total = sum(self.options.batch_list)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.options.optim.lr,
            betas=(self.options.optim.adam_beta1, 0.999),
            weight_decay=self.options.optim.wd)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        self.losses = AverageMeter()

        self.last_epoch = 0

        self.save_dir = ROOT_PATH + 'checkpoints/' + self.options.name

        if state_file is not None:
            self.load_checkpoint(state_file)

    def load_checkpoint(self, ckpt_file):
        print("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(ROOT_PATH, ckpt_file))
        self.model.module.load_state_dict(checkpoint['model'], False)
        if 'epoch' in checkpoint:
            self.last_epoch = checkpoint['epoch']
        for _ in range(self.last_epoch):
            self.lr_scheduler.step()
        print("Start from epoch %03d" % (self.last_epoch + 1))

    def train_step(self, batch, epoch, step):
        out = self.model(batch["images"], batch['cameras'])

        loss, loss_summary = self.loss_func(out, batch, epoch)
        self.losses.update(loss.detach().cpu().item())

        loss = loss / self.options.optim.backward_steps
        loss.backward()
        if (step + 1) % self.options.optim.backward_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return recursive_detach(loss_summary)

    def train(self):
        for epoch in range(self.last_epoch + 1, self.options.train.num_epochs + 1):
            if self.options.target == "pcr":
                self.model.module.psgn.requires_grad_(False)
            data_loader = DataLoader(self.dataset,
                                     batch_size=self.bs_total,
                                     num_workers=self.options.num_workers,
                                     pin_memory=True,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=shapenet_collate)
            self.losses.reset()
            self.model.train()
            self.optimizer.zero_grad()

            loop = tqdm(enumerate(data_loader), total=len(data_loader), dynamic_ncols=True)
            for step, batch in loop:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                loss_summary = self.train_step(batch, epoch, step)

                loop.set_description(
                    f'Epoch [{epoch}/{self.options.train.num_epochs}] ' + 'Loss=%.4e' % self.losses.avg)
                info_dict = {}
                for k, v in loss_summary.items():
                    info_dict.update({k: '%.3e' % v})
                loop.set_postfix(ordered_dict=info_dict)

            self.lr_scheduler.step()

            self.dump_checkpoint(epoch)

            print("Epoch %03d, Time elapsed %s, Loss %.6f (%.6f)" % (
                epoch, self.time_elapsed, self.losses.val, self.losses.avg))

            if epoch % self.options.train.test_epochs == 0:
                torch.cuda.empty_cache()
                self.test(epoch)

            torch.cuda.empty_cache()

    def dump_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model.module.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        checkpoint_file = os.path.join(self.save_dir, "%03d.ckpt" % epoch)
        print(("Dumping to checkpoint file: %s" % checkpoint_file))
        torch.save(checkpoint, checkpoint_file)

    def test(self, epoch):
        ckpt_file = 'checkpoints/' + self.options.name + '/%03d.ckpt' % epoch
        tester = Tester(self.options, data_type='validation', state_file=ckpt_file)
        tester.test()

    @property
    def time_elapsed(self):
        return timedelta(seconds=time.time() - self.time_start)
