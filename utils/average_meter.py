from collections.abc import Iterable

import torch

import numpy as np


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        if isinstance(val, Iterable):
            val = np.array(val)
            self.update(np.mean(np.array(val)), n=val.size)
        else:
            self.val = self.multiplier * val
            self.sum += self.multiplier * val * n
            self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self, e=False):
        if e:
            return "%.4e (%.4e)" % (self.val, self.avg)
        else:
            return "%.4f (%.4f)" % (self.val, self.avg)

    def str(self, e=False):
        return self.__str__(e)
