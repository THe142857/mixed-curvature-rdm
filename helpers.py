import os
import torch
import datetime

from torch import nn


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# noinspection PyAttributeOutsideInit
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ExponentialMovingAverage(nn.Module):

    def __init__(self, ema_module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        super(ExponentialMovingAverage, self).__init__()
        self.ema_module = ema_module
        self.decay = decay

    def init(self, target_module):
        ema_params = dict(self.ema_module.named_parameters())
        for name, param in target_module.named_parameters():
            ema_params[name].data.copy_(param.data)
        return self.ema_module

    def step(self, target_module):
        ema_params = dict(self.ema_module.named_parameters())
        with torch.no_grad():
            for name, param in target_module.named_parameters():
                ema_params[name].data.mul_(self.decay).add_(param.data * (1 - self.decay))
        return self.ema_module


def getpaths(save_folder='default'):
    droot = os.environ['DATASETS']
    sroot = os.environ['SAVEPATH']
    spath = sroot + '/' + save_folder
    create(spath)
    return droot, sroot, spath


def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


def logging(s, path='./', filename='log.txt', print_=True, log_=True):
    s = str(datetime.datetime.now()) + '\t' + str(s)
    if print_:
        print(s)
    if log_:
        assert path, 'path is not define. path: {}'.format(path)
    with open(os.path.join(path, filename), 'a+') as f_log:
        f_log.write(s + '\n')
