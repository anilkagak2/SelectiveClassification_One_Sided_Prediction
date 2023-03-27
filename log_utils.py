
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='Default', fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, logger, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger is not None:
            self.logger.log('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Logger(object):
    def __init__(self, log_dir, seed, create_model_dir=True):
        """Create a summary writer logging to log_dir."""
        self.seed = int(seed)
        self.log_dir = Path(log_dir)
        self.model_dir = Path(log_dir) / "checkpoint"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger_path = self.log_dir / "seed-{:}-T-{:}.log".format(
            self.seed, time.strftime("%d-%h-at-%H-%M-%S", time.gmtime(time.time()))
        )
        self.logger_file = open(self.logger_path, "w")

    def __repr__(self):
        return "{name}(dir={log_dir})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write("{:}\n".format(string))
            self.logger_file.flush()


