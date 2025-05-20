import time
from enum import Enum

import numpy as np
import torch


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, initVal=0, name=""):
        self.initVal = initVal
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.val = self.initVal
        self.avg = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.val = val

    def inc(self, val=1):
        self.update(self.val + val)

    def to_dict(self, prefix):
        return {
            prefix + "-" + self.name: self.avg,
        }


class RunningMovingAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(
        self,
        initVal=None,
        momentum=0.99,
    ):
        self.momentum = momentum
        self.initVal = initVal
        self.val = initVal
        self.avg = 0

    def reset(self):
        self.val = self.initVal
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

    def inc(self, val=1):
        self.update(self.val + val)


class TimeMeter(object):

    class TimeStateEnum(Enum):
        START = 1
        STOP = 2

    def __init__(self):
        self.reset()

    def start(self):
        if self.state == TimeMeter.TimeStateEnum.START:
            return self
        self.preTime = time.time()
        self.currentTime = self.preTime
        self.state = TimeMeter.TimeStateEnum.START
        return self

    def reset(self):
        self.startTime = time.time()
        self.currentTime = 0
        self.preTime = 0
        self.state = TimeMeter.TimeStateEnum.STOP

    def stop(self):
        if self.state == TimeMeter.TimeStateEnum.STOP:
            return self
        self.currentTime = time.time()
        self.state = TimeMeter.TimeStateEnum.STOP
        return self

    def interval(self):
        return self.currentTime - self.preTime

    def intervalToNow(self):
        return time.time() - self.preTime

    def totalTime(self):
        self.currentTime = time.time()
        return self.currentTime - self.startTime

    def intervalToNowAndRestart(self):
        timeInterval = self.intervalToNow()
        self.stop()
        self.start()
        return timeInterval


class HistogramMeter(object):

    def __init__(self, desc):
        self.count_dict = dict()
        self.sum_dict = dict()
        self.desc = desc

    def add_metric_with_group(self, metrics, groups):
        for i in range(groups.shape[0]):
            key = f"{self.desc}-{groups[i].item()}"
            if key in self.sum_dict:
                self.sum_dict[key] += metrics[i].item()
                self.count_dict["cnt-" + key] += 1
            else:
                self.sum_dict[key] = metrics[i].item()
                self.count_dict["cnt-" + key] = 1

    def add_metric(self, metrics):
        # print(metrics)
        for key in metrics.keys():
            if key in self.sum_dict:
                self.sum_dict[key] += metrics[key]
                self.count_dict["cnt-" + key] += metrics[key]
            else:
                self.sum_dict[key] = metrics[key]
                self.count_dict["cnt-" + key] = metrics[key]

    def get_avg_metric_dict(self):
        for key in self.sum_dict.keys():
            self.sum_dict[key] = self.sum_dict[key] / self.count_dict["cnt-" + key]

        self.sum_dict.update(self.count_dict)
        return self.sum_dict

    def get_avg_ratio_metric_dict(self, prefix):
        total_cnt = np.array(list(self.count_dict.values())).sum().item()
        ratio_dict = {prefix + "-" + key: self.sum_dict[key] / total_cnt for key in self.sum_dict.keys()}
        return ratio_dict

    def get_avg_metric_dict_without_cnt(self):
        for key in self.sum_dict.keys():
            self.sum_dict[key] = self.sum_dict[key] / self.count_dict["cnt-" + key]
        return self.sum_dict

    def reset(self):
        self.count_dict.clear()
        self.sum_dict.clear()


def cuda_memory(device):

    memory_info = {
        "m_max_reserved": torch.cuda.max_memory_reserved(device) / 2**30,
        "m_allocated": torch.cuda.memory_allocated(device) / 2**30,
        "m_max_allocated": torch.cuda.max_memory_allocated(device) / 2**30,
        "m_reserved": torch.cuda.memory_reserved(device) / 2**30,
    }
    torch.cuda.reset_peak_memory_stats(device)
    return memory_info


def fileter_object_states(object):
    state = {}
    for key, val in object.__dict__.items():
        if isinstance(val, (int, float, str, bool, bytes, bytearray, complex, tuple)):
            state[key] = val
        if isinstance(val, dict):
            state.update(val)
    return state
