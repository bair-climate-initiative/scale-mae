import numpy as np


class ResolutionScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def get_target_size(self, epoch):
        raise NotImplemented


class ConstantResolutionScheduler(ResolutionScheduler):
    def __init__(self, target_size):
        self.target_size = target_size

    def get_target_size(self, epoch):
        return self.target_size


class RandomResolutionScheduler(ResolutionScheduler):
    def __init__(self, target_size, n=1):
        self.target_size = target_size
        self.n = n

    def get_target_size(self, epoch):
        return sorted(np.random.choice(self.target_size, self.n).tolist(), reverse=True)
