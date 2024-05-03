import numpy as np 

class Scheduler:
    def __init__(self, k=None):
        self.k = k 

    def __call__(self, curr_step):
        return 1.0


class CosineScheduler(Scheduler):
    def __call__(self, curr_step):
        return np.cos((np.pi/2) * (1/self.k) * curr_step)


class InverseSigmoidScheduler(Scheduler):
    def __call__(self, curr_step):
        return self.k / (self.k + np.exp(curr_step / self.k))