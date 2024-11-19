import collections
import random
from typing import List
import numpy as np
import torch

import mnn.logging


class MyLRScheduler:

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.param_groups_initial_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_groups_initial_lrs.append(param_group["lr"])

        self.max_len = 100
        self.losses = collections.deque(maxlen=self.max_len)
        self.loss_moving_average = collections.deque(maxlen=self.max_len)
        self.logger = mnn.logging.get_logger("MyLRScheduler")

    def _reset_moving_average(self):
        self.loss_moving_average = collections.deque(maxlen=self.max_len)

    def _fit_line(self, data_points: List[float]):
        x = [i for i in range(len(data_points))]
        y = data_points
        m, b = np.polyfit(x, y, 1)
        return m, b

    def add_batch_loss(self, loss: torch.nn.Module):
        self.losses.append(loss.item())
        current_mean = torch.Tensor(self.losses).mean()
        self.loss_moving_average.append(current_mean)
        if len(self.loss_moving_average) == self.loss_moving_average.maxlen:
            line_angle, b = self._fit_line(self.loss_moving_average)
            if line_angle > 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    temp = param_group["lr"]
                    perc = random.uniform(0.85, 0.95)
                    param_group["lr"] *= perc
                    if param_group["lr"] < 0.000001:
                        param_group["lr"] = self.param_groups_initial_lrs[i] / 10
                    self.logger.info(
                        f"Updating 'lr' for param_group-{i} from '{temp:.7f}' to {param_group['lr']:.7f} "
                    )
            self._reset_moving_average()
