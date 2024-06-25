from datetime import datetime

import torch
from torch.optim.lr_scheduler import ChainedScheduler, ReduceLROnPlateau


def myprint(text, func=""):
    """
    Prints a formatted string with the current timestamp and optional function name.
    
    Args:
        text (str): The message to print.
        func (str, optional): The name of the function calling myprint. Defaults to "".
    """
    hour = str(datetime.now())
    if func != "":
        func = "[ {} ]".format(func)
    text_to_print = "{} {} {}".format(hour, func, text)
    print(text_to_print)


class LROnStagnationPlateau(ReduceLROnPlateau):
    """
    Custom learning rate scheduler that reduces the learning rate when a metric has stopped improving.
    
    Extends:
        ReduceLROnPlateau
    """
    def step(self, metrics, epoch=None):
        """
        Update learning rate based on the provided metrics.
        
        Args:
            metrics (float or Tensor): The value of the monitored metric.
            epoch (int, optional): The current epoch number. If None, uses the last epoch + 1.
        """
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        elif self.is_worse(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def is_worse(self, a, best):
        """
        Determine if the current metric is worse than the best recorded value based on the mode and threshold.
        
        Args:
            a (float): The current metric value.
            best (float): The best recorded metric value.
        
        Returns:
            bool: True if the current metric is worse than the best, False otherwise.
        """
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 + self.threshold
            return a > best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a > best + self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold - 1.0
            return a < best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a < best - self.threshold


class MetricsChainedScheduler(ChainedScheduler):
    """
    Custom chained scheduler that steps through a list of schedulers using a common metric.
    
    Extends:
        ChainedScheduler
    """
    def step(self, metrics):
        for scheduler in self._schedulers:
            scheduler.step(metrics)
        self._last_lr = [
            group["lr"] for group in self._schedulers[-1].optimizer.param_groups
        ]


def set_device():
    """
    Sets the default device for PyTorch to CUDA if available, otherwise to CPU.
    
    Returns:
        torch.device: The device being used (CUDA or CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print("Using device:", device)
    return device

def expand_properties_tensor(tensor):
    """
    Expands a tensor to the shape (1, 3, 128, 128) by adding dimensions and then expanding.
    
    Args:
        tensor (torch.Tensor): The input tensor.
    
    Returns:
        torch.Tensor: The expanded tensor.
    """
    tensor = tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor.expand(1, 3, 128, 128)
