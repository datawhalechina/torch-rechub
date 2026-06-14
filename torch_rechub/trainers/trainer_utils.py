"""Small utilities shared by trainer classes."""

import torch


def build_scheduler(optimizer, scheduler_fn, scheduler_params=None):
    """Build a scheduler while allowing ``scheduler_params`` to be omitted."""
    if scheduler_fn is None:
        return None
    if scheduler_params is None:
        scheduler_params = {}
    return scheduler_fn(optimizer, **scheduler_params)


def get_current_lr(optimizer):
    """Return the first param group's learning rate."""
    return optimizer.param_groups[0]["lr"] if optimizer.param_groups else None


def step_scheduler(scheduler, metric=None):
    """Step an epoch-level scheduler and support metric-based schedulers."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            return False
        scheduler.step(metric)
        return True
    scheduler.step()
    return True
