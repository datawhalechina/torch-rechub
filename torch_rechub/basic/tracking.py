"""Experiment tracking utilities for Torch-RecHub.

This module exposes lightweight adapters for common visualization and
experiment tracking tools, namely Weights & Biases (wandb), SwanLab, and
TensorBoardX.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseLogger(ABC):
    """Base interface for experiment tracking backends.

    Methods
    -------
    log_metrics(metrics, step=None)
        Record scalar metrics at a given step.
    log_hyperparams(params)
        Store hyperparameters and run configuration.
    finish()
        Flush pending logs and release resources.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to the tracking backend.

        Parameters
        ----------
        metrics : dict of str to Any
            Metric name-value pairs to record.
        step : int, optional
            Explicit global step or epoch index. When ``None``, the backend
            uses its own default step handling.
        """
        raise NotImplementedError

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log experiment hyperparameters.

        Parameters
        ----------
        params : dict of str to Any
            Hyperparameters or configuration values to persist with the run.
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        """Finalize logging and free any backend resources."""
        raise NotImplementedError


class WandbLogger(BaseLogger):
    """Weights & Biases logger implementation.

    Parameters
    ----------
    project : str
        Name of the wandb project to log to.
    name : str, optional
        Display name for the run.
    config : dict, optional
        Initial hyperparameter configuration to record.
    tags : list of str, optional
        Optional tags for grouping runs.
    notes : str, optional
        Long-form notes shown in the run overview.
    dir : str, optional
        Local directory for wandb artifacts and cache.
    **kwargs : dict
        Additional keyword arguments forwarded to ``wandb.init``.

    Raises
    ------
    ImportError
        If ``wandb`` is not installed in the current environment.
    """

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None, notes: Optional[str] = None, dir: Optional[str] = None, **kwargs):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")

        self.run = self._wandb.init(project=project, name=name, config=config, tags=tags, notes=notes, dir=dir, **kwargs)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is not None:
            self._wandb.log(metrics, step=step)
        else:
            self._wandb.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self.run is not None:
            self.run.config.update(params)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class SwanLabLogger(BaseLogger):
    """SwanLab logger implementation.

    Parameters
    ----------
    project : str, optional
        Project identifier for grouping experiments.
    experiment_name : str, optional
        Display name for the experiment or run.
    description : str, optional
        Text description shown alongside the run.
    config : dict, optional
        Hyperparameters or configuration to log at startup.
    logdir : str, optional
        Directory where logs and artifacts are stored.
    **kwargs : dict
        Additional keyword arguments forwarded to ``swanlab.init``.

    Raises
    ------
    ImportError
        If ``swanlab`` is not installed in the current environment.
    """

    def __init__(self, project: Optional[str] = None, experiment_name: Optional[str] = None, description: Optional[str] = None, config: Optional[Dict[str, Any]] = None, logdir: Optional[str] = None, **kwargs):
        try:
            import swanlab
            self._swanlab = swanlab
        except ImportError:
            raise ImportError("swanlab is not installed. Install it with: pip install swanlab")

        self.run = self._swanlab.init(project=project, experiment_name=experiment_name, description=description, config=config, logdir=logdir, **kwargs)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is not None:
            self._swanlab.log(metrics, step=step)
        else:
            self._swanlab.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self.run is not None:
            self.run.config.update(params)

    def finish(self) -> None:
        self._swanlab.finish()


class TensorBoardXLogger(BaseLogger):
    """TensorBoardX logger implementation.

    Parameters
    ----------
    log_dir : str
        Directory where event files will be written.
    comment : str, default=""
        Comment appended to the log directory name.
    **kwargs : dict
        Additional keyword arguments forwarded to
        ``tensorboardX.SummaryWriter``.

    Raises
    ------
    ImportError
        If ``tensorboardX`` is not installed in the current environment.
    """

    def __init__(self, log_dir: str, comment: str = "", **kwargs):
        try:
            from tensorboardX import SummaryWriter
            self._SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError("tensorboardX is not installed. Install it with: pip install tensorboardX")

        self.writer = self._SummaryWriter(log_dir=log_dir, comment=comment, **kwargs)
        self._step = 0

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            step = self._step
            self._step += 1

        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        hparam_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        self.writer.add_text("hyperparameters", hparam_str, 0)

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
