"""
Experiment Tracking Module for Torch-RecHub.

This module provides interfaces for experiment tracking tools including
Weights & Biases (wandb), SwanLab, and TensorBoardX.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import warnings


class BaseLogger(ABC):
    """Abstract base class for experiment tracking loggers."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to the tracking platform."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the tracking platform."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish logging and cleanup resources."""
        pass


class WandbLogger(BaseLogger):
    """Weights & Biases logger implementation."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
        **kwargs
    ):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self.run = self._wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=dir,
            **kwargs
        )

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
    """SwanLab logger implementation."""

    def __init__(
        self,
        project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logdir: Optional[str] = None,
        **kwargs
    ):
        try:
            import swanlab
            self._swanlab = swanlab
        except ImportError:
            raise ImportError(
                "swanlab is not installed. Install it with: pip install swanlab"
            )

        self.run = self._swanlab.init(
            project=project,
            experiment_name=experiment_name,
            description=description,
            config=config,
            logdir=logdir,
            **kwargs
        )

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
    """TensorBoardX logger implementation."""

    def __init__(
        self,
        log_dir: str,
        comment: str = "",
        **kwargs
    ):
        try:
            from tensorboardX import SummaryWriter
            self._SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError(
                "tensorboardX is not installed. Install it with: pip install tensorboardX"
            )

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
