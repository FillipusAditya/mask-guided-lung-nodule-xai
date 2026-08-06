"""Early-stopping state management for model training."""

import torch


class EarlyStopping:
    """
    Track metric improvements and signal when training should stop.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs to wait after the last improvement before stopping.

    mode : str, default="min"
        Optimization mode.

        - "min" : lower metric is better (e.g. validation loss)
        - "max" : higher metric is better (e.g. Dice Score, Accuracy)

    min_delta : float, default=0.0
        Minimum improvement required to reset the patience counter.

    verbose : bool, default=True
        Whether to print the patience counter after a non-improving epoch.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        """Initialize the early-stopping controller."""

        if mode not in ("min", "max"):
            raise ValueError("mode must be either 'min' or 'max'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False

        self.best_score = None
        self.best_epoch = None

    def __call__(
        self,
        metric: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> bool:
        """
        Update the early-stopping state using a validation metric.

        Parameters
        ----------
        metric : float
            Validation metric.

        model : torch.nn.Module
            Model associated with the validation metric. It is accepted to keep
            the training-loop interface consistent; this class stores only
            early-stopping state.

        epoch : int
            Current epoch.

        Returns
        -------
        bool
            True if training should stop.
        """

        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            return False

        improved = self._is_improved(metric)

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improved(
        self,
        metric: float,
    ) -> bool:
        """
        Return whether ``metric`` improves on the best recorded score.
        """

        if self.mode == "min":
            return metric < (self.best_score - self.min_delta)

        return metric > (self.best_score + self.min_delta)

    def state_dict(self) -> dict:
        """
        Return a serializable snapshot of the early-stopping state.
        """

        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }
    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        """
        Restore state previously returned by :meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict
            Serialized early-stopping state.
        """

        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict["best_epoch"]
        self.early_stop = state_dict["early_stop"]
