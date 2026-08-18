"""Early-stopping state management for classification training."""


class EarlyStopping:
    """Track validation-metric improvements and signal when training stops."""

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

        if patience < 1:
            raise ValueError("patience must be at least 1.")

        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False
        self.best_score: float | None = None
        self.best_epoch: int | None = None

    def __call__(
        self,
        metric: float,
        epoch: int,
    ) -> bool:
        """Update the state and return whether training should stop."""

        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            return False

        if self._is_improved(metric):
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

    def reset_counter(self) -> None:
        """Reset patience while retaining the best metric and its epoch."""

        self.counter = 0
        self.early_stop = False

    def _is_improved(self, metric: float) -> bool:
        """Return whether the metric sufficiently improves on the best value."""

        if self.mode == "min":
            return metric < (self.best_score - self.min_delta)

        return metric > (self.best_score + self.min_delta)

    def state_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the early-stopping state."""

        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict["best_epoch"]
        self.early_stop = state_dict["early_stop"]
