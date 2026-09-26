"""Learning-rate schedulers used by U-Net training."""

from math import isclose

import torch.optim as optim


class DiscreteReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce the learning rate through an explicit sequence of levels.

    Plateau detection follows :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
    but each reduction selects the next configured learning rate instead of
    multiplying the current rate by a constant factor.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        learning_rates: list[float],
        mode: str = "min",
        patience: int = 10,
        threshold: float = 1e-4,
    ) -> None:
        rates = [float(rate) for rate in learning_rates]
        if not rates:
            raise ValueError("learning_rates must contain at least one value.")
        if any(rate <= 0 for rate in rates):
            raise ValueError("Every configured learning rate must be positive.")
        if any(current <= following for current, following in zip(rates, rates[1:])):
            raise ValueError("learning_rates must be strictly decreasing.")

        initial_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if any(
            not isclose(rate, rates[0], rel_tol=1e-12, abs_tol=0.0)
            for rate in initial_rates
        ):
            raise ValueError(
                "The optimizer learning rate must equal the first configured "
                f"scheduler level ({rates[0]:.6e}); received {initial_rates}."
            )

        self.learning_rates = rates
        self.level_index = 0
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=0.5,
            patience=patience,
            threshold=threshold,
            min_lr=rates[-1],
        )

    def _reduce_lr(self, epoch: int) -> None:
        """Advance every optimizer parameter group to the next allowed level."""

        if self.level_index >= len(self.learning_rates) - 1:
            return

        self.level_index += 1
        next_learning_rate = self.learning_rates[self.level_index]
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = next_learning_rate
