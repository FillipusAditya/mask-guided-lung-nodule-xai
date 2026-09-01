"""Convert classification probabilities into discrete predictions."""

from numbers import Integral, Real

import torch
from torch import Tensor


def binary_probabilities_to_predictions(
    probabilities: Tensor,
    threshold: float = 0.5,
    positive_class_index: int = 1,
) -> Tensor:
    """Convert binary probabilities to thresholded class indices.

    A sample is assigned to the positive class when its positive-class
    probability is greater than or equal to ``threshold``. Otherwise, it is
    assigned to the other binary class.

    Parameters
    ----------
    probabilities : Tensor
        Probability tensor with shape ``[batch_size, 2]``.
    threshold : float, default=0.5
        Inclusive decision threshold for the positive-class probability.
    positive_class_index : int, default=1
        Probability column and class index treated as positive.

    Returns
    -------
    Tensor
        One-dimensional ``torch.long`` tensor of predicted class indices on
        the same device as ``probabilities``.
    """

    if not isinstance(probabilities, Tensor):
        raise TypeError("probabilities must be a torch.Tensor.")

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "probabilities must have shape [batch_size, 2], but received "
            f"{tuple(probabilities.shape)}."
        )

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number.")

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    if (
        isinstance(positive_class_index, bool)
        or not isinstance(positive_class_index, Integral)
        or positive_class_index not in (0, 1)
    ):
        raise ValueError("positive_class_index must be either 0 or 1.")

    positive_class_index = int(positive_class_index)
    negative_class_index = 1 - positive_class_index
    output_shape = probabilities.shape[:1]
    positive_predictions = torch.full(
        output_shape,
        positive_class_index,
        dtype=torch.long,
        device=probabilities.device,
    )
    negative_predictions = torch.full(
        output_shape,
        negative_class_index,
        dtype=torch.long,
        device=probabilities.device,
    )

    return torch.where(
        probabilities[:, positive_class_index] >= threshold,
        positive_predictions,
        negative_predictions,
    )
