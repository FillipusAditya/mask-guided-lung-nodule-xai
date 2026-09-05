"""Compatibility helpers for pylidc 0.2.3 on current NumPy releases."""

import numpy as np


def enable_pylidc_numpy_compatibility() -> None:
    """Restore aliases still referenced internally by pylidc 0.2.3.

    NumPy removed ``np.int``, ``np.float``, and ``np.bool`` after long
    deprecation periods. pylidc 0.2.3 still uses these names while clustering
    annotations and constructing masks. Adding the aliases here keeps the
    compatibility workaround scoped to processes that run this pipeline.
    """
    aliases = {
        "bool": np.bool_,
        "float": float,
        "int": int,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

