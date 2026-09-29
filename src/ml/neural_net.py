"""Feed-forward neural network, forward propagation (book ch08)."""

import numpy as np

from ml.linear_models import add_bias
from ml.logistic import sigmoid

Array = np.ndarray


def forward(X: Array, thetas: list[Array]) -> list[Array]:
    """Forward pass $A^{(l+1)} = S\\left([1, A^{(l)}]\\,\\Theta^{(l)}\\right)$.

    Parameters
    ----------
    X
        Inputs, shape `(m, n)`, *without* bias column.
    thetas
        Weight matrices; `thetas[l]` has shape `(units_l + 1, units_{l+1})`,
        row 0 holding the bias weights.

    Returns
    -------
    The activations of every layer, `[A1 = X, A2, ..., A_out]`.
    """
    activations = [np.asarray(X, dtype=float)]
    for Theta in thetas:
        activations.append(sigmoid(add_bias(activations[-1]) @ Theta))
    return activations
