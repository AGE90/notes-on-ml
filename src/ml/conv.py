"""Cross-correlation, the "convolution" of CNN layers (book ch11)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

Array = np.ndarray


def cross_correlate(x: Array, y: Array, stride: int = 1, padding: int = 0) -> Array:
    """Sliding inner product $z_j = \\sum_i x_i\\, y_{i + j}$ of filter `x` over `y`.

    Works for any rank: 1-D signals, 2-D images and 3-D (height, width, channel)
    tensors. A filter as deep as the input (e.g. 3 channels for RGB) sums over
    channels, as in a CNN layer.

    Parameters
    ----------
    x : Array
        Filter (kernel), same rank as `y` and no larger along any axis.
    y : Array
        Input.
    stride : int
        Step $s$ between output samples.
    padding : int
        Zeros added on both sides of every axis of `y`.

    Returns
    -------
    Array
        Output with $(N_y - N_x + 2p) / s + 1$ samples (floor) along each axis.
    """
    y = np.pad(y, padding)
    windows = sliding_window_view(y, x.shape)
    windows = windows[tuple(slice(None, None, stride) for _ in range(y.ndim))]
    axes = tuple(range(y.ndim, 2 * y.ndim))
    return np.tensordot(windows, x, axes=(axes, tuple(range(x.ndim))))
