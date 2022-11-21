import logging
from time import time
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        minutes_taken = time_taken // 60
        seconds_taken = time_taken % 60
        if hours_taken:
            message = \
                f"func:{f.__name__} took: {hours_taken:0.0f} hr and " + \
                f"{minutes_taken:0.0f} min"
        elif minutes_taken:
            message = \
                f"func:{f.__name__} took: {minutes_taken:0.0f} min and " + \
                f"{seconds_taken:0.2f} sec"
        else:
            message = f"func:{f.__name__} took: {seconds_taken:0.2f} sec"
        logger.info(message)
        return result
    return wrap


def kspace(X: np.ndarray) -> np.ndarray:
    """
    Transform k-space array to obtain normalized absolute array

    Args:
        X (np.ndarray): _description_

    Returns:
        np.ndarray: Normalized absolute array
    """
    Y = np.abs(X)
    Y = np.average(Y, axis=0)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    return Y


def fft(X: np.ndarray) -> np.ndarray:
    """
    Fourier transform k-space array to obtain MR image array

    Args:
        X (np.ndarray): k-space array

    Returns:
        np.ndarray: MR image array
    """
    Y = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))
    Y = np.abs(Y)
    Y = np.average(Y, axis=0)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    return Y
