"""
PyFHOG - Fast Felzenszwalb HOG feature extraction for Python

Provides a minimal wrapper around dlib's optimized FHOG implementation.

Example:
    >>> import pyfhog
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    >>> features = pyfhog.extract_fhog_features(img)
    >>> features.shape
    (4464,)  # For 96x96 image with cell_size=8
"""

from ._pyfhog import extract_fhog_features, __version__

__all__ = ['extract_fhog_features', '__version__']
