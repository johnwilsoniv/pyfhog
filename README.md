# PyFHOG - Fast Felzenszwalb HOG Feature Extraction

A minimal Python package that wraps dlib's optimized FHOG (Felzenszwalb Histogram of Oriented Gradients) implementation for cross-platform distribution.

**Validated:** Perfect correlation (r = 1.0, RMSE = 0.0) with OpenFace 2.2

## Features

- **Fast**: Full C++ SIMD performance via dlib
- **Cross-platform**: Distributes as wheels for Mac/Linux/Windows
- **Simple API**: Single function for FHOG extraction
- **Compatible**: Produces identical output to OpenFace 2.2 (validated r = 1.0)

## Installation

```bash
pip install pyfhog
```

## Usage

```python
import pyfhog
import numpy as np
import cv2

# Load image (must be RGB format)
img = cv2.imread('face.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract FHOG features
features = pyfhog.extract_fhog_features(img_rgb, cell_size=8)

# For 112x112 image (OpenFace standard): features.shape = (4464,)
# Formula: (112/8 - 2) * (112/8 - 2) * 31 = 12 * 12 * 31 = 4464
# Note: dlib excludes border cells, hence the -2
```

## FHOG Details

FHOG (Felzenszwalb HOG) is an enhanced version of standard HOG with:
- **31 features per cell** (vs 9 for standard HOG)
  - 18 signed orientation bins
  - 9 unsigned orientation bins
  - 4 texture energy features
- **Cell size**: Typically 8x8 pixels
- **Output**: Flattened 1D array of features

## Why PyFHOG?

PyFHOG solves the cross-platform distribution problem for FHOG extraction:

| Approach | Pros | Cons |
|----------|------|------|
| **Pure Python** | No compilation | 20-40% slower, accuracy risk |
| **Platform binaries** | Fast | Hard to distribute, platform-specific |
| **PyFHOG (wheels)** | Fast, Cross-platform, Easy install | - |

## Technical Details

- **Based on**: dlib's `extract_fhog_features()` implementation
- **SIMD optimized**: Uses SSE/AVX on x86, NEON on ARM
- **Wheel size**: ~200-500KB (minimal overhead)
- **Python binding**: pybind11

## Building from Source

```bash
git clone https://github.com/johnwilsoniv/pyfhog.git
cd pyfhog
pip install pybind11 numpy
python setup.py build_ext --inplace
python -m pytest tests/
```

## License

Boost Software License 1.0 (matches dlib's license)

## Credits

- **dlib**: C++ FHOG implementation by Davis King
- **OpenFace**: Reference implementation by Tadas Baltrušaitis
