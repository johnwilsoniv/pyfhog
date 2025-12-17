# pyfhog

Python wrapper for dlib's FHOG (Felzenszwalb HOG) feature extraction.

## Installation

```bash
pip install pyfhog
```

## Usage

```python
import pyfhog

features = pyfhog.extract(image, cell_size=8)
```

## What it does

Extracts FHOG features identical to OpenFace 2.2 (validated r=1.0, RMSE=0.0). Uses dlib's optimized C++ SIMD implementation under the hood.

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 — free for non-commercial use with attribution.
