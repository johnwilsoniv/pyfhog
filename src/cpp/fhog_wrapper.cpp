// Disable dlib's version mismatch and assertion checking to avoid linker issues
// These must be defined before including any dlib headers
#define USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ DLIB_NO_WARN_UNUSED static const int dlib_check_assert_value = 0
#define DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0 DLIB_NO_WARN_UNUSED static const int dlib_check_version_value = 0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dlib/image_transforms/fhog.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"

namespace py = pybind11;

// Convert NumPy array to dlib image
dlib::array2d<dlib::rgb_pixel> numpy_to_dlib_image(
    py::array_t<uint8_t> img_array
) {
    auto buf = img_array.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input must be HxWx3 RGB image");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];

    dlib::array2d<dlib::rgb_pixel> img(height, width);
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = (r * width + c) * 3;
            img[r][c].red = ptr[idx];
            img[r][c].green = ptr[idx + 1];
            img[r][c].blue = ptr[idx + 2];
        }
    }

    return img;
}

// Convert dlib FHOG output to NumPy array
py::array_t<double> dlib_hog_to_numpy(
    const dlib::array2d<dlib::matrix<float,31,1>>& hog
) {
    int num_rows = hog.nr();
    int num_cols = hog.nc();
    int num_features = 31;

    // Allocate NumPy array
    auto result = py::array_t<double>(num_rows * num_cols * num_features);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Flatten in same order as OpenFace (row, col, orientation)
    // Row-major order: iterate rows first, then columns
    int idx = 0;
    for (int x = 0; x < num_rows; ++x) {
        for (int y = 0; y < num_cols; ++y) {
            for (int o = 0; o < num_features; ++o) {
                ptr[idx++] = hog[x][y](o);
            }
        }
    }

    return result;
}

// Main extraction function
py::array_t<double> extract_fhog_features(
    py::array_t<uint8_t> image,
    int cell_size = 8
) {
    // Convert NumPy to dlib format
    auto dlib_img = numpy_to_dlib_image(image);

    // Extract FHOG using dlib
    dlib::array2d<dlib::matrix<float,31,1>> hog;
    dlib::extract_fhog_features(dlib_img, hog, cell_size);

    // Convert back to NumPy
    return dlib_hog_to_numpy(hog);
}

// Python module definition
PYBIND11_MODULE(_pyfhog, m) {
    m.doc() = "Fast FHOG feature extraction using dlib";

    m.def("extract_fhog_features", &extract_fhog_features,
          py::arg("image"),
          py::arg("cell_size") = 8,
          R"pbdoc(
              Extract Felzenszwalb HOG features from an image.

              Args:
                  image: NumPy array of shape (H, W, 3) in RGB format, dtype=uint8
                  cell_size: Size of HOG cells in pixels (default: 8)

              Returns:
                  1D NumPy array of FHOG features (flattened)

              Example:
                  >>> import pyfhog
                  >>> import numpy as np
                  >>> img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
                  >>> features = pyfhog.extract_fhog_features(img)
                  >>> features.shape
                  (4464,)  # For 96x96 image with cell_size=8
          )pbdoc");

    m.attr("__version__") = "0.1.0";
}
