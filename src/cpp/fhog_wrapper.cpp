#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dlib/image_transforms/fhog.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"

namespace py = pybind11;

// Convert NumPy array to dlib image
dlib::array2d<dlib::bgr_pixel> numpy_to_dlib_image(
    py::array_t<uint8_t> img_array
) {
    auto buf = img_array.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input must be HxWx3 BGR image");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];

    dlib::array2d<dlib::bgr_pixel> img(height, width);
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = (r * width + c) * 3;
            img[r][c].blue = ptr[idx];
            img[r][c].green = ptr[idx + 1];
            img[r][c].red = ptr[idx + 2];
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

    // Flatten in same order as OpenFace (col, row, orientation)
    // Column-major order: iterate columns first, then rows (matching OpenFace)
    int idx = 0;
    for (int y = 0; y < num_cols; ++y) {
        for (int x = 0; x < num_rows; ++x) {
            for (int o = 0; o < num_features; ++o) {
                ptr[idx++] = hog[x][y](o);  // hog is indexed as [row][col]
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
                  image: NumPy array of shape (H, W, 3) in BGR format (OpenCV default), dtype=uint8
                  cell_size: Size of HOG cells in pixels (default: 8)

              Returns:
                  1D NumPy array of FHOG features (flattened in column-major order matching OpenFace)

              Example:
                  >>> import pyfhog
                  >>> import cv2
                  >>> img = cv2.imread('face.jpg')  # BGR format
                  >>> features = pyfhog.extract_fhog_features(img)
                  >>> features.shape
                  (4464,)  # For 112x112 image with cell_size=8
          )pbdoc");

    m.attr("__version__") = "0.1.0";
}
