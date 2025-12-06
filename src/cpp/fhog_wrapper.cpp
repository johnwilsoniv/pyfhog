#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dlib/image_transforms/fhog.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"

namespace py = pybind11;

// A minimal cv_image-like wrapper that directly references NumPy data
// This matches how OpenFace's cv_image wraps OpenCV Mat data
template <typename pixel_type>
class numpy_image_wrapper {
public:
    numpy_image_wrapper(py::array_t<uint8_t>& arr) {
        auto buf = arr.request();
        _data = static_cast<uint8_t*>(buf.ptr);
        _nr = buf.shape[0];
        _nc = buf.shape[1];
        // NumPy stride is in bytes per row
        _widthStep = buf.strides[0];
    }

    long nr() const { return _nr; }
    long nc() const { return _nc; }
    long width_step() const { return _widthStep; }

    const pixel_type* operator[](long row) const {
        return reinterpret_cast<const pixel_type*>(_data + _widthStep * row);
    }

    pixel_type* operator[](long row) {
        return reinterpret_cast<pixel_type*>(_data + _widthStep * row);
    }

private:
    uint8_t* _data;
    long _nr;
    long _nc;
    long _widthStep;
};

// Make numpy_image_wrapper compatible with dlib's generic image concept
// These must be in the dlib namespace to be found by ADL
namespace dlib {
    template <typename T>
    struct image_traits<numpy_image_wrapper<T>> {
        typedef T pixel_type;
    };

    template <typename T>
    inline long num_rows(const numpy_image_wrapper<T>& img) { return img.nr(); }

    template <typename T>
    inline long num_columns(const numpy_image_wrapper<T>& img) { return img.nc(); }

    template <typename T>
    inline void* image_data(numpy_image_wrapper<T>& img) {
        if (img.nr() != 0 && img.nc() != 0)
            return &img[0][0];
        return nullptr;
    }

    template <typename T>
    inline const void* image_data(const numpy_image_wrapper<T>& img) {
        if (img.nr() != 0 && img.nc() != 0)
            return &img[0][0];
        return nullptr;
    }

    template <typename T>
    inline long width_step(const numpy_image_wrapper<T>& img) {
        return img.width_step();
    }
}

// Convert NumPy array to dlib image (fallback for copy-based approach)
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

    // Match OpenFace Face_utils.cpp line 259-268 EXACTLY:
    // num_cols = hog.nc();
    // num_rows = hog.nr();
    // for(int y = 0; y < num_cols; ++y) {
    //     for(int x = 0; x < num_rows; ++x) {
    //         for(unsigned int o = 0; o < 31; ++o) {
    //             *descriptor_it++ = (double)hog[y][x](o);
    //
    // NOTE: dlib::array2d uses [row][col] indexing, but OpenFace uses [y][x]
    // where y is column index and x is row index. This is the same as what we had.
    // The actual issue must be elsewhere.
    int idx = 0;
    for (int y = 0; y < num_cols; ++y) {
        for (int x = 0; x < num_rows; ++x) {
            for (int o = 0; o < num_features; ++o) {
                ptr[idx++] = hog[y][x](o);
            }
        }
    }

    return result;
}

// Main extraction function using zero-copy wrapper (like cv_image)
py::array_t<double> extract_fhog_features(
    py::array_t<uint8_t> image,
    int cell_size = 8
) {
    auto buf = image.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input must be HxWx3 BGR image");
    }

    // Use zero-copy wrapper that directly references NumPy memory
    // This matches how OpenFace's cv_image wraps OpenCV Mat data
    numpy_image_wrapper<dlib::bgr_pixel> wrapped_img(image);

    // Extract FHOG using dlib (same function, different image wrapper)
    dlib::array2d<dlib::matrix<float,31,1>> hog;
    dlib::extract_fhog_features(wrapped_img, hog, cell_size);

    // Convert back to NumPy
    return dlib_hog_to_numpy(hog);
}

// Legacy function using copy-based approach (for comparison)
py::array_t<double> extract_fhog_features_copy(
    py::array_t<uint8_t> image,
    int cell_size = 8
) {
    // Convert NumPy to dlib format (makes a copy)
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
