// Dlib configuration source file
// This provides the necessary symbols for dlib's version checking system

#define DLIB_NO_GUI_SUPPORT
#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

// Include dlib config header to get version info
#include "dlib/revision.h"
#include "dlib/config.h"

// Provide the version check symbol that dlib expects
extern "C" {
    const char DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0 = 0;
    const char USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ = 0;
}
