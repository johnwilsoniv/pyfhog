// Dlib configuration source file
// This provides the necessary symbols for dlib's version checking system
//
// These symbols are referenced by dlib/test_for_odr_violations.h and must be
// defined exactly once in the compiled code.

#define DLIB_NO_GUI_SUPPORT

// Include dlib config to trigger the configuration
#include "dlib/config.h"

// Provide the version check and configuration symbols that dlib expects
// These must be in extern "C" block and must be const int (not const char)
extern "C" {
    // Version mismatch check - from DLIB_CHECK_FOR_VERSION_MISMATCH
    const int DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0 = 0;

    // Build configuration check - trailing underscore because DLIB_DISABLE_ASSERTS is set
    const int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ = 0;
}
