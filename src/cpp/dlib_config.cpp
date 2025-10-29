// Dlib configuration source file
// This provides the DEFINITIONS for dlib's version checking symbols
//
// These symbols are DECLARED as "const extern int" in dlib/test_for_odr_violations.h
// and must be DEFINED exactly once in the compiled code.
//
// IMPORTANT: We do NOT include any dlib headers here - that would create references,
// not definitions. We just define the symbols directly.

extern "C" {
    // Version mismatch check - from DLIB_CHECK_FOR_VERSION_MISMATCH macro in dlib/config.h
    const int DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0 = 0;

    // Build configuration check - trailing underscore because DLIB_DISABLE_ASSERTS is set in dlib/config.h
    // (without ENABLE_ASSERTS, the header expects the symbol with trailing underscore)
    const int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ = 0;
}
