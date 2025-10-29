#!/usr/bin/env python3
"""
Basic test for pyfhog - Validates FHOG extraction functionality
"""

import numpy as np
import sys

def test_import():
    """Test that pyfhog can be imported"""
    try:
        import pyfhog
        print(f"Successfully imported pyfhog version {pyfhog.__version__}")
        return True
    except ImportError as e:
        print(f"Failed to import pyfhog: {e}")
        return False

def test_basic_extraction():
    """Test basic FHOG extraction on random image"""
    try:
        import pyfhog

        # Create test image (96x96 RGB)
        img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

        # Extract FHOG features
        features = pyfhog.extract_fhog_features(img, cell_size=8)

        # Validate output
        expected_dims = (96 // 8) * (96 // 8) * 31  # 12 * 12 * 31 = 4464

        if features.shape == (expected_dims,):
            print(f"FHOG extraction successful: {features.shape}")
            print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  Feature mean: {features.mean():.4f}")
            return True
        else:
            print(f"Unexpected feature shape: {features.shape}, expected ({expected_dims},)")
            return False

    except Exception as e:
        print(f"FHOG extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_sizes():
    """Test FHOG extraction with different image sizes"""
    try:
        import pyfhog

        test_cases = [
            (64, 64, 8, 8 * 8 * 31),   # 64x64 with cell_size=8 -> 8x8 cells
            (96, 96, 8, 12 * 12 * 31),  # 96x96 with cell_size=8 -> 12x12 cells
            (128, 128, 8, 16 * 16 * 31), # 128x128 with cell_size=8 -> 16x16 cells
        ]

        all_passed = True
        for h, w, cell_size, expected_dims in test_cases:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            features = pyfhog.extract_fhog_features(img, cell_size=cell_size)

            if features.shape == (expected_dims,):
                print(f"{h}x{w} with cell_size={cell_size} -> {features.shape}")
            else:
                print(f"{h}x{w} with cell_size={cell_size}: got {features.shape}, expected ({expected_dims},)")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"Size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("PyFHOG Basic Validation Tests")
    print("="*80)
    print()

    tests = [
        ("Import Test", test_import),
        ("Basic Extraction", test_basic_extraction),
        ("Different Sizes", test_different_sizes),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        print("-" * 40)
        passed = test_func()
        results.append((name, passed))
        print()

    print("="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
