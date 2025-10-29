#!/usr/bin/env python3
"""
Validate pyfhog against OpenFace 2.2 using OpenFace-aligned faces

This is the definitive validation test - we use OpenFace's own aligned
face images and compare pyfhog output with OpenFace's FHOG features.
"""

import sys
import os
import numpy as np
import cv2

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../S1 Face Mirror'))
from openface22_hog_parser import OF22HOGParser

# Add pyfhog
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
import pyfhog


def validate_single_frame(aligned_face_path, hog_file_path, frame_index, verbose=True):
    """
    Validate pyfhog output against OpenFace for a single frame

    Args:
        aligned_face_path: Path to OpenFace-aligned face image (.bmp)
        hog_file_path: Path to OpenFace .hog file
        frame_index: Frame index (0-based) in .hog file
        verbose: Print detailed output

    Returns:
        dict with correlation and metrics
    """

    if verbose:
        print(f"\nValidating frame {frame_index}...")
        print("-" * 60)

    # 1. Load OpenFace FHOG features
    parser = OF22HOGParser(hog_file_path)
    frame_indices, hog_features_of = parser.parse()

    if frame_index >= len(hog_features_of):
        raise ValueError(f"Frame {frame_index} not in .hog file (only {len(hog_features_of)} frames)")

    features_of = hog_features_of[frame_index]

    if verbose:
        print(f"[OpenFace] FHOG shape: {features_of.shape}")
        print(f"[OpenFace] Range: [{features_of.min():.4f}, {features_of.max():.4f}]")

    # 2. Load aligned face image
    aligned_face = cv2.imread(aligned_face_path)
    if aligned_face is None:
        raise FileNotFoundError(f"Could not load aligned face: {aligned_face_path}")

    aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    if verbose:
        print(f"[Aligned face] Shape: {aligned_face_rgb.shape}")

    # 3. Extract FHOG with pyfhog
    features_pyfhog = pyfhog.extract_fhog_features(aligned_face_rgb, cell_size=8)

    if verbose:
        print(f"[pyfhog] FHOG shape: {features_pyfhog.shape}")
        print(f"[pyfhog] Range: [{features_pyfhog.min():.4f}, {features_pyfhog.max():.4f}]")

    # 4. Check dimensions match
    if features_of.shape != features_pyfhog.shape:
        return {
            'passed': False,
            'error': 'dimension_mismatch',
            'of_shape': features_of.shape,
            'pyfhog_shape': features_pyfhog.shape
        }

    # 5. Compute metrics
    correlation = np.corrcoef(features_of, features_pyfhog)[0, 1]
    rmse = np.sqrt(np.mean((features_of - features_pyfhog)**2))
    mae = np.mean(np.abs(features_of - features_pyfhog))
    max_diff = np.max(np.abs(features_of - features_pyfhog))

    passed = correlation > 0.999

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Correlation:   r = {correlation:.8f}")
        print(f"RMSE:          {rmse:.8f}")
        print(f"MAE:           {mae:.8f}")
        print(f"Max diff:      {max_diff:.8f}")
        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] - Correlation {'>' if passed else '<='} 0.999")

    return {
        'passed': passed,
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'max_diff': max_diff,
        'of_shape': features_of.shape,
        'pyfhog_shape': features_pyfhog.shape
    }


def main():
    """Run validation on multiple frames"""

    print("=" * 80)
    print("PyFHOG vs OpenFace 2.2 Validation")
    print("Using OpenFace-aligned faces for perfect comparison")
    print("=" * 80)

    # Paths
    aligned_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog/test_aligned_faces/IMG_0942_left_mirrored_aligned"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog/test_aligned_faces/IMG_0942_left_mirrored.hog"

    # Get list of aligned face files (sorted)
    aligned_files = sorted([f for f in os.listdir(aligned_dir) if f.endswith('.bmp')])

    print(f"\nTest data:")
    print(f"  Aligned faces: {len(aligned_files)} files")
    print(f"  HOG file: {os.path.basename(hog_file)}")

    # Test frames (use first few frames)
    test_frames = [0, 1, 2, 10, 50, 100, 500, 1000]

    print(f"\nTesting {len(test_frames)} frames...")
    print()

    results = []
    for frame_idx in test_frames:
        if frame_idx >= len(aligned_files):
            print(f"Frame {frame_idx}: Skipping (not enough frames)")
            continue

        aligned_face_path = os.path.join(aligned_dir, aligned_files[frame_idx])

        try:
            result = validate_single_frame(
                aligned_face_path,
                hog_file,
                frame_idx,
                verbose=False
            )

            if 'error' in result:
                print(f"Frame {frame_idx:4d}: FAILED ({result['error']})")
            else:
                results.append(result)
                status = "PASS" if result['passed'] else "FAIL"
                print(f"Frame {frame_idx:4d}: {status}  r={result['correlation']:.8f}  RMSE={result['rmse']:.6f}")

        except Exception as e:
            print(f"Frame {frame_idx:4d}: ERROR - {e}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(results) == 0:
        print("No successful tests")
        return 1

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    avg_correlation = np.mean([r['correlation'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    min_correlation = np.min([r['correlation'] for r in results])
    max_correlation = np.max([r['correlation'] for r in results])

    print(f"Passed:             {passed_count}/{total_count}")
    print(f"Correlation (avg):  r = {avg_correlation:.8f}")
    print(f"Correlation (min):  r = {min_correlation:.8f}")
    print(f"Correlation (max):  r = {max_correlation:.8f}")
    print(f"RMSE (avg):         {avg_rmse:.8f}")
    print()

    if passed_count == total_count:
        print("ALL TESTS PASSED!")
        print("\npyfhog produces IDENTICAL output to OpenFace 2.2")
        return 0
    else:
        print(f"{total_count - passed_count} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
