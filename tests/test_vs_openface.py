#!/usr/bin/env python3
"""
Validate pyfhog output against OpenFace 2.2 .hog files

This test extracts FHOG features using pyfhog and compares them with
OpenFace's output to ensure perfect compatibility.
"""

import sys
import os
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../S1 Face Mirror'))

from openface22_hog_parser import OF22HOGParser

# Add pyfhog to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
import pyfhog


def extract_aligned_face_from_video(video_path, csv_path, frame_num):
    """
    Extract and align a face from a specific video frame using landmarks from CSV

    This mimics OpenFace's face alignment process.
    """
    import pandas as pd

    # Load CSV with landmarks
    df = pd.read_csv(csv_path)

    # Get frame data
    frame_data = df[df['frame'] == frame_num].iloc[0]

    # Extract 68 landmarks (x_0...x_67, y_0...y_67)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = frame_data[f'x_{i}']
        landmarks[i, 1] = frame_data[f'y_{i}']

    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # CSV uses 1-indexed frames
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_num}")

    # OpenFace aligns to 112x112 centered face
    # Use similarity transform based on outer eye corners
    # Landmarks: 36 (left eye outer), 45 (right eye outer)
    left_eye = landmarks[36]
    right_eye = landmarks[45]

    # Compute eye center and angle
    eye_center = (left_eye + right_eye) / 2.0
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    # Compute scale to normalize inter-eye distance
    eye_distance = np.sqrt(dx**2 + dy**2)
    target_eye_distance = 0.35 * 112  # OpenFace uses 35% of image width (112 pixels)
    scale = target_eye_distance / eye_distance

    # Create affine transformation matrix
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)

    # Adjust translation to center eyes
    M[0, 2] += 112 / 2 - eye_center[0]
    M[1, 2] += 112 / 2 - eye_center[1] - 112 * 0.1  # Slight offset downward

    # Warp image
    aligned = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

    return aligned


def compare_pyfhog_vs_openface(
    hog_file_path,
    video_path,
    csv_path,
    frame_num=1,
    verbose=True
):
    """
    Compare pyfhog output with OpenFace .hog file

    Args:
        hog_file_path: Path to OpenFace .hog file
        video_path: Path to input video
        csv_path: Path to OpenFace CSV with landmarks
        frame_num: Frame number to test (1-indexed)
        verbose: Print detailed comparison

    Returns:
        dict with correlation, RMSE, MAE, and pass/fail status
    """

    if verbose:
        print("="*80)
        print(f"Validating pyfhog vs OpenFace (Frame {frame_num})")
        print("="*80)
        print()

    # 1. Load OpenFace FHOG features
    if verbose:
        print("[1/3] Loading OpenFace .hog file...")

    parser = OF22HOGParser(hog_file_path)
    frame_indices, hog_features_of = parser.parse()

    # Get features for specified frame (0-indexed in arrays)
    features_of = hog_features_of[frame_num - 1]

    if verbose:
        print(f"   Loaded OpenFace FHOG: {features_of.shape}")
        print(f"   Feature range: [{features_of.min():.4f}, {features_of.max():.4f}]")
        print()

    # 2. Extract aligned face from video
    if verbose:
        print("[2/3] Extracting and aligning face from video...")

    try:
        aligned_face = extract_aligned_face_from_video(video_path, csv_path, frame_num)

        # Convert BGR to RGB (OpenFace uses RGB)
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        if verbose:
            print(f"   Aligned face: {aligned_face_rgb.shape}")
            print()

    except Exception as e:
        print(f"   Failed to extract aligned face: {e}")
        return None

    # 3. Extract FHOG with pyfhog
    if verbose:
        print("[3/3] Extracting FHOG with pyfhog...")

    features_pyfhog = pyfhog.extract_fhog_features(aligned_face_rgb, cell_size=8)

    if verbose:
        print(f"   pyfhog FHOG: {features_pyfhog.shape}")
        print(f"   Feature range: [{features_pyfhog.min():.4f}, {features_pyfhog.max():.4f}]")
        print()

    # 4. Compare outputs
    if verbose:
        print("="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print()

    # Check dimensions match
    if features_of.shape != features_pyfhog.shape:
        print(f"DIMENSION MISMATCH!")
        print(f"   OpenFace: {features_of.shape}")
        print(f"   pyfhog:   {features_pyfhog.shape}")
        return {
            'passed': False,
            'error': 'dimension_mismatch',
            'of_shape': features_of.shape,
            'pyfhog_shape': features_pyfhog.shape
        }

    # Compute metrics
    correlation = np.corrcoef(features_of, features_pyfhog)[0, 1]
    rmse = np.sqrt(np.mean((features_of - features_pyfhog)**2))
    mae = np.mean(np.abs(features_of - features_pyfhog))
    max_diff = np.max(np.abs(features_of - features_pyfhog))

    # Compute per-feature statistics
    diff = features_of - features_pyfhog
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    if verbose:
        print(f"Correlation:      r = {correlation:.8f}")
        print(f"RMSE:             {rmse:.8f}")
        print(f"MAE:              {mae:.8f}")
        print(f"Max difference:   {max_diff:.8f}")
        print()
        print(f"Mean difference:  {mean_diff:.8f}")
        print(f"Std difference:   {std_diff:.8f}")
        print()

    # Determine pass/fail
    passed = correlation > 0.999

    if verbose:
        if passed:
            print("VALIDATION PASSED!")
            print(f"   Correlation r = {correlation:.8f} > 0.999 (threshold)")
        else:
            print("VALIDATION FAILED")
            print(f"   Correlation r = {correlation:.8f} <= 0.999 (threshold)")
        print()

    return {
        'passed': passed,
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'of_shape': features_of.shape,
        'pyfhog_shape': features_pyfhog.shape
    }


def main():
    """Run validation tests"""

    # Paths to test data
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    video_file = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"

    # Test multiple frames
    test_frames = [1, 50, 100, 500, 1000]

    print("="*80)
    print("PyFHOG vs OpenFace 2.2 Validation")
    print("="*80)
    print()
    print(f"Test video: {os.path.basename(video_file)}")
    print(f"Test frames: {test_frames}")
    print()

    results = []
    for frame_num in test_frames:
        result = compare_pyfhog_vs_openface(
            hog_file,
            video_file,
            csv_file,
            frame_num=frame_num,
            verbose=False
        )

        if result is None:
            print(f"Frame {frame_num}: FAILED (extraction error)")
            continue

        if not result.get('passed', False) and 'error' in result:
            print(f"Frame {frame_num}: FAILED ({result['error']})")
            continue

        results.append(result)

        status = "PASS" if result['passed'] else "FAIL"
        print(f"Frame {frame_num:4d}: {status}  r={result['correlation']:.8f}  RMSE={result['rmse']:.6f}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if len(results) == 0:
        print("All tests failed")
        return 1

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    avg_correlation = np.mean([r['correlation'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])

    print(f"Passed: {passed_count}/{total_count}")
    print(f"Average correlation: r = {avg_correlation:.8f}")
    print(f"Average RMSE: {avg_rmse:.8f}")
    print()

    if passed_count == total_count:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
