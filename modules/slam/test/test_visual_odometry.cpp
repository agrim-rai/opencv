// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

// Test cases will live in this TU. Suggested initial coverage:
//   - Map: add/get keyframe + map point round-trip, addObservation wires both
//     directions, removeMapPoint cleans up keyframe links.
//   - VisualOdometry::create rejects empty intrinsics / null detector / null
//     matcher with the expected cv::Error code.
//   - End-to-end bootstrap + tracking on a synthetic point cloud rendered
//     through two virtual cameras (matcher mocked to return ground-truth
//     correspondences). Verifies state transitions, that median-depth
//     normalization sets median Z to 1, and that recovered poses are close
//     to ground truth up to scale.
