// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_HPP
#define OPENCV_SLAM_HPP

/**
@defgroup slam SLAM and Visual Odometry

This module implements visual SLAM building blocks. The current entry point is
@ref cv::slam::VisualOdometry, a monocular visual odometry pipeline that
bootstraps an initial map from two-view geometry and then tracks subsequent
frames with PnP, growing the map at keyframe promotions.

Subsequent additions (bundle adjustment, loop closure, relocalization) will
operate on the shared @ref cv::slam::Map container produced by this pipeline.
*/

#include "opencv2/slam/types.hpp"
#include "opencv2/slam/map.hpp"
#include "opencv2/slam/odometry_params.hpp"
#include "opencv2/slam/visual_odometry.hpp"

#endif // OPENCV_SLAM_HPP
