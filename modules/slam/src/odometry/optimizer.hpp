// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_OPTIMIZER_HPP
#define OPENCV_SLAM_OPTIMIZER_HPP

#include "frame.hpp"

namespace cv {
namespace slam {
namespace Optimizer {

/** @brief Pose-only bundle adjustment for @p frame.

When the module is built with g2o (HAVE_G2O defined) this runs a real 6-DoF
pose optimisation:
  - Variable: frame.pose_cw (SE3, 6-DoF)
  - Fixed:    all MapPoint 3D positions
  - Cost:     Huber( || π(T·Xᵢ) − uᵢ ||² ),  δ = √5.991 px
  - Solver:   g2o Levenberg-Marquardt, two 4-iteration passes
  - Pass 1 with Huber robust kernel on all edges; pass 2 pure-L2 on inliers.
  - frame.pose_cw is updated in-place with the refined pose.

When @p enable is false (or the module is built without g2o) the call
degrades to a reprojection-only inlier check: each association is classified
by reprojection distance and outliers are marked, but frame.pose_cw is NOT
changed.

@param frame         Current frame. pose_cw must be set, mappoints[] populated.
@param K             3×3 camera intrinsic matrix (CV_64F).
@param reproj_thresh Reprojection threshold (px) for the fallback inlier check.
@param enable        Whether to run g2o pose-only optimization.
@returns             Number of inlier correspondences after optimisation.
*/
int PoseOptimization(Frame& frame, const Mat& K, double reproj_thresh,
                     bool enable);

/** @brief Local bundle adjustment over a sliding window of keyframes.

Optimises the poses of @p new_kf and its top-10 covisible keyframes together
with all 3D map points they observe.  Keyframes that observe local map points
but lie outside the window are added as fixed anchors (gauge constraint for
monocular scale).

When @p enable is false (or the module is built without g2o) this is a no-op.

@param new_kf    The keyframe that was just promoted (window centre).
@param K         3×3 camera intrinsic matrix (CV_64F).
@param enable    Whether to run local bundle adjustment.
@param stop_flag Optional pointer; set to true externally to abort mid-run.
*/
void LocalBundleAdjustment(KeyFrame* new_kf, const Mat& K, bool enable,
                            bool* stop_flag = nullptr);

} // namespace Optimizer
}} // namespace cv::slam

#endif // OPENCV_SLAM_OPTIMIZER_HPP
