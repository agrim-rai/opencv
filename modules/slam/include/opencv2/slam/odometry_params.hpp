// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_ODOMETRY_PARAMS_HPP
#define OPENCV_SLAM_ODOMETRY_PARAMS_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Tunable parameters for @ref VisualOdometry.

Defaults are sensible starting values. Adjust per-dataset via
@ref VisualOdometry::setParams before calling @ref VisualOdometry::run, or
pass a configured instance to @ref VisualOdometry::create.
*/
struct CV_EXPORTS OdometryParams
{
    OdometryParams();

    // --- Bootstrap (H/F two-view initialization) ---------------------------

    /** Minimum feature matches required before attempting bootstrap. */
    int    min_init_inliers      = 40;

    /** Minimum parallax (degrees) for a triangulated point to be accepted. */
    double min_init_parallax_deg = 3.0;

    /** Minimum triangulated points required to accept the bootstrap result.
        Ensures the initial map is large enough to support tracking. */
    int    min_init_points       = 100;

    /** Homography / Fundamental model-selection ratio.
        RH = n_H / (n_H + n_E).  If RH > threshold → decompose H (planar scene),
        else decompose E/F (general scene). ORB-SLAM3 default = 0.45. */
    double hf_ratio_thresh       = 0.45;

    /** Minimum parallax (degrees) for a new point added during incremental
        map growth between keyframes. */
    double min_growth_parallax_deg = 0.1;

    /** RANSAC reprojection threshold (pixels) for findEssentialMat / findHomography. */
    double essential_ransac_thresh = 1.0;

    /** RANSAC confidence for findEssentialMat / findHomography. */
    double essential_ransac_confidence = 0.999;

    // --- Tracking (PnP) ----------------------------------------------------

    /** Reprojection threshold (pixels) for PnP RANSAC and map-growth filtering. */
    double pnp_reproj_thresh = 4.0;

    /** Minimum inliers required for any tracking stage to succeed. */
    int    pnp_min_inliers   = 6;

    /** Maximum RANSAC iterations for solvePnPRansac. */
    int    pnp_ransac_iters  = 500;

    /** RANSAC confidence for solvePnPRansac. */
    double pnp_confidence    = 0.99;

    // --- Motion-model tracking (Stage B) -----------------------------------

    /** Projection search radius (pixels) for normal motion-model matching. */
    double motion_model_radius      = 15.0;

    /** Wider search radius used when normal radius yields too few matches. */
    double motion_model_radius_wide = 30.0;

    /** Minimum matches after projection search to accept the motion model;
        below this the wider radius is tried, then Stage B is abandoned. */
    int    motion_model_min_matches = 20;

    /** Descriptor distance threshold for projection-based matching.
        Set to ~1.0 for L2-normalised float descriptors (ALIKED),
        or ~50 for Hamming binary descriptors (ORB). */
    double desc_proj_thresh = 1.0;

    // --- Optical-flow fallback (Stage D) -----------------------------------

    /** Minimum solvePnP inliers for the optical-flow fallback to succeed. */
    int    optical_flow_min_inliers = 10;

    // --- Keyframe promotion ------------------------------------------------

    /** Minimum frames between consecutive keyframe promotions (anti-spam). */
    int    kf_min_frames     = 1;

    /** Absolute timeout: promote a keyframe if frames since the last one
        exceeds this value regardless of other conditions. */
    int    kf_max_frames     = 30;

    /** Inlier-ratio condition: promote if
        n_inliers_after_local_map < kf_inlier_ratio × last_kf_inliers. */
    double kf_inlier_ratio   = 0.75;

    /** Absolute minimum inlier count below which a keyframe is always
        promoted (safety net when last_kf_inliers_ is 0 or unreliable). */
    int    kf_min_inliers    = 40;

    /** Promote a keyframe when rotation from the last keyframe exceeds
        this angle (degrees). */
    double kf_rot_thresh_deg = 5.0;

    // --- Local-map tracking (Stage E) --------------------------------------

    /** Top-K covisibility neighbours of the current keyframe whose
        MapPoints form the local map for Stage E. */
    int    local_map_top_k      = 10;

    /** For each top-K neighbour, also pull in their own top-K neighbours. */
    int    local_map_neighbor_k = 5;

    /** Projection search radius (pixels) used in Stage E local-map search.
        Tighter than motion-model search to reduce false positives. */
    double local_map_radius     = 7.0;

    // Note: the g2o refinement stages (pose-only optimization, local bundle
    // adjustment) are toggled on the VisualOdometry instance itself —
    // VisualOdometry::setPoseOptimization / setLocalBA — not via this struct.
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_ODOMETRY_PARAMS_HPP
