// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_VO_IMPL_HPP
#define OPENCV_SLAM_VO_IMPL_HPP

#include "../precomp.hpp"
#include "frame.hpp"
#include "optimizer.hpp"

#include <fstream>

namespace cv {
namespace slam {

/** @brief Concrete VisualOdometry implementation (pimpl target).

Holds all mutable pipeline state.  Stage logic is split across separate
translation units:
  - vo_bootstrap.cpp  : two-view H/F initialisation
  - vo_tracking.cpp   : multi-stage per-frame localisation (B/C/D/E)
  - vo_keyframe.cpp   : keyframe promotion decision + covisibility helpers
  - vo_map_growth.cpp : triangulation of new map points at promotion time
  - visual_odometry.cpp : factory, run(), processFrame(), IO writers
*/
class VisualOdometryImpl CV_FINAL : public VisualOdometry
{
public:
    VisualOdometryImpl(const Ptr<Feature2D>& detector,
                       const Ptr<DescriptorMatcher>& matcher,
                       const String& imagesFolder,
                       const String& outputFolder,
                       const Mat& cameraMatrix,
                       const Mat& distCoeffs,
                       const OdometryParams& params);

    // --- VisualOdometry interface -------------------------------------------

    bool run() CV_OVERRIDE;
    bool processFrame(InputArray image) CV_OVERRIDE;
    void reset() CV_OVERRIDE;

    OdometryState               getState()      const CV_OVERRIDE { return state_; }
    Matx44d                     getLastPose()   const CV_OVERRIDE { return last_pose_cw_; }
    const Map&                  getMap()        const CV_OVERRIDE { return map_; }
    const std::vector<Matx44d>& getTrajectory() const CV_OVERRIDE { return map_.trajectory(); }
    const OdometryParams&       getParams()     const CV_OVERRIDE { return params_; }
    void setParams(const OdometryParams& p)           CV_OVERRIDE { params_ = p; }

    void setPoseOptimization(bool e)                  CV_OVERRIDE { enable_pose_opt_ = e; }
    bool getPoseOptimization() const                  CV_OVERRIDE { return enable_pose_opt_; }
    void setLocalBA(bool e)                           CV_OVERRIDE { enable_local_ba_ = e; }
    bool getLocalBA() const                           CV_OVERRIDE { return enable_local_ba_; }

    const String& getImagesFolder() const CV_OVERRIDE { return images_folder_; }
    const String& getOutputFolder() const CV_OVERRIDE { return output_folder_; }
    void setOutputFolder(const String& f)  CV_OVERRIDE { output_folder_ = f; }

    // --- Stage entry points -------------------------------------------------

    /** H/F two-view bootstrap. */
    bool bootstrap(Frame& cur);

    /** Multi-stage per-frame tracking (calls sub-stages below). */
    bool track(Frame& cur);

    // Stage B: motion-model projection search
    bool trackWithMotionModel(Frame& cur);
    // Stage C: descriptor match against last keyframe
    bool trackWithReferenceKF(Frame& cur);
    // Stage D: optical-flow fallback
    bool trackWithOpticalFlow(Frame& cur);
    // Stage E: local-map expansion + re-optimise (always after B/C/D succeeds)
    void trackLocalMap(Frame& cur);

    /** Keyframe promotion criterion (OR of several conditions). */
    bool shouldPromoteKeyframe(int n_inliers, const Matx44d& T_cw,
                               String& reason) const;

    /** Create a new KeyFrame, triangulate new points, update covisibility. */
    void promoteKeyframeAndGrowMap(Frame& cur);

    // --- Shared helpers (visual_odometry.cpp) --------------------------------

    void extractFeatures(InputArray image, Frame& out) const;

    /** Match (q_kp, q_desc, q_sz) against (t_kp, t_desc, t_sz).
        Handles LightGlueMatcher spatial context automatically. */
    void matchFrames(const std::vector<KeyPoint>& q_kp, const Mat& q_desc, Size q_sz,
                     const std::vector<KeyPoint>& t_kp, const Mat& t_desc, Size t_sz,
                     std::vector<DMatch>& matches) const;

    // --- IO helpers (visual_odometry.cpp) ------------------------------------

    void writeTrajectoryText(const String& path) const;
    void writeTrajectoryBin (const String& path) const;
    void writeMapPoints     (const String& path) const;
    void writeKeypoints     (const String& path) const;
    void writeImagesTxt     (const String& path) const;

    // --- Owned state ---------------------------------------------------------

    Ptr<Feature2D>         detector_;
    Ptr<DescriptorMatcher> matcher_;
    Mat                    K_;        // 3×3 CV_64F camera intrinsics
    Mat                    dist_;     // distortion coefficients (may be empty)
    OdometryParams         params_;

    // g2o refinement toggles (config, not pipeline state — preserved across
    // reset()). Set via setPoseOptimization() / setLocalBA().
    bool enable_pose_opt_ = true;
    bool enable_local_ba_ = true;

    String images_folder_;
    String output_folder_;

    OdometryState state_        = NOT_INITIALIZED;
    Matx44d       last_pose_cw_ = Matx44d::eye();  // most recent successfully tracked pose

    Frame     ref_frame_;            // reference frame held during INITIALIZING
    KeyFrame* last_kf_ = nullptr;    // last promoted keyframe (not owned by impl)
    int       frames_since_kf_  = 0;
    int       last_kf_inliers_  = 0; // inlier count from the frame that was promoted

    // Constant-velocity motion model:  Tcw_pred = velocity_ * last_pose_cw_
    Matx44d velocity_     = Matx44d::eye();
    bool    has_velocity_ = false;

    // Previous frame image + mappoints kept for Stage D (optical-flow).
    Frame   prev_frame_;
    bool    has_prev_frame_ = false;

    // Per-frame log reason for keyframe promotion / tracking loss.
    String last_event_;

    // Parallel to map_.trajectory(); populated only during run().
    std::vector<String> pose_filenames_;

    Map map_;
};

// ---------------------------------------------------------------------------
// Shared geometry helpers (vo_keyframe.cpp)
// ---------------------------------------------------------------------------

namespace detail {

double  rotationAngleDeg(const Matx44d& A_cw, const Matx44d& B_cw);
double  parallaxDeg(const Point3d& X_world,
                    const Matx44d& A_cw, const Matx44d& B_cw);
Matx34d projectionFromPose(const Matx44d& T_cw);
Matx44d makePose(const Mat& R, const Mat& t);
Point3d cameraCenterWorld(const Matx44d& T_cw);

/** Rebuild covisibility map and ordered list for @p kf and all its observers. */
void updateCovisibility(KeyFrame* kf);

} // namespace detail

}} // namespace cv::slam

#endif // OPENCV_SLAM_VO_IMPL_HPP
