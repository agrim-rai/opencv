// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

namespace {
// -------------------helper functions------------------- 

// Project a 3D world point through pose T_cw; returns camera-frame depth and pixel (u, v).
// Returns false if point is behind the camera.
bool projectPoint(const Matx44d& T_cw, const Mat& K,
                  double Xw, double Yw, double Zw,
                  double& u, double& v)
{
    const double Xc = T_cw(0,0)*Xw + T_cw(0,1)*Yw + T_cw(0,2)*Zw + T_cw(0,3);
    const double Yc = T_cw(1,0)*Xw + T_cw(1,1)*Yw + T_cw(1,2)*Zw + T_cw(1,3);
    const double Zc = T_cw(2,0)*Xw + T_cw(2,1)*Yw + T_cw(2,2)*Zw + T_cw(2,3);
    if (Zc <= 0.0) return false;
    u = K.at<double>(0,0) * Xc / Zc + K.at<double>(0,2);
    v = K.at<double>(1,1) * Yc / Zc + K.at<double>(1,2);
    return true;
}

// Descriptor distance: Hamming for CV_8U, L2 for float.
inline double descDist(const Mat& a, const Mat& b)
{
    if (a.empty() || b.empty()) return std::numeric_limits<double>::max();
    int norm_type = (a.type() == CV_8U) ? NORM_HAMMING : NORM_L2;
    return norm(a, b, norm_type);
}

// Collect non-bad MapPoints visible from the given KeyFrame + its top-K covis neighbours.
void buildLocalMapPoints(const KeyFrame* kf, int top_k,
                         std::set<MapPoint*>& local_mps)
{
    if (!kf) return;
    for (MapPoint* mp : kf->mappoints)
        if (mp && !mp->bad) local_mps.insert(mp);

    int k = 0;
    for (const auto& [nb_kf, cnt] : kf->ordered_covisibility)
    {
        if (k++ >= top_k) break;
        for (MapPoint* mp : nb_kf->mappoints)
            if (mp && !mp->bad) local_mps.insert(mp);
    }
}

// Run solvePnPRansac + solvePnPRefineLM on (obj, img) pairs; update frame.pose_cw.
// Returns number of inliers, or -1 on failure.
int runPnP(const std::vector<Point3f>& obj, const std::vector<Point2f>& img,
           Frame& frame,
           const Mat& K, double reproj_thresh, int max_iters, double confidence,
           int min_inliers)
{
    if ((int)obj.size() < min_inliers) return -1;

    Mat rvec, tvec, inlier_idx;
    bool ok = solvePnPRansac(obj, img, K, Mat() /*undistorted*/,
                              rvec, tvec, false,
                              max_iters, (float)reproj_thresh, confidence,
                              inlier_idx, SOLVEPNP_AP3P);
    if (!ok || inlier_idx.rows < min_inliers) return -1;

    // Refine on inliers.
    std::vector<Point3f> obj_in;
    std::vector<Point2f> img_in;
    obj_in.reserve(inlier_idx.rows); img_in.reserve(inlier_idx.rows);
    for (int i = 0; i < inlier_idx.rows; ++i)
    {
        int idx = inlier_idx.at<int>(i);
        obj_in.push_back(obj[idx]);
        img_in.push_back(img[idx]);
    }
    solvePnPRefineLM(obj_in, img_in, K, Mat(), rvec, tvec);

    Mat R;
    Rodrigues(rvec, R);
    frame.pose_cw = detail::makePose(R, tvec);
    return inlier_idx.rows;
}

} // anonymous namespace

// =============================================================================
// Stage B: motion-model projection search
// =============================================================================

bool VisualOdometryImpl::trackWithMotionModel(Frame& cur)
{
    if (!last_kf_) return false;

    // Predict pose from constant-velocity model.
    cur.pose_cw = velocity_ * last_pose_cw_;

    // Build local map: last KF + top-K covis neighbours.
    std::set<MapPoint*> local_mps;
    buildLocalMapPoints(last_kf_, params_.local_map_top_k, local_mps);

    // Reset any stale matches in the Frame.
    std::fill(cur.mappoints.begin(), cur.mappoints.end(), nullptr);
    std::fill(cur.outliers.begin(), cur.outliers.end(), false);

    auto doSearch = [&](float radius) -> int {
        int n = 0;
        for (MapPoint* mp : local_mps)
        {
            double u, v;
            if (!projectPoint(cur.pose_cw, K_, mp->pos.x, mp->pos.y, mp->pos.z, u, v))
                continue;
            if (u < 0 || u >= cur.imageSize.width ||
                v < 0 || v >= cur.imageSize.height) continue;

            mp->visible_count++;

            auto cands = cur.getKeypointsInRadius((float)u, (float)v, radius);
            double best_d = std::numeric_limits<double>::max();
            size_t best_i = std::numeric_limits<size_t>::max();
            for (size_t idx : cands)
            {
                if (cur.mappoints[idx]) continue;
                double d = descDist(mp->ref_desc, cur.descriptors.row((int)idx));
                if (d < best_d) { best_d = d; best_i = idx; }
            }
            if (best_i != std::numeric_limits<size_t>::max() &&
                best_d  < params_.desc_proj_thresh)
            {
                cur.mappoints[best_i] = mp;
                mp->found_count++;
                ++n;
            }
        }
        return n;
    };

    int n = doSearch((float)params_.motion_model_radius);
    if (n < params_.motion_model_min_matches)
    {
        std::fill(cur.mappoints.begin(), cur.mappoints.end(), nullptr);
        n = doSearch((float)params_.motion_model_radius_wide);
    }

    if (n < params_.pnp_min_inliers) return false;

    // Refine pose with PnP using the projection matches.
    std::vector<Point3f> obj; std::vector<Point2f> img;
    obj.reserve(n); img.reserve(n);
    for (size_t i = 0; i < cur.mappoints.size(); ++i)
    {
        if (!cur.mappoints[i]) continue;
        const MapPoint* mp = cur.mappoints[i];
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(cur.undist_kpts[i]);
    }

    int n_inliers = runPnP(obj, img, cur, K_,
                           params_.pnp_reproj_thresh,
                           params_.pnp_ransac_iters,
                           params_.pnp_confidence,
                           params_.pnp_min_inliers);
    if (n_inliers < 0) return false;

    int n_opt = Optimizer::PoseOptimization(cur, K_, params_.pnp_reproj_thresh, enable_pose_opt_);
    return n_opt >= params_.pnp_min_inliers;
}

// =============================================================================
// Stage C: descriptor match against last keyframe (reference-KF fallback)
// Descriptor-matches the current frame against last_kf_ directly. Builds 3D-2D correspondences using last_kf_->mappoints. 
// Runs solvePnPRansac (AP3P) + solvePnPRefineLM to solve for cur.pose_cw
// =============================================================================

bool VisualOdometryImpl::trackWithReferenceKF(Frame& cur)
{
    if (!last_kf_) return false;

    std::fill(cur.mappoints.begin(), cur.mappoints.end(), nullptr);
    std::fill(cur.outliers.begin(), cur.outliers.end(), false);

    std::vector<DMatch> matches;
    matchFrames(last_kf_->keypoints, last_kf_->descriptors, last_kf_->imageSize,
                cur.keypoints, cur.descriptors, cur.imageSize, matches);

    // Build 3D-2D correspondences and keep parallel MapPoint pointers.
    std::vector<Point3f>  obj;
    std::vector<Point2f>  img;
    std::vector<MapPoint*> corr_mps;   // mp for each (obj, img) pair
    std::vector<int>       corr_kp;    // cur keypoint index for each pair
    obj.reserve(matches.size()); img.reserve(matches.size());
    corr_mps.reserve(matches.size()); corr_kp.reserve(matches.size());

    for (const auto& m : matches)
    {
        if ((size_t)m.queryIdx >= last_kf_->mappoints.size()) continue;
        MapPoint* mp = last_kf_->mappoints[m.queryIdx];
        if (!mp || mp->bad) continue;
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(cur.undist_kpts[m.trainIdx]);
        corr_mps.push_back(mp);
        corr_kp.push_back(m.trainIdx);
    }

    if ((int)obj.size() < params_.pnp_min_inliers)
    {
        last_event_ = format("refKF: 2d3d=%d < %d", (int)obj.size(), params_.pnp_min_inliers);
        return false;
    }

    int n_inliers = runPnP(obj, img, cur, K_,
                           params_.pnp_reproj_thresh, params_.pnp_ransac_iters,
                           params_.pnp_confidence, params_.pnp_min_inliers);
    if (n_inliers < 0)
    {
        last_event_ = "refKF: PnP failed";
        return false;
    }

    // Populate cur.mappoints for all matched pairs (PoseOptimisation will filter).
    for (size_t k = 0; k < corr_mps.size(); ++k)
    {
        int kp_idx = corr_kp[k];
        if ((size_t)kp_idx < cur.mappoints.size() && !cur.mappoints[kp_idx])
            cur.mappoints[kp_idx] = corr_mps[k];
    }

    int n_opt = Optimizer::PoseOptimization(cur, K_, params_.pnp_reproj_thresh, enable_pose_opt_);
    return n_opt >= params_.pnp_min_inliers;
}

// =============================================================================
// Stage D: optical-flow fallback
// Lucas-Kanade calcOpticalFlowPyrLK tracks the previous frame's keypoints into the current image.
// 
// =============================================================================

bool VisualOdometryImpl::trackWithOpticalFlow(Frame& cur)
{
    if (!has_prev_frame_ || prev_frame_.image.empty()) return false;

    const Frame& prev = prev_frame_;
    if (prev.undist_kpts.empty()) return false;

    // Track previous undistorted keypoints into the current image.
    std::vector<Point2f> prev_pts(prev.undist_kpts.begin(), prev.undist_kpts.end());
    std::vector<Point2f> cur_pts;
    std::vector<uchar>   status;
    std::vector<float>   err;

    calcOpticalFlowPyrLK(prev.image, cur.image, prev_pts, cur_pts,
                         status, err, Size(21, 21), 3,
                         TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));

    // Build 3D-2D correspondences from flow pairs that have a MapPoint.
    std::vector<Point3f> obj; std::vector<Point2f> img;
    obj.reserve(prev_pts.size()); img.reserve(prev_pts.size());

    for (size_t i = 0; i < prev_pts.size(); ++i)
    {
        if (!status[i]) continue;
        if (i >= prev.mappoints.size()) continue;
        MapPoint* mp = prev.mappoints[i];
        if (!mp || mp->bad) continue;
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(cur_pts[i]);
    }

    if ((int)obj.size() < params_.optical_flow_min_inliers)
    {
        last_event_ = format("optflow: corr=%d < %d", (int)obj.size(), params_.optical_flow_min_inliers);
        return false;
    }

    int n_inliers = runPnP(obj, img, cur, K_,
                            params_.pnp_reproj_thresh, params_.pnp_ransac_iters,
                            params_.pnp_confidence, params_.optical_flow_min_inliers);
    if (n_inliers < 0)
    {
        last_event_ = "optflow: PnP failed";
        return false;
    }

    // cur.pose_cw is set; cur.mappoints will be filled by Stage E.
    return true;
}

// =============================================================================
// Stage E: local-map expansion + re-optimisation
// always runs
// =============================================================================

void VisualOdometryImpl::trackLocalMap(Frame& cur)
{
    if (!last_kf_) return;

    // Build local map: last KF + top-K neighbours + their top-K neighbours.
    std::set<MapPoint*> local_mps;
    buildLocalMapPoints(last_kf_, params_.local_map_top_k, local_mps);

    int nb_k = 0;
    for (const auto& [nb_kf, cnt] : last_kf_->ordered_covisibility)
    {
        if (nb_k++ >= params_.local_map_top_k) break;
        int nb2_k = 0;
        for (const auto& [nb2_kf, cnt2] : nb_kf->ordered_covisibility)
        {
            if (nb2_k++ >= params_.local_map_neighbor_k) break;
            for (MapPoint* mp : nb2_kf->mappoints)
                if (mp && !mp->bad) local_mps.insert(mp);
        }
    }

    // Fast set of already-matched MPs.
    std::set<MapPoint*> already_matched;
    for (MapPoint* mp : cur.mappoints)
        if (mp) already_matched.insert(mp);

    bool any_new = false;
    const float r = (float)params_.local_map_radius;

    for (MapPoint* mp : local_mps)
    {
        if (already_matched.count(mp)) continue;

        double u, v;
        if (!projectPoint(cur.pose_cw, K_, mp->pos.x, mp->pos.y, mp->pos.z, u, v))
            continue;
        if (u < 0 || u >= cur.imageSize.width ||
            v < 0 || v >= cur.imageSize.height) continue;

        mp->visible_count++;

        auto cands = cur.getKeypointsInRadius((float)u, (float)v, r);
        double best_d = std::numeric_limits<double>::max();
        size_t best_i = std::numeric_limits<size_t>::max();
        for (size_t idx : cands)
        {
            if (cur.mappoints[idx]) continue;
            double d = descDist(mp->ref_desc, cur.descriptors.row((int)idx));
            if (d < best_d) { best_d = d; best_i = idx; }
        }
        if (best_i != std::numeric_limits<size_t>::max() &&
            best_d  < params_.desc_proj_thresh)
        {
            cur.mappoints[best_i] = mp;
            cur.outliers[best_i]  = false;
            mp->found_count++;
            already_matched.insert(mp);
            any_new = true;
        }
    }

    // Re-run pose optimisation with the expanded correspondence set.
    if (any_new)
        Optimizer::PoseOptimization(cur, K_, params_.pnp_reproj_thresh, enable_pose_opt_);
}

// =============================================================================
// Top-level track()
// Stage B/C/D + E -> tracking
// =============================================================================

bool VisualOdometryImpl::track(Frame& cur)
{
    if (!last_kf_)
    {
        ref_frame_ = cur;
        state_     = INITIALIZING;
        return false;
    }

    // --- Stage B: motion model (requires velocity from a previous track) ---
    bool ok = false;
    if (has_velocity_)
        ok = trackWithMotionModel(cur);

    // --- Stage C: reference-keyframe descriptor match (always reliable) ---
    if (!ok)
        ok = trackWithReferenceKF(cur);

    // --- Stage D: optical-flow fallback ---
    if (!ok)
        ok = trackWithOpticalFlow(cur);

    if (!ok)
    {
        last_event_ = "track lost: all stages failed";
        ref_frame_  = cur;
        state_      = INITIALIZING;
        return false;
    }

    // --- Stage E: local-map expansion + re-optimise ---
    trackLocalMap(cur);

    // Count inliers after Stage E.
    int n_inliers = 0;
    for (size_t i = 0; i < cur.mappoints.size(); ++i)
        if (cur.mappoints[i] && !cur.outliers[i]) ++n_inliers;

    // Update velocity: Tcw_cur * Tcw_last.inv()
    // after first successful track() 
    {
        Matx44d Tcw_last_inv = last_pose_cw_.inv();
        velocity_     = cur.pose_cw * Tcw_last_inv;
        has_velocity_ = true;
    }

    // Commit current pose.
    last_pose_cw_ = cur.pose_cw;
    map_.appendPose(cur.pose_cw);
    ++frames_since_kf_;

    // Store current frame for Stage D next time.
    prev_frame_     = cur;
    has_prev_frame_ = true;

    // --- Keyframe promotion + Local BA --------------------------------------
    String kf_reason;
    if (shouldPromoteKeyframe(n_inliers, cur.pose_cw, kf_reason))
    {
        int mp_before = map_.numMapPoints();
        promoteKeyframeAndGrowMap(cur);
        // Run local BA over the new KF + its covisible window (no-op if the
        // backend is disabled via setLocalBA(false) or g2o is absent).
        // last_kf_ now points to the newly promoted KF.
        Optimizer::LocalBundleAdjustment(last_kf_, K_, enable_local_ba_, nullptr);
        // Sync velocity model with the (possibly BA-refined) pose so Stage B's
        // next prediction is consistent with the updated map point positions.
        last_pose_cw_ = last_kf_->pose_cw;
        last_event_ = format("keyframe: %s, +%d mp%s",
                             kf_reason.c_str(),
                             map_.numMapPoints() - mp_before,
                             enable_local_ba_ ? " (LocalBA)" : "");
    }

    return true;
}

}} // namespace cv::slam
