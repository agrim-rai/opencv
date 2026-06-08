// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

namespace {

// Try all H decomposition candidates; return index of the one with the most
// cheirality-consistent triangulations.
// Helper function
int bestHomographyCandidate(
    const std::vector<Mat>& Rs,
    const std::vector<Mat>& ts,
    const std::vector<Point2f>& pts1,
    const std::vector<Point2f>& pts2,
    const Mat& P1)
{
    int best_idx   = 0;
    int best_count = -1;
    const int n_sol = (int)Rs.size();

    for (int s = 0; s < n_sol; ++s)
    {
        Mat Rt(3, 4, CV_64F);
        Rs[s].copyTo(Rt(Rect(0, 0, 3, 3)));
        ts[s].reshape(1, 3).copyTo(Rt(Rect(3, 0, 1, 3)));
        Mat P2s = P1 * Mat::eye(4, 4, CV_64F); // will overwrite below
        // Build K*[R|t] for this candidate
        // We do not have K here; use K embedded in P1 by passing it directly.
        // P1 = K*[I|0], so we just build [R|t] rows for the candidate.
        // Actually we need K. Pass P1 = K*I_3x4 and compute P2 = K * Rt.
        // Simplification: just use triangulatePoints with P1 and K*Rt.
        // Since we don't have K here, this helper is called with P1 already = K*I,
        // so P2 = P1 * [R|t;0 0 0 1] approach doesn't work directly.
        // Instead callers should provide P2 arrays. Rebuild it here using P1's K.
        // P1 = K * [I|0] => K = P1(0:3, 0:3).
        Mat K_local = P1(Rect(0, 0, 3, 3)).clone();

        Mat P2_s(3, 4, CV_64F);
        Mat tmp = K_local * Rt;
        tmp.copyTo(P2_s);

        Mat pts4D;
        triangulatePoints(P1, P2_s, pts1, pts2, pts4D);

        int count = 0;
        for (int i = 0; i < pts4D.cols; ++i)
        {
            float w = pts4D.at<float>(3, i);
            if (std::abs(w) < 1e-9f) continue;
            float Z1 = pts4D.at<float>(2, i) / w;
            float X  = pts4D.at<float>(0, i) / w;
            float Y  = pts4D.at<float>(1, i) / w;
            float Z  = pts4D.at<float>(2, i) / w;
            float Z2 = (float)(Rs[s].at<double>(2,0)*X + Rs[s].at<double>(2,1)*Y
                              + Rs[s].at<double>(2,2)*Z
                              + ts[s].reshape(1,3).at<double>(2,0));
            if (Z1 > 0 && Z2 > 0) ++count;
        }
        if (count > best_count) { best_count = count; best_idx = s; }
    }
    return best_idx;
}

} // anonymous namespace

bool VisualOdometryImpl::bootstrap(Frame& cur)
{
    // Match reference frame <-> current frame.
    std::vector<DMatch> matches;
    matchFrames(ref_frame_.keypoints, ref_frame_.descriptors, ref_frame_.imageSize,
                cur.keypoints, cur.descriptors, cur.imageSize, matches);

    if ((int)matches.size() < params_.min_init_inliers)
    {
        // Views are too dissimilar — slide reference forward so we don't keep
        // trying to match against a frame that is now too far away.
        ref_frame_ = std::move(cur);
        return false;
    }

    // Collect undistorted matched point pairs.
    std::vector<Point2f> ref_u, cur_u;
    ref_u.reserve(matches.size());
    cur_u.reserve(matches.size());
    for (const auto& m : matches)
    {
        ref_u.push_back(ref_frame_.undist_kpts[m.queryIdx]);
        cur_u.push_back(cur.undist_kpts[m.trainIdx]);
    }

    // -------------------------------------------------------------------------
    // Compute E (via findEssentialMat) and H (via findHomography) in parallel.
    // Both operate on undistorted pixel coordinates.
    // Computes both Essential matrix (E) and Homography (H) simultaneously via RANSAC. 
    // Picks which to use based on the ratio RH = n_H / (n_H + n_E). If RH > hf_ratio_thresh, 
    // the scene is planar → use H. Otherwise use E.
    // -------------------------------------------------------------------------
    Mat mask_E, mask_H;
    Mat E = findEssentialMat(ref_u, cur_u, K_, RANSAC,
                              params_.essential_ransac_confidence,
                              params_.essential_ransac_thresh, 1000, mask_E);
    Mat H = findHomography(ref_u, cur_u, RANSAC,
                           params_.essential_ransac_thresh, mask_H);

    const int n_E = (!E.empty() && !mask_E.empty()) ? countNonZero(mask_E) : 0;
    const int n_H = (!H.empty() && !mask_H.empty()) ? countNonZero(mask_H) : 0;

    if (n_E < params_.min_init_inliers && n_H < params_.min_init_inliers)
        return false;

    // RH = n_H / (n_H + n_E);  if RH > threshold use H (planar scene), else E/F.
    const double RH = (double)n_H / ((double)n_H + (double)n_E + 1e-9);

    // -------------------------------------------------------------------------
    // Decompose selected model into (R, t) and pick the best candidate.
    // -------------------------------------------------------------------------
    Mat R, t;
    Mat model_mask;

    Mat I0 = Mat::eye(3, 4, CV_64F);
    Mat P1 = K_ * I0;

    if (RH > params_.hf_ratio_thresh && !H.empty())
    {
        // --- Homography path: up to 8 candidates ---
        std::vector<Mat> Rs, ts, normals;
        int n_sol = decomposeHomographyMat(H, K_, Rs, ts, normals);
        if (n_sol <= 0) return false;

        // Collect H inlier pairs for the candidate evaluation
        std::vector<Point2f> p1_in, p2_in;
        for (size_t i = 0; i < matches.size(); ++i)
            if (!mask_H.empty() && mask_H.at<uchar>((int)i))
            { p1_in.push_back(ref_u[i]); p2_in.push_back(cur_u[i]); }
        if (p1_in.empty()) return false;

        int best = bestHomographyCandidate(Rs, ts, p1_in, p2_in, P1);
        R = Rs[best].clone();
        t = ts[best].clone();
        model_mask = mask_H;
    }
    else
    {
        // --- Essential matrix path: recoverPose picks best of 4 ---
        if (E.empty() || n_E < params_.min_init_inliers) return false;
        Mat recover_mask = mask_E.clone();
        int n_pose = recoverPose(E, ref_u, cur_u, K_, R, t, recover_mask);
        if (n_pose < params_.min_init_inliers) return false;
        model_mask = recover_mask;
    }

    // -------------------------------------------------------------------------
    // Collect inlier pairs and triangulate.
    // -------------------------------------------------------------------------
    std::vector<Point2f> ref_in, cur_in;
    std::vector<int>     match_in;
    for (size_t i = 0; i < matches.size(); ++i)
        if (!model_mask.empty() && model_mask.at<uchar>((int)i))
        {
            ref_in.push_back(ref_u[i]);
            cur_in.push_back(cur_u[i]);
            match_in.push_back((int)i);
        }

    if ((int)ref_in.size() < params_.min_init_inliers)
        return false;

    Mat Rt(3, 4, CV_64F);
    R.copyTo(Rt(Rect(0, 0, 3, 3)));
    t.reshape(1, 3).copyTo(Rt(Rect(3, 0, 1, 3)));
    Mat P2 = K_ * Rt;

    Mat pts4D;
    triangulatePoints(P1, P2, ref_in, cur_in, pts4D); // 4×N, CV_32F

    Matx44d T_ref = Matx44d::eye();
    Matx44d T_cur = detail::makePose(R, t);

    // -------------------------------------------------------------------------
    // Filter: cheirality + parallax; collect quality statistics.
    // -------------------------------------------------------------------------
    std::vector<Point3d> good_pts;
    std::vector<int>     good_match;
    int n_valid = 0; // finite homogeneous points
    int n_pos   = 0; // positive depth in both cameras

    for (int i = 0; i < pts4D.cols; ++i)
    {
        double w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        ++n_valid;

        double X = pts4D.at<float>(0, i) / w;
        double Y = pts4D.at<float>(1, i) / w;
        double Z = pts4D.at<float>(2, i) / w;

        if (Z <= 0) continue;
        double Z2 = R.at<double>(2,0)*X + R.at<double>(2,1)*Y
                  + R.at<double>(2,2)*Z + t.reshape(1,3).at<double>(2,0);
        if (Z2 <= 0) continue;
        ++n_pos;

        Point3d Xw(X, Y, Z);
        if (detail::parallaxDeg(Xw, T_ref, T_cur) < params_.min_init_parallax_deg)
            continue;

        good_pts.push_back(Xw);
        good_match.push_back(match_in[i]);
    }

    // Require positive-depth ratio >= 0.9 (rejects degenerate decompositions).
    if (n_valid > 0 && (double)n_pos / n_valid < 0.9)
        return false;
    if ((int)good_pts.size() < params_.min_init_points)
        return false;

    // -------------------------------------------------------------------------
    // Scale normalisation: set median scene depth in camera 1 = 1.0.
    // -------------------------------------------------------------------------
    std::vector<double> depths;
    depths.reserve(good_pts.size());
    for (const auto& p : good_pts) depths.push_back(p.z);
    std::nth_element(depths.begin(), depths.begin() + depths.size()/2, depths.end());
    double med = depths[depths.size()/2];
    if (med < 1e-9) return false;

    double scale = 1.0 / med;
    for (auto& p : good_pts) { p.x *= scale; p.y *= scale; p.z *= scale; }

    Mat t_sc;
    t.reshape(1, 3).convertTo(t_sc, CV_64F);
    t_sc = t_sc * scale;
    T_cur = detail::makePose(R, t_sc);

    // -------------------------------------------------------------------------
    // Create KF0 and KF1.
    // -------------------------------------------------------------------------
    auto makeKF = [](const Frame& f) -> KeyFrame* {
        KeyFrame* kf  = new KeyFrame();
        kf->pose_cw   = Matx44d::eye();
        kf->keypoints   = f.keypoints;
        kf->descriptors = f.descriptors.clone();
        kf->undist_kpts = f.undist_kpts;
        kf->imageSize   = f.imageSize;
        kf->mappoints.assign(f.keypoints.size(), nullptr);
        return kf;
    };

    KeyFrame* kf_ref = makeKF(ref_frame_);
    kf_ref->pose_cw  = T_ref;

    KeyFrame* kf_cur = makeKF(cur);
    kf_cur->pose_cw  = T_cur;
    kf_cur->parent   = kf_ref; // spanning-tree root = kf_ref

    map_.addKeyframe(kf_ref);
    map_.addKeyframe(kf_cur);

    // -------------------------------------------------------------------------
    // Create MapPoints, wire observations.
    // -------------------------------------------------------------------------
    for (size_t i = 0; i < good_pts.size(); ++i)
    {
        MapPoint* mp  = new MapPoint();
        mp->pos       = good_pts[i];
        const DMatch& m = matches[good_match[i]];
        mp->ref_desc  = ref_frame_.descriptors.row(m.queryIdx).clone();

        map_.addMapPoint(mp);
        map_.addObservation(kf_ref, (size_t)m.queryIdx, mp);
        map_.addObservation(kf_cur, (size_t)m.trainIdx, mp);
    }

    // Build covisibility graph for both bootstrap KFs.
    detail::updateCovisibility(kf_ref);
    // updateCovisibility(kf_ref) already updates kf_cur's covisibility too.

    map_.setRefKeyframe(kf_ref);
    map_.setCurrentKeyframe(kf_cur);

    // -------------------------------------------------------------------------
    // Advance pipeline state.
    // -------------------------------------------------------------------------
    last_kf_         = kf_cur;
    frames_since_kf_ = 0;
    last_kf_inliers_ = (int)good_pts.size();
    last_pose_cw_    = T_cur;
    state_           = TRACKING;

    // Velocity not reliable yet (bootstrap spans potentially many frames).
    has_velocity_    = false;

    map_.appendPose(T_ref);
    map_.appendPose(T_cur);

    ref_frame_ = Frame(); // release reference frame
    return true;
}

}} // namespace cv::slam
