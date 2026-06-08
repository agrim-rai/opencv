// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

namespace {

// Project a world point through 3×4 projection matrix P; returns pixel.
inline Point2d projectThrough(const Mat& P, double X, double Y, double Z)
{
    double u = P.at<double>(0,0)*X + P.at<double>(0,1)*Y + P.at<double>(0,2)*Z + P.at<double>(0,3);
    double v = P.at<double>(1,0)*X + P.at<double>(1,1)*Y + P.at<double>(1,2)*Z + P.at<double>(1,3);
    double w = P.at<double>(2,0)*X + P.at<double>(2,1)*Y + P.at<double>(2,2)*Z + P.at<double>(2,3);
    if (std::abs(w) < 1e-12) return Point2d(0, 0);
    return Point2d(u / w, v / w);
}

// Camera-frame depth of a world point under a 4×4 pose.
inline double cameraDepth(const Matx44d& T_cw, double X, double Y, double Z)
{
    return T_cw(2,0)*X + T_cw(2,1)*Y + T_cw(2,2)*Z + T_cw(2,3);
}

} // anonymous namespace

void VisualOdometryImpl::promoteKeyframeAndGrowMap(Frame& cur)
{
    // -------------------------------------------------------------------------
    // 1. Create the new KeyFrame from the current Frame.
    // -------------------------------------------------------------------------
    KeyFrame* new_kf     = new KeyFrame();
    new_kf->pose_cw      = cur.pose_cw;
    new_kf->keypoints    = cur.keypoints;
    new_kf->descriptors  = cur.descriptors.clone();
    new_kf->undist_kpts  = cur.undist_kpts;
    new_kf->imageSize    = cur.imageSize;
    new_kf->mappoints.assign(cur.keypoints.size(), nullptr);
    new_kf->parent       = last_kf_; // spanning-tree link

    map_.addKeyframe(new_kf);

    // -------------------------------------------------------------------------
    // 2. Inherit MapPoint links from the Frame's tracking result (non-outliers).
    // -------------------------------------------------------------------------
    for (size_t i = 0; i < cur.mappoints.size(); ++i)
    {
        MapPoint* mp = cur.mappoints[i];
        if (!mp || mp->bad || cur.outliers[i]) continue;
        map_.addObservation(new_kf, i, mp);
    }

    // -------------------------------------------------------------------------
    // 3. Match last_kf_ against cur to find NEW points to triangulate.
    // -------------------------------------------------------------------------
    std::vector<DMatch> kf_to_cur;
    matchFrames(last_kf_->keypoints, last_kf_->descriptors, last_kf_->imageSize,
                cur.keypoints, cur.descriptors, cur.imageSize, kf_to_cur);

    // Build projection matrices.
    Mat Rt1(3, 4, CV_64F), Rt2(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
        {
            Rt1.at<double>(i,j) = last_kf_->pose_cw(i,j);
            Rt2.at<double>(i,j) = cur.pose_cw(i,j);
        }
    Mat P1 = K_ * Rt1;
    Mat P2 = K_ * Rt2;

    // Collect unlinked match pairs for triangulation.
    std::vector<Point2f> pts1, pts2;
    std::vector<int>     tri_match_idx;
    for (size_t i = 0; i < kf_to_cur.size(); ++i)
    {
        const DMatch& m = kf_to_cur[i];
        if ((size_t)m.queryIdx >= last_kf_->mappoints.size()) continue;
        if (last_kf_->mappoints[m.queryIdx] != nullptr) continue; // already has MP
        if ((size_t)m.trainIdx >= new_kf->mappoints.size()) continue;
        if (new_kf->mappoints[m.trainIdx] != nullptr) continue;   // already linked
        pts1.push_back(last_kf_->undist_kpts[m.queryIdx]);
        pts2.push_back(cur.undist_kpts[m.trainIdx]);
        tri_match_idx.push_back((int)i);
    }

    if (pts1.empty())
    {
        // Nothing new to triangulate; commit and update covisibility.
        detail::updateCovisibility(new_kf);
        map_.setCurrentKeyframe(new_kf);
        last_kf_         = new_kf;
        last_kf_inliers_ = 0;
        frames_since_kf_ = 0;
        return;
    }

    // -------------------------------------------------------------------------
    // 4. Triangulate and filter survivors.
    // -------------------------------------------------------------------------
    Mat pts4D;
    triangulatePoints(P1, P2, pts1, pts2, pts4D); // 4×N, CV_32F

    int n_new = 0;
    for (int i = 0; i < pts4D.cols; ++i)
    {
        double w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        double X = pts4D.at<float>(0, i) / w;
        double Y = pts4D.at<float>(1, i) / w;
        double Z = pts4D.at<float>(2, i) / w;

        if (cameraDepth(last_kf_->pose_cw, X, Y, Z) <= 0) continue;
        if (cameraDepth(cur.pose_cw,         X, Y, Z) <= 0) continue;

        Point2d p1p = projectThrough(P1, X, Y, Z);
        Point2d p2p = projectThrough(P2, X, Y, Z);
        double e1 = std::hypot(p1p.x - pts1[i].x, p1p.y - pts1[i].y);
        double e2 = std::hypot(p2p.x - pts2[i].x, p2p.y - pts2[i].y);
        if (e1 > params_.pnp_reproj_thresh || e2 > params_.pnp_reproj_thresh) continue;

        Point3d Xw(X, Y, Z);
        if (detail::parallaxDeg(Xw, last_kf_->pose_cw, cur.pose_cw)
                < params_.min_growth_parallax_deg) continue;

        // Create and register the new MapPoint.
        MapPoint* mp     = new MapPoint();
        mp->pos          = Xw;
        const DMatch& dm = kf_to_cur[tri_match_idx[i]];
        mp->ref_desc     = cur.descriptors.row(dm.trainIdx).clone();

        map_.addMapPoint(mp);
        map_.addObservation(last_kf_, (size_t)dm.queryIdx, mp);
        map_.addObservation(new_kf,   (size_t)dm.trainIdx, mp);
        ++n_new;
    }

    // -------------------------------------------------------------------------
    // 5. Update covisibility graph; commit the new KF as the reference.
    // -------------------------------------------------------------------------
    detail::updateCovisibility(new_kf);

    map_.setCurrentKeyframe(new_kf);
    last_kf_         = new_kf;
    last_kf_inliers_ = 0; // will be recomputed on the next tracking frame
    frames_since_kf_ = 0;

    (void)n_new; // used implicitly via map_.numMapPoints() in the caller's log
}

}} // namespace cv::slam
