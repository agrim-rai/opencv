// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "optimizer.hpp"

namespace cv { namespace slam {

// =============================================================================
//  Reprojection-only association check  (always compiled)
//
//  Used as the fallback for PoseOptimization when g2o pose optimization is
//  disabled at runtime (enable == false) or when the module was built without
//  g2o. It does NOT move frame.pose_cw — it only
//  classifies each keypoint↔MapPoint association as inlier/outlier by
//  reprojection distance, which the tracking stages still need.
// =============================================================================

static int poseOptimizationReproj(Frame& frame, const Mat& K, double reproj_thresh)
{
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    const Matx44d& T = frame.pose_cw;

    int n_inliers = 0;
    for (size_t i = 0; i < frame.mappoints.size(); ++i)
    {
        frame.outliers[i] = true;
        MapPoint* mp = frame.mappoints[i];
        if (!mp || mp->bad) continue;

        const double Xc = T(0,0)*mp->pos.x + T(0,1)*mp->pos.y + T(0,2)*mp->pos.z + T(0,3);
        const double Yc = T(1,0)*mp->pos.x + T(1,1)*mp->pos.y + T(1,2)*mp->pos.z + T(1,3);
        const double Zc = T(2,0)*mp->pos.x + T(2,1)*mp->pos.y + T(2,2)*mp->pos.z + T(2,3);
        if (Zc <= 0.0) continue;

        const double u  = fx * Xc / Zc + cx;
        const double v  = fy * Yc / Zc + cy;
        const double dx = u - static_cast<double>(frame.undist_kpts[i].x);
        const double dy = v - static_cast<double>(frame.undist_kpts[i].y);

        if (std::sqrt(dx * dx + dy * dy) <= reproj_thresh)
        {
            frame.outliers[i] = false;
            ++n_inliers;
        }
    }
    return n_inliers;
}

}} // namespace cv::slam

// =============================================================================
//  Real g2o backends  (compiled when HAVE_G2O is defined by CMake)
// =============================================================================
#ifdef HAVE_G2O

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

// Dense solver — for PoseOptimization (small problem, ~300 edges)
#if __has_include(<g2o/solvers/dense/linear_solver_dense.h>)
#  include <g2o/solvers/dense/linear_solver_dense.h>
#elif __has_include(<g2o/solvers/linear_solver_dense.h>)
#  include <g2o/solvers/linear_solver_dense.h>
#else
#  error "Cannot locate g2o LinearSolverDense header."
#endif

// Eigen solver — for LocalBA (larger sparse problem)
#if __has_include(<g2o/solvers/eigen/linear_solver_eigen.h>)
#  include <g2o/solvers/eigen/linear_solver_eigen.h>
#elif __has_include(<g2o/solvers/linear_solver_eigen.h>)
#  include <g2o/solvers/linear_solver_eigen.h>
#else
#  error "Cannot locate g2o LinearSolverEigen header."
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cv { namespace slam {

// -----------------------------------------------------------------------------
// EdgeSE3ProjectXYZOnlyPose
//
// Unary reprojection edge: one variable vertex (camera pose T_cw as SE3Quat),
// one FIXED 3D point Xw baked into the edge at construction time.
//
// Error = obs_2d − π(T_cw · Xw)
//
// Jacobian convention (matches g2o's bundled ORB-SLAM SE3Quat):
//   update vector = [ω₀, ω₁, ω₂, υ₀, υ₁, υ₂]  (rotation first, then translation)
//   perturbation applied LEFT:  T_new = exp(δ) * T_cw
//
// Derivation (camera-frame point Xc = T_cw.map(Xw) = [x, y, z]):
//   dXc/d[ω, υ] = [-hat(Xc), I₃]
//   dproj/dXc   = [[fx/z,  0, -fx·x/z²],
//                  [  0, fy/z, -fy·y/z²]]
//   J = d(obs − proj)/d[ω, υ] = − dproj/dXc · dXc/d[ω, υ]
// -----------------------------------------------------------------------------

class EdgeSE3ProjectXYZOnlyPose :
    public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZOnlyPose(const Eigen::Vector3d& Xw,
                               double fx, double fy, double cx, double cy)
        : Xw_(Xw), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
    bool read(std::istream&)        { return false; }
    bool write(std::ostream&) const { return false; }
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

    void computeError() CV_OVERRIDE
    {
        const auto* v = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const Eigen::Vector3d Xc = v->estimate().map(Xw_);
        _error = _measurement - Eigen::Vector2d(fx_ * Xc[0] / Xc[2] + cx_,
                                                 fy_ * Xc[1] / Xc[2] + cy_);
    }

    bool isDepthPositive() const
    {
        const auto* v = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return v->estimate().map(Xw_)[2] > 0.0;
    }

    // Analytical Jacobian — avoids the cost of numerical differentiation.
    // Columns [0,1,2] = rotation (ω), columns [3,4,5] = translation (υ).
    void linearizeOplus() CV_OVERRIDE
    {
        const auto* v = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const Eigen::Vector3d Xc   = v->estimate().map(Xw_);
        const double x    = Xc[0], y = Xc[1];
        const double invz = 1.0 / Xc[2];
        const double iz2  = invz * invz;

        // Row 0  (u component)
        _jacobianOplusXi(0, 0) =  x * y * iz2 * fx_;
        _jacobianOplusXi(0, 1) = -(1.0 + x * x * iz2) * fx_;
        _jacobianOplusXi(0, 2) =  y * invz * fx_;
        _jacobianOplusXi(0, 3) = -invz * fx_;
        _jacobianOplusXi(0, 4) =  0.0;
        _jacobianOplusXi(0, 5) =  x * iz2 * fx_;

        // Row 1  (v component)
        _jacobianOplusXi(1, 0) =  (1.0 + y * y * iz2) * fy_;
        _jacobianOplusXi(1, 1) = -x * y * iz2 * fy_;
        _jacobianOplusXi(1, 2) = -x * invz * fy_;
        _jacobianOplusXi(1, 3) =  0.0;
        _jacobianOplusXi(1, 4) = -invz * fy_;
        _jacobianOplusXi(1, 5) =  y * iz2 * fy_;
    }

private:
    Eigen::Vector3d Xw_;
    double fx_, fy_, cx_, cy_;
};

// -----------------------------------------------------------------------------
// poseOptimizationG2O — real 6-DoF pose-only bundle adjustment
// -----------------------------------------------------------------------------

static int poseOptimizationG2O(Frame& frame, const Mat& K)
{
    // Intrinsics
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    // ---- Build optimizer ---------------------------------------------------
    // BlockSolver<6 pose, 3 landmark> — only pose is variable here.
    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LSolver = g2o::LinearSolverDense<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // Modern g2o (conda-forge) BlockSolver and LM take std::unique_ptr.
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));

    // ---- Pose vertex -------------------------------------------------------
    // Convert Matx44d T_cw → g2o::SE3Quat
    const Matx44d& T = frame.pose_cw;
    Eigen::Matrix3d R_eig;
    R_eig << T(0,0), T(0,1), T(0,2),
             T(1,0), T(1,1), T(1,2),
             T(2,0), T(2,1), T(2,2);
    const Eigen::Vector3d t_eig(T(0,3), T(1,3), T(2,3));

    auto* vPose = new g2o::VertexSE3Expmap();
    vPose->setId(0);
    vPose->setFixed(false);
    vPose->setEstimate(g2o::SE3Quat(Eigen::Quaterniond(R_eig).normalized(), t_eig));
    optimizer.addVertex(vPose);

    // ---- One edge per keypoint-to-MapPoint association --------------------
    constexpr double kHuberDelta = 2.4495;  // sqrt(5.991) — chi² 95%, 2-DOF
    constexpr double kChi2Thresh = 5.991;
    constexpr int    kIters      = 4;

    const int N = static_cast<int>(frame.mappoints.size());
    std::vector<EdgeSE3ProjectXYZOnlyPose*> edges(N, nullptr);

    for (int i = 0; i < N; ++i)
    {
        frame.outliers[i] = true;   // assume outlier until proven otherwise
        MapPoint* mp = frame.mappoints[i];
        if (!mp || mp->bad) continue;

        auto* e = new EdgeSE3ProjectXYZOnlyPose(
            Eigen::Vector3d(mp->pos.x, mp->pos.y, mp->pos.z),
            fx, fy, cx, cy);

        e->setId(i + 1);
        e->setVertex(0, vPose);
        e->setMeasurement(Eigen::Vector2d(frame.undist_kpts[i].x,
                                          frame.undist_kpts[i].y));
        e->setInformation(Eigen::Matrix2d::Identity());

        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(kHuberDelta);
        e->setRobustKernel(rk);

        optimizer.addEdge(e);
        edges[i] = e;
    }

    // ---- Pass 1 (all edges, Huber) -----------------------------------------
    optimizer.initializeOptimization(0);
    optimizer.optimize(kIters);

    // Mark outliers; prepare pass 2 (inliers only, no robust kernel).
    for (int i = 0; i < N; ++i)
    {
        auto* e = edges[i];
        if (!e) continue;

        const bool good = (e->chi2() < kChi2Thresh) && e->isDepthPositive();
        if (good)
        {
            e->setLevel(0);
            e->setRobustKernel(nullptr);  // pure L2 for pass 2 (tighter)
        }
        else
        {
            e->setLevel(1);               // exclude from pass 2
        }
    }

    // ---- Pass 2 (inliers only, L2) -----------------------------------------
    {
        int n_pass2 = 0;
        for (int i = 0; i < N; ++i)
            if (edges[i] && edges[i]->level() == 0) ++n_pass2;
        if (n_pass2 > 0)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(kIters);
        }
    }

    // ---- Write refined pose back and count final inliers -------------------
    {
        const g2o::SE3Quat est = vPose->estimate();
        const Eigen::Quaterniond q = est.rotation();
        const Eigen::Vector3d    t = est.translation();
        const Eigen::Matrix3d    R = q.toRotationMatrix();
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c) frame.pose_cw(r, c) = R(r, c);
            frame.pose_cw(r, 3) = t[r];
        }
        frame.pose_cw(3, 0) = frame.pose_cw(3, 1) = frame.pose_cw(3, 2) = 0.0;
        frame.pose_cw(3, 3) = 1.0;
    }

    int n_inliers = 0;
    for (int i = 0; i < N; ++i)
    {
        auto* e = edges[i];
        if (!e) continue;
        const bool good = (e->level() == 0)
                       && (e->chi2() < kChi2Thresh)
                       && e->isDepthPositive();
        frame.outliers[i] = !good;
        if (good) ++n_inliers;
    }
    return n_inliers;
}

// =============================================================================
//  EdgeSE3ProjectXYZ — binary reprojection edge for Local BA
//
//  vertex 0 : g2o::VertexSBAPointXYZ  (3D world point — variable)
//  vertex 1 : g2o::VertexSE3Expmap    (camera pose   — variable or fixed)
//
//  error = obs_2d − π(T_cw · Xw)
//
//  Jacobians:
//    _jacobianOplusXi (2×3) = d(error)/d(Xw) = −d(proj)/d(Xc) · R
//    _jacobianOplusXj (2×6) = d(error)/d(pose) — identical to unary edge
// =============================================================================

class EdgeSE3ProjectXYZ :
    public g2o::BaseBinaryEdge<2, Eigen::Vector2d,
                               g2o::VertexSBAPointXYZ,
                               g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ(double fx, double fy, double cx, double cy)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
    bool read(std::istream&)        { return false; }
    bool write(std::ostream&) const { return false; }
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

    void computeError() CV_OVERRIDE
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d Xc = vT->estimate().map(vp->estimate());
        _error = _measurement - Eigen::Vector2d(fx_ * Xc[0] / Xc[2] + cx_,
                                                 fy_ * Xc[1] / Xc[2] + cy_);
    }

    bool isDepthPositive() const
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        return vT->estimate().map(vp->estimate())[2] > 0.0;
    }

    void linearizeOplus() CV_OVERRIDE
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d Xc = vT->estimate().map(vp->estimate());
        const double x = Xc[0], y = Xc[1];
        const double invz = 1.0 / Xc[2], iz2 = invz * invz;
        const Eigen::Matrix3d R = vT->estimate().rotation().toRotationMatrix();

        // d(error)/d(Xw) = −d(proj)/d(Xc) · R   (2×3)
        for (int j = 0; j < 3; ++j)
        {
            _jacobianOplusXi(0, j) = -fx_ * invz * R(0,j) + fx_ * x * iz2 * R(2,j);
            _jacobianOplusXi(1, j) = -fy_ * invz * R(1,j) + fy_ * y * iz2 * R(2,j);
        }

        // d(error)/d(pose)  (2×6) — same formula as unary edge
        _jacobianOplusXj(0, 0) =  x * y * iz2 * fx_;
        _jacobianOplusXj(0, 1) = -(1.0 + x * x * iz2) * fx_;
        _jacobianOplusXj(0, 2) =  y * invz * fx_;
        _jacobianOplusXj(0, 3) = -invz * fx_;
        _jacobianOplusXj(0, 4) =  0.0;
        _jacobianOplusXj(0, 5) =  x * iz2 * fx_;

        _jacobianOplusXj(1, 0) =  (1.0 + y * y * iz2) * fy_;
        _jacobianOplusXj(1, 1) = -x * y * iz2 * fy_;
        _jacobianOplusXj(1, 2) = -x * invz * fy_;
        _jacobianOplusXj(1, 3) =  0.0;
        _jacobianOplusXj(1, 4) = -invz * fy_;
        _jacobianOplusXj(1, 5) =  y * iz2 * fy_;
    }

private:
    double fx_, fy_, cx_, cy_;
};

// -----------------------------------------------------------------------------
// localBundleAdjustmentG2O
// -----------------------------------------------------------------------------

static void localBundleAdjustmentG2O(KeyFrame* new_kf, const Mat& K,
                                     bool* stop_flag)
{
    if (!new_kf) return;

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

    constexpr int    kMaxLocalKFs = 10;
    constexpr double kChi2Thresh  = 5.991;
    constexpr double kHuberDelta  = 2.4495;  // sqrt(5.991)
    constexpr int    kItersCoarse = 5;
    constexpr int    kItersFine   = 10;

    // ---- 1. Local KFs: new_kf + top-K covisible ----------------------------
    std::vector<KeyFrame*> local_kfs;
    std::set<KeyFrame*>    local_kf_set;
    local_kfs.push_back(new_kf);
    local_kf_set.insert(new_kf);

    int k = 0;
    for (const auto& [nb_kf, cnt] : new_kf->ordered_covisibility)
    {
        if (k++ >= kMaxLocalKFs) break;
        if (!nb_kf) continue;
        local_kfs.push_back(nb_kf);
        local_kf_set.insert(nb_kf);
    }
    if (local_kfs.size() < 2) return;

    // ---- 2. Local MPs: all seen by any local KF ----------------------------
    std::vector<MapPoint*> local_mps;
    std::set<MapPoint*>    local_mp_set;
    for (KeyFrame* kf : local_kfs)
        for (MapPoint* mp : kf->mappoints)
            if (mp && !mp->bad && !local_mp_set.count(mp))
            { local_mps.push_back(mp); local_mp_set.insert(mp); }

    if (local_mps.empty()) return;

    // ---- 3. Fixed KFs: observe local MPs but are outside the window --------
    std::set<KeyFrame*> fixed_kf_set;
    for (MapPoint* mp : local_mps)
        for (const auto& [obs_kf, kp_idx] : mp->observations)
            if (!local_kf_set.count(obs_kf))
                fixed_kf_set.insert(obs_kf);

    // Monocular gauge freedom: need >= 2 anchors. If not enough external
    // fixed KFs, promote the oldest local KFs to fixed.
    if (fixed_kf_set.size() < 2)
    {
        std::vector<KeyFrame*> by_age = local_kfs;
        std::sort(by_age.begin(), by_age.end(),
                  [](const KeyFrame* a, const KeyFrame* b){ return a->id < b->id; });
        for (KeyFrame* kf : by_age)
        {
            if (fixed_kf_set.size() >= 2) break;
            if (kf == new_kf) continue;
            local_kf_set.erase(kf);
            fixed_kf_set.insert(kf);
        }
    }

    // maxKFid used to offset MP vertex IDs so they don't collide with KF IDs.
    int maxKFid = 0;
    for (KeyFrame* kf : local_kfs)    maxKFid = std::max(maxKFid, kf->id);
    for (KeyFrame* kf : fixed_kf_set) maxKFid = std::max(maxKFid, kf->id);

    // ---- 4. Build g2o optimizer --------------------------------------------
    using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));

    if (stop_flag) optimizer.setForceStopFlag(stop_flag);

    // Helper: Matx44d pose_cw → g2o::SE3Quat
    auto toSE3 = [](const KeyFrame* kf) -> g2o::SE3Quat {
        const Matx44d& T = kf->pose_cw;
        Eigen::Matrix3d R;
        R << T(0,0), T(0,1), T(0,2),
             T(1,0), T(1,1), T(1,2),
             T(2,0), T(2,1), T(2,2);
        return g2o::SE3Quat(Eigen::Quaterniond(R).normalized(),
                            Eigen::Vector3d(T(0,3), T(1,3), T(2,3)));
    };

    // ---- 5. KF vertices ----------------------------------------------------
    for (KeyFrame* kf : local_kfs)
    {
        if (!local_kf_set.count(kf)) continue;  // promoted to fixed
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(kf->id);
        v->setEstimate(toSE3(kf));
        v->setFixed(false);
        optimizer.addVertex(v);
    }
    for (KeyFrame* kf : fixed_kf_set)
    {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(kf->id);
        v->setEstimate(toSE3(kf));
        v->setFixed(true);
        optimizer.addVertex(v);
    }

    // ---- 6. MP vertices (marginalized) -------------------------------------
    for (MapPoint* mp : local_mps)
    {
        auto* v = new g2o::VertexSBAPointXYZ();
        v->setId(mp->id + maxKFid + 1);
        v->setEstimate(Eigen::Vector3d(mp->pos.x, mp->pos.y, mp->pos.z));
        v->setMarginalized(true);
        optimizer.addVertex(v);
    }

    // ---- 7. Reprojection edges (one per KF×MP observation) -----------------
    struct EdgeRec { EdgeSE3ProjectXYZ* e; KeyFrame* kf; MapPoint* mp; size_t kp_idx; };
    std::vector<EdgeRec> recs;
    recs.reserve(local_mps.size() * 4);

    for (MapPoint* mp : local_mps)
    {
        const int pt_vid = mp->id + maxKFid + 1;
        for (const auto& [obs_kf, kp_idx] : mp->observations)
        {
            if (!local_kf_set.count(obs_kf) && !fixed_kf_set.count(obs_kf)) continue;
            if (kp_idx >= obs_kf->undist_kpts.size()) continue;

            auto* e = new EdgeSE3ProjectXYZ(fx, fy, cx, cy);
            e->setVertex(0, optimizer.vertex(pt_vid));
            e->setVertex(1, optimizer.vertex(obs_kf->id));
            e->setMeasurement(Eigen::Vector2d(obs_kf->undist_kpts[kp_idx].x,
                                              obs_kf->undist_kpts[kp_idx].y));
            e->setInformation(Eigen::Matrix2d::Identity());

            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(kHuberDelta);
            e->setRobustKernel(rk);

            optimizer.addEdge(e);
            recs.push_back({e, obs_kf, mp, kp_idx});
        }
    }

    if (recs.empty()) return;
    if (stop_flag && *stop_flag) return;

    // ---- 8. Pass 1: coarse iters, all edges, Huber -------------------------
    optimizer.initializeOptimization();
    optimizer.optimize(kItersCoarse);

    // Cull outliers; remove Huber from inliers for tighter pass 2.
    for (auto& r : recs)
    {
        const bool bad = (r.e->chi2() > kChi2Thresh) || !r.e->isDepthPositive();
        r.e->setLevel(bad ? 1 : 0);
        if (!bad) r.e->setRobustKernel(nullptr);
    }

    // ---- 9. Pass 2: fine iters, inliers only, L2 ---------------------------
    if (stop_flag && *stop_flag) return;
    {
        int n_pass2 = 0;
        for (auto& r : recs)
            if (r.e->level() == 0) ++n_pass2;
        if (n_pass2 > 0)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(kItersFine);
        }
    }

    // ---- 10. Write back: KF poses ------------------------------------------
    for (KeyFrame* kf : local_kfs)
    {
        if (!local_kf_set.count(kf)) continue;  // was promoted to fixed
        auto* v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf->id));
        if (!v) continue;
        const Eigen::Quaterniond q = v->estimate().rotation();
        const Eigen::Vector3d    t = v->estimate().translation();
        const Eigen::Matrix3d    R = q.toRotationMatrix();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) kf->pose_cw(r, c) = R(r, c);
            kf->pose_cw(r, 3) = t[r];
        }
        kf->pose_cw(3, 0) = kf->pose_cw(3, 1) = kf->pose_cw(3, 2) = 0.0;
        kf->pose_cw(3, 3) = 1.0;
    }

    // ---- 11. Write back: MP positions --------------------------------------
    for (MapPoint* mp : local_mps)
    {
        auto* v = static_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(mp->id + maxKFid + 1));
        if (!v) continue;
        const auto est = v->estimate();
        mp->pos = cv::Point3d(est[0], est[1], est[2]);
    }

    // ---- 12. Erase final outlier KF↔MP links ------------------------------
    for (auto& r : recs)
    {
        if (r.e->level() == 0) continue;  // inlier, keep
        if (r.kp_idx < r.kf->mappoints.size())
            r.kf->mappoints[r.kp_idx] = nullptr;
        r.mp->observations.erase(r.kf);
        if (r.mp->observations.empty())
            r.mp->bad = true;
    }
}

}} // namespace cv::slam

#endif // HAVE_G2O

// =============================================================================
//  Public entry points (always compiled) — dispatch on runtime toggles
// =============================================================================

namespace cv { namespace slam { namespace Optimizer {

int PoseOptimization(Frame& frame, const Mat& K, double reproj_thresh, bool enable)
{
#ifdef HAVE_G2O
    if (enable)
        return poseOptimizationG2O(frame, K);
#else
    (void)enable;
#endif
    // Disabled at runtime, or g2o not compiled in: reprojection check only.
    return poseOptimizationReproj(frame, K, reproj_thresh);
}

void LocalBundleAdjustment(KeyFrame* new_kf, const Mat& K, bool enable,
                           bool* stop_flag)
{
#ifdef HAVE_G2O
    if (enable)
        localBundleAdjustmentG2O(new_kf, K, stop_flag);
#else
    (void)new_kf; (void)K; (void)enable; (void)stop_flag;  // no-op without g2o
#endif
}

} // namespace Optimizer
}} // namespace cv::slam
