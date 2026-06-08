// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "vo_impl.hpp"

#include <fstream>
#include <sstream>
#include <cctype>

namespace cv {
namespace slam {

namespace {

bool isImageFile(const String& path)
{
    static const char* kExts[] =
        { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pgm", ".ppm" };
    auto dot = path.find_last_of('.');
    if (dot == String::npos) return false;
    String ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    for (const char* e : kExts)
        if (ext == e) return true;
    return false;
}

const char* stateName(OdometryState s)
{
    switch (s)
    {
    case NOT_INITIALIZED: return "NOT_INITIALIZED";
    case INITIALIZING:    return "INITIALIZING";
    case TRACKING:        return "TRACKING";
    }
    return "UNKNOWN";
}

String joinPath(const String& dir, const String& name)
{
    if (dir.empty()) return name;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

VisualOdometry::VisualOdometry()  = default;
VisualOdometry::~VisualOdometry() = default;

Ptr<VisualOdometry> VisualOdometry::create(
    const Ptr<Feature2D>& detector,
    const Ptr<DescriptorMatcher>& matcher,
    const String& imagesFolder,
    const String& outputFolder,
    InputArray cameraMatrix,
    InputArray distCoeffs,
    const OdometryParams& params)
{
    CV_Assert(detector && "VisualOdometry::create: detector must not be null");
    CV_Assert(matcher  && "VisualOdometry::create: matcher must not be null");

    Mat K = cameraMatrix.getMat();
    CV_Assert(!K.empty() && K.rows == 3 && K.cols == 3);
    Mat dist = distCoeffs.empty() ? Mat() : distCoeffs.getMat();

    return makePtr<VisualOdometryImpl>(
        detector, matcher, imagesFolder, outputFolder, K, dist, params);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

VisualOdometryImpl::VisualOdometryImpl(
    const Ptr<Feature2D>& detector,
    const Ptr<DescriptorMatcher>& matcher,
    const String& imagesFolder,
    const String& outputFolder,
    const Mat& cameraMatrix,
    const Mat& distCoeffs,
    const OdometryParams& params)
    : detector_(detector), matcher_(matcher), params_(params),
      images_folder_(imagesFolder), output_folder_(outputFolder)
{
    cameraMatrix.convertTo(K_, CV_64F);
    if (!distCoeffs.empty())
        distCoeffs.convertTo(dist_, CV_64F);
}

// ---------------------------------------------------------------------------
// reset / processFrame
// ---------------------------------------------------------------------------

void VisualOdometryImpl::reset()
{
    state_           = NOT_INITIALIZED;
    last_pose_cw_    = Matx44d::eye();
    ref_frame_       = Frame();
    last_kf_         = nullptr;
    frames_since_kf_ = 0;
    last_kf_inliers_ = 0;
    velocity_        = Matx44d::eye();
    has_velocity_    = false;
    prev_frame_      = Frame();
    has_prev_frame_  = false;
    last_event_.clear();
    pose_filenames_.clear();
    map_.clear();
}

bool VisualOdometryImpl::processFrame(InputArray image)
{
    CV_INSTRUMENT_REGION();

    if (image.empty()) return false;
    last_event_.clear();

    Frame cur;
    extractFeatures(image, cur);
    if (cur.keypoints.empty() || cur.descriptors.empty()) return false;

    cur.mappoints.assign(cur.keypoints.size(), nullptr);
    cur.outliers.assign(cur.keypoints.size(), false);
    cur.buildGrid();

    switch (state_)
    {
    case NOT_INITIALIZED:
        ref_frame_ = cur;
        state_     = INITIALIZING;
        return false;

    case INITIALIZING:
        return bootstrap(cur);

    case TRACKING:
        return track(cur);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

void VisualOdometryImpl::extractFeatures(InputArray image, Frame& out) const
{
    Mat img = image.getMat();
    out.imageSize = img.size();
    out.keypoints.clear();

    // Detect and compute on the original image (color/grey is up to the detector).
    detector_->detectAndCompute(img, noArray(), out.keypoints, out.descriptors);

    // Store a greyscale copy for the optical-flow fallback (Stage D).
    if (img.channels() > 1)
        cvtColor(img, out.image, COLOR_BGR2GRAY);
    else
        out.image = img.clone();

    // Pre-compute undistorted pixel coordinates used by every stage.
    if (!out.keypoints.empty())
    {
        std::vector<Point2f> raw;
        raw.reserve(out.keypoints.size());
        for (const auto& kp : out.keypoints)
            raw.push_back(kp.pt);

        if (!dist_.empty())
            undistortPoints(raw, out.undist_kpts, K_, dist_, noArray(), K_);
        else
            out.undist_kpts = raw;
    }
}

// ---------------------------------------------------------------------------
// Frame matching helper
// ---------------------------------------------------------------------------

void VisualOdometryImpl::matchFrames(
    const std::vector<KeyPoint>& q_kp, const Mat& q_desc, Size q_sz,
    const std::vector<KeyPoint>& t_kp, const Mat& t_desc, Size t_sz,
    std::vector<DMatch>& matches) const
{
    matches.clear();
    if (q_desc.empty() || t_desc.empty()) return;
    if (q_kp.empty()   || t_kp.empty())   return;

    LightGlueMatcher* lg = dynamic_cast<LightGlueMatcher*>(matcher_.get());
    if (lg)
    {
        Mat qk((int)q_kp.size(), 2, CV_32F);
        for (size_t i = 0; i < q_kp.size(); ++i)
        { qk.at<float>((int)i,0) = q_kp[i].pt.x; qk.at<float>((int)i,1) = q_kp[i].pt.y; }

        Mat tk((int)t_kp.size(), 2, CV_32F);
        for (size_t i = 0; i < t_kp.size(); ++i)
        { tk.at<float>((int)i,0) = t_kp[i].pt.x; tk.at<float>((int)i,1) = t_kp[i].pt.y; }

        lg->setPairInfo(qk, tk, q_sz, t_sz);
    }

    matcher_->match(q_desc, t_desc, matches);
}

// ---------------------------------------------------------------------------
// Batch run()
// ---------------------------------------------------------------------------

bool VisualOdometryImpl::run()
{
    CV_INSTRUMENT_REGION();

    if (images_folder_.empty())
    {
        CV_LOG_ERROR(NULL, "VisualOdometry::run: imagesFolder is empty");
        return false;
    }

    std::vector<String> all_files;
    try { cv::glob(images_folder_, all_files, false); }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VisualOdometry::run: glob failed: " << e.what());
        return false;
    }

    std::vector<String> img_files;
    img_files.reserve(all_files.size());
    for (const auto& f : all_files)
        if (isImageFile(f)) img_files.push_back(f);
    std::sort(img_files.begin(), img_files.end());

    if (img_files.empty())
    {
        CV_LOG_WARNING(NULL, "VisualOdometry::run: no images in " << images_folder_);
        return false;
    }

    std::ofstream log;
    if (!output_folder_.empty())
    {
        cv::utils::fs::createDirectories(output_folder_);
        log.open(joinPath(output_folder_, "vo.log").c_str());
    }
    auto logln = [&](const String& s) { if (log.is_open()) log << s << "\n"; };

    #ifdef HAVE_G2O
    {
        std::ostringstream os;
        os << "[INFO] optimizer = g2o"
           << " | pose_optimization=" << (enable_pose_opt_ ? "on" : "off")
           << " | local_ba="          << (enable_local_ba_ ? "on" : "off");
        logln(os.str());
    }
    #else
        logln("[INFO] optimizer = stub (g2o not compiled in; reprojection inlier "
              "check only — pose_optimization/local_ba toggles ignored)");
    #endif
    logln(String("[INFO] images_folder = ") + images_folder_);
    logln(String("[INFO] output_folder = ") + output_folder_);
    {
        std::ostringstream ss;
        ss << "[INFO] found " << img_files.size() << " image(s)";
        logln(ss.str());
    }

    reset();

    int    n_emitted     = 0;
    size_t prev_traj_len = 0;
    String ref_filename;

    for (size_t i = 0; i < img_files.size(); ++i)
    {
        Mat img = imread(img_files[i]);
        if (img.empty())
        {
            std::ostringstream ss;
            ss << "[FRAME " << i << "] file=" << img_files[i] << " ERROR: imread failed";
            logln(ss.str()); continue;
        }

        OdometryState before = state_;
        bool emitted = processFrame(img);
        OdometryState after  = state_;
        if (emitted) ++n_emitted;

        // Track which input image maps to each trajectory pose.
        if (before == NOT_INITIALIZED ||
            (before == TRACKING && after == INITIALIZING))
            ref_filename = img_files[i];

        const size_t added = map_.trajectory().size() - prev_traj_len;
        if (added == 1)
            pose_filenames_.push_back(img_files[i]);
        else if (added == 2)
        {
            pose_filenames_.push_back(ref_filename);
            pose_filenames_.push_back(img_files[i]);
        }
        prev_traj_len = map_.trajectory().size();

        std::ostringstream ss;
        ss << "[FRAME " << i << "] file=" << img_files[i]
           << " state=" << stateName(before);
        if (before != after) ss << "->" << stateName(after);
        ss << " emitted=" << (emitted ? "yes" : "no")
           << " keyframes=" << map_.numKeyframes()
           << " map_points=" << map_.numMapPoints();
        if (!last_event_.empty()) ss << " [" << last_event_ << "]";
        if (emitted)
        {
            Point3d C = detail::cameraCenterWorld(last_pose_cw_);
            ss << " C=(" << C.x << "," << C.y << "," << C.z << ")";
        }
        logln(ss.str());
    }

    if (!output_folder_.empty())
    {
        writeTrajectoryText(joinPath(output_folder_, "trajectory.txt"));
        writeTrajectoryBin (joinPath(output_folder_, "trajectory.bin"));
        writeMapPoints     (joinPath(output_folder_, "map_points.txt"));
        writeKeypoints     (joinPath(output_folder_, "keypoints.txt"));
        writeImagesTxt     (joinPath(output_folder_, "images.txt"));

        std::ostringstream ss;
        ss << "[INFO] run complete: frames=" << img_files.size()
           << " emitted=" << n_emitted
           << " keyframes=" << map_.numKeyframes()
           << " map_points=" << map_.numMapPoints();
        logln(ss.str());
        logln("[INFO] wrote trajectory.txt, trajectory.bin, map_points.txt, "
              "keypoints.txt, images.txt");
    }

    return n_emitted > 0;
}

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------

void VisualOdometryImpl::writeTrajectoryText(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeTrajectoryText: cannot open " << path); return; }
    f << "# Per-frame camera center in world coordinates.\n# Columns: Cx Cy Cz\n";
    f.setf(std::ios::scientific); f.precision(9);
    for (const auto& T : map_.trajectory())
    {
        Point3d C = detail::cameraCenterWorld(T);
        f << C.x << " " << C.y << " " << C.z << "\n";
    }
}

void VisualOdometryImpl::writeTrajectoryBin(const String& path) const
{
    std::ofstream f(path.c_str(), std::ios::binary);
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeTrajectoryBin: cannot open " << path); return; }
    const char magic[4] = {'V','O','T','R'};
    f.write(magic, 4);
    int32_t version = 1;
    f.write(reinterpret_cast<const char*>(&version), sizeof(int32_t));
    int32_t n = (int32_t)map_.trajectory().size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    for (const auto& T : map_.trajectory())
    {
        double buf[16];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) buf[r*4+c] = T(r,c);
        f.write(reinterpret_cast<const char*>(buf), sizeof(buf));
    }
}

void VisualOdometryImpl::writeMapPoints(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeMapPoints: cannot open " << path); return; }
    f << "# Map points in world coordinates.\n# Columns: id X Y Z n_observations\n";
    f.setf(std::ios::scientific); f.precision(9);
    for (MapPoint* mp : map_.mapPoints())
    {
        if (!mp || mp->bad) continue;
        f << mp->id << " "
          << mp->pos.x << " " << mp->pos.y << " " << mp->pos.z << " "
          << mp->observations.size() << "\n";
    }
}

void VisualOdometryImpl::writeKeypoints(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeKeypoints: cannot open " << path); return; }
    f << "# Per-keyframe keypoints.\n"
      << "# Block header: KF kf_id Cx Cy Cz n_keypoints\n"
      << "# Followed by n_keypoints rows: kp_idx x y size angle response octave mp_id\n"
      << "# mp_id = -1 if the keypoint has no map point.\n";
    f.setf(std::ios::fixed); f.precision(6);

    for (KeyFrame* kf : map_.keyframes())
    {
        if (!kf) continue;
        Point3d C = detail::cameraCenterWorld(kf->pose_cw);
        f << "KF " << kf->id << " " << C.x << " " << C.y << " " << C.z
          << " " << kf->keypoints.size() << "\n";
        for (size_t i = 0; i < kf->keypoints.size(); ++i)
        {
            const KeyPoint& kp = kf->keypoints[i];
            int mp_id = -1;
            if (i < kf->mappoints.size() && kf->mappoints[i])
                mp_id = kf->mappoints[i]->id;
            f << i << " " << kp.pt.x << " " << kp.pt.y << " "
              << kp.size << " " << kp.angle << " " << kp.response << " "
              << kp.octave << " " << mp_id << "\n";
        }
    }
}

// Shepperd's method: numerically-stable R → unit quaternion (qw, qx, qy, qz).
static void rotMatToQuat(const Matx33d& R,
                         double& qw, double& qx, double& qy, double& qz)
{
    const double tr = R(0,0) + R(1,1) + R(2,2);
    if (tr > 0.0)
    {
        double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (R(2,1) - R(1,2)) / s;
        qy = (R(0,2) - R(2,0)) / s;
        qz = (R(1,0) - R(0,1)) / s;
    }
    else if (R(0,0) > R(1,1) && R(0,0) > R(2,2))
    {
        double s = std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2)) * 2.0;
        qw = (R(2,1) - R(1,2)) / s; qx = 0.25 * s;
        qy = (R(0,1) + R(1,0)) / s; qz = (R(0,2) + R(2,0)) / s;
    }
    else if (R(1,1) > R(2,2))
    {
        double s = std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2)) * 2.0;
        qw = (R(0,2) - R(2,0)) / s; qx = (R(0,1) + R(1,0)) / s;
        qy = 0.25 * s;              qz = (R(1,2) + R(2,1)) / s;
    }
    else
    {
        double s = std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1)) * 2.0;
        qw = (R(1,0) - R(0,1)) / s; qx = (R(0,2) + R(2,0)) / s;
        qy = (R(1,2) + R(2,1)) / s; qz = 0.25 * s;
    }
}

static String basenameOf(const String& path)
{
    const size_t slash = path.find_last_of("/\\");
    return (slash == String::npos) ? path : path.substr(slash + 1);
}

void VisualOdometryImpl::writeImagesTxt(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeImagesTxt: cannot open " << path); return; }

    const auto& traj = map_.trajectory();
    f << "# Image list with two lines of data per image:\n"
      << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
      << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
      << "# Number of images: " << traj.size() << ", mean observations per image: 0.0\n";
    f.setf(std::ios::fixed); f.precision(6);

    for (size_t i = 0; i < traj.size(); ++i)
    {
        const Matx44d& T = traj[i];
        Matx33d R;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) R(r,c) = T(r,c);
        double qw, qx, qy, qz;
        rotMatToQuat(R, qw, qx, qy, qz);

        const String name = (i < pose_filenames_.size())
            ? basenameOf(pose_filenames_[i])
            : (String("pose_") + std::to_string(i));

        f << i << " " << qw << " " << qx << " " << qy << " " << qz << " "
          << T(0,3) << " " << T(1,3) << " " << T(2,3) << " " << 1 << " " << name << "\n\n";
    }
}

}} // namespace cv::slam
