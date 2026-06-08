// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_VISUAL_ODOMETRY_HPP
#define OPENCV_SLAM_VISUAL_ODOMETRY_HPP

#include "opencv2/core.hpp"
#include "opencv2/features.hpp"

#include "opencv2/slam/types.hpp"
#include "opencv2/slam/map.hpp"
#include "opencv2/slam/odometry_params.hpp"

#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Monocular visual odometry pipeline.

Construct with @ref create supplying a folder of images, an output folder for
artifacts, and the camera intrinsics; call @ref run to process the whole
sequence and write outputs to disk. For incremental use, @ref processFrame
feeds one frame at a time.

State machine:
- `NOT_INITIALIZED` -> first frame stored as reference, state advances to
  `INITIALIZING`.
- `INITIALIZING` -> for each subsequent frame, attempt H/F two-view
  bootstrap (homography vs. fundamental scored by RH = nH/(nH+nE),
  decomposed and triangulated). On success, two keyframes and the initial
  map points are committed; state advances to `TRACKING`.
- `TRACKING` -> per-frame localisation via motion-model (Stage B),
  reference-KF descriptor match (Stage C), optical-flow (Stage D), and
  local-map refinement (Stage E). Keyframes are promoted when the inlier
  ratio drops or rotation/timeout conditions trigger. Tracking failure
  rewinds the state to `INITIALIZING`.

@ref run writes the following files into `outputFolder` (created if missing):
- `trajectory.txt` — one camera center (Cx Cy Cz, world coordinates) per
  successfully tracked frame.
- `trajectory.bin` — binary dump: 4-byte magic "VOTR", int32 version, int32
  pose count, then 16 doubles (T_cw row-major) per pose.
- `images.txt` — COLMAP-style pose dump, one row per emitted pose:
  `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME` followed by an empty
  POINTS2D[] line.
- `map_points.txt` — `id X Y Z` per persisted map point.
- `keypoints.txt` — keypoints for every keyframe (id, position, score).
- `vo.log` — per-frame textual log of state transitions and map growth.
*/
class CV_EXPORTS VisualOdometry
{
public:
    virtual ~VisualOdometry();

    /** @brief Create a VisualOdometry instance.

    @param detector       Feature detector/descriptor (e.g. @ref ALIKED).
    @param matcher        Descriptor matcher (e.g. @ref LightGlueMatcher).
    @param imagesFolder   Directory containing input images. Files are
                          processed in sorted order; image extensions
                          (.jpg/.jpeg/.png/.bmp/.tif/.tiff/.pgm/.ppm) are
                          auto-detected.
    @param outputFolder   Directory where @ref run will write its artifacts.
                          Created if it does not exist. Pass an empty string
                          to disable file output.
    @param cameraMatrix   3x3 camera intrinsic matrix.
    @param distCoeffs     Distortion coefficients (empty for no distortion).
    @param params         Tunable parameters; see @ref OdometryParams.
    */
    static Ptr<VisualOdometry> create(
        const Ptr<Feature2D>& detector,
        const Ptr<DescriptorMatcher>& matcher,
        const String& imagesFolder,
        const String& outputFolder,
        InputArray cameraMatrix,
        InputArray distCoeffs = noArray(),
        const OdometryParams& params = OdometryParams());

    /** @brief Run the pipeline over every image in the configured folder,
    writing trajectory / map / keypoints / log into the output folder.

    @return true if at least one image was processed successfully. */
    virtual bool run() = 0;

    // --- Incremental API (for streaming use) --------------------------------

    /** Feed the next image into the pipeline.
        @return true if a world->camera pose was emitted for this frame. */
    virtual bool processFrame(InputArray image) = 0;

    /** Reset to NOT_INITIALIZED. Clears the map and trajectory. */
    virtual void reset() = 0;

    // --- Accessors ----------------------------------------------------------

    virtual OdometryState getState() const = 0;
    virtual Matx44d getLastPose() const = 0;
    virtual const Map& getMap() const = 0;
    virtual const std::vector<Matx44d>& getTrajectory() const = 0;

    virtual const OdometryParams& getParams() const = 0;
    virtual void setParams(const OdometryParams& params) = 0;

    // --- Modular backend toggles --------------------------------------------
    // The front-end (bootstrap, PnP tracking, map growth) always runs; these
    // switches decide whether the g2o refinement runs on top of it. Both
    // default to enabled and may be flipped at any time between frames. They
    // have no effect if the module was built without g2o.

    /** Enable/disable g2o pose-only optimization in the tracking stages.
        When off, the PnP pose is used directly with a reprojection-only
        inlier check (no 6-DoF refinement). Default: enabled. */
    virtual void setPoseOptimization(bool enable) = 0;
    virtual bool getPoseOptimization() const = 0;

    /** Enable/disable local bundle adjustment at each keyframe promotion.
        When off, the map still grows but keyframe poses and map points are
        not jointly refined. Default: enabled. */
    virtual void setLocalBA(bool enable) = 0;
    virtual bool getLocalBA() const = 0;

    virtual const String& getImagesFolder() const = 0;
    virtual const String& getOutputFolder() const = 0;
    virtual void setOutputFolder(const String& outputFolder) = 0;

protected:
    VisualOdometry();
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_VISUAL_ODOMETRY_HPP
