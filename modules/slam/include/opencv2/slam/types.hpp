// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_TYPES_HPP
#define OPENCV_SLAM_TYPES_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"

#include <map>
#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Lifecycle state of the visual odometry pipeline. */
enum OdometryState
{
    NOT_INITIALIZED = 0, //!< First frame ever, or just after a reset.
    INITIALIZING    = 1, //!< Reference frame held; waiting for a second view with parallax.
    TRACKING        = 2  //!< Map exists; localizing frame by frame.
};

// Forward declarations (MapPoint and KeyFrame reference each other via pointers).
struct MapPoint;
struct KeyFrame;

/** @brief A 3D landmark in world coordinates, observed by one or more keyframes.

Owned and lifetime-managed by @ref Map.  Raw pointers remain valid until the
point is removed via @ref Map::removeMapPoint or @ref Map::clear.
*/
struct CV_EXPORTS MapPoint
{
    int     id  = -1;
    Point3d pos { 0, 0, 0 };                      //!< World coordinates (x, y, z).
    Mat     ref_desc;                              //!< Best descriptor across all observations.

    std::map<KeyFrame*, size_t> observations;      //!< keyframe -> keypoint index.

    int  visible_count = 0;  //!< Times this point was projected into a tracking frame.
    int  found_count   = 0;  //!< Times it was actually matched (quality = found/visible).
    bool bad           = false; //!< Soft-delete flag; always check before use.
};

/** @brief A keyframe: 6-DoF pose, keypoints/descriptors, per-keypoint MapPoint links,
and covisibility graph edges.

Owned and lifetime-managed by @ref Map.
*/
struct CV_EXPORTS KeyFrame
{
    int     id      = -1;
    Matx44d pose_cw = Matx44d::eye();              //!< World -> camera 4×4 transform.

    std::vector<KeyPoint>  keypoints;
    Mat                    descriptors;            //!< NxD matrix; row i = descriptor for keypoints[i].
    std::vector<Point2f>   undist_kpts;            //!< Undistorted pixel coords, parallel to keypoints.
    Size                   imageSize;

    std::vector<MapPoint*> mappoints;              //!< Parallel to keypoints; null = no 3D match yet.

    std::map<KeyFrame*, int>               covisibility;          //!< KF -> shared MapPoint count.
    std::vector<std::pair<KeyFrame*, int>> ordered_covisibility;  //!< Same, sorted descending.

    KeyFrame* parent = nullptr;  //!< Spanning-tree parent (used by pose-graph in M3).
    Mat       global_desc;       //!< Place-recognition descriptor (filled by CosPlace in M3).
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_TYPES_HPP
