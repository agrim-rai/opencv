// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_MAP_HPP
#define OPENCV_SLAM_MAP_HPP

#include "opencv2/slam/types.hpp"

#include <set>
#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Thread-safe container for all persistent SLAM state.

Owns (allocates and destroys) every @ref KeyFrame and @ref MapPoint object.
Raw pointers returned by @ref addKeyframe / @ref addMapPoint remain valid until
the object is removed via @ref removeMapPoint / @ref clear.

Map is pure storage — no algorithm logic lives here.  It is designed to be
shared between the odometry front-end (M1), local-mapping thread (M2), and
loop-closure / relocalization modules (M3).
*/
class CV_EXPORTS Map
{
public:
    Map();
    ~Map();

    Map(const Map&)            = delete;
    Map& operator=(const Map&) = delete;

    // --- Keyframes ---------------------------------------------------------

    /** Takes ownership of @p kf, assigns a fresh id if kf->id < 0.
        Returns @p kf (same pointer, now owned by the map). */
    KeyFrame* addKeyframe(KeyFrame* kf);

    /** Returns the keyframe with @p id, or nullptr if not found. */
    KeyFrame* getKeyframe(int id) const;

    const std::set<KeyFrame*>& keyframes() const;
    int numKeyframes() const;

    // --- Map points --------------------------------------------------------

    /** Takes ownership of @p mp, assigns a fresh id if mp->id < 0.
        Returns @p mp (same pointer, now owned by the map). */
    MapPoint* addMapPoint(MapPoint* mp);

    /** Returns the map point with @p id, or nullptr if not found. */
    MapPoint* getMapPoint(int id) const;

    const std::set<MapPoint*>& mapPoints() const;
    int numMapPoints() const;

    /** Wires a 2D-3D correspondence: sets kf->mappoints[kp_idx] = mp and
        mp->observations[kf] = kp_idx.  A no-op if kp_idx is already occupied. */
    void addObservation(KeyFrame* kf, size_t kp_idx, MapPoint* mp);

    /** Removes the single observation link between @p kf and @p mp. */
    void removeObservation(KeyFrame* kf, MapPoint* mp);

    /** Erases @p mp from the map, unlinks it from all keyframes, and deletes it. */
    void removeMapPoint(MapPoint* mp);

    // --- Reference / current keyframes ------------------------------------

    void      setRefKeyframe    (KeyFrame* kf);
    KeyFrame* getRefKeyframe    () const;

    void      setCurrentKeyframe(KeyFrame* kf);
    KeyFrame* getCurrentKeyframe() const;

    // --- Trajectory (ordered per-frame poses stored for output) -----------

    void appendPose(const Matx44d& T_cw);
    const std::vector<Matx44d>& trajectory() const;

    // --- Lifecycle ---------------------------------------------------------

    void clear();

private:
    struct Impl;
    Ptr<Impl> impl_;
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_MAP_HPP
