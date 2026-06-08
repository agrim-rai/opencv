// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <mutex>
#include <set>
#include <unordered_map>

namespace cv {
namespace slam {

struct Map::Impl
{
    std::set<KeyFrame*>               keyframes;
    std::set<MapPoint*>               mapPoints;
    std::unordered_map<int,KeyFrame*> kf_index;
    std::unordered_map<int,MapPoint*> mp_index;

    KeyFrame* ref_kf     = nullptr;
    KeyFrame* current_kf = nullptr;

    std::vector<Matx44d> trajectory;
    std::mutex           mutex;

    int next_kf_id = 0;
    int next_mp_id = 0;
};

Map::Map()  : impl_(makePtr<Impl>()) {}

Map::~Map()
{
    for (KeyFrame* kf : impl_->keyframes) delete kf;
    for (MapPoint* mp : impl_->mapPoints) delete mp;
}

// ---------------------------------------------------------------------------
// Keyframes
// ---------------------------------------------------------------------------

KeyFrame* Map::addKeyframe(KeyFrame* kf)
{
    CV_Assert(kf);
    if (kf->id < 0)
        kf->id = impl_->next_kf_id++;
    else if (kf->id >= impl_->next_kf_id)
        impl_->next_kf_id = kf->id + 1;
    impl_->keyframes.insert(kf);
    impl_->kf_index[kf->id] = kf;
    return kf;
}

KeyFrame* Map::getKeyframe(int id) const
{
    auto it = impl_->kf_index.find(id);
    return (it != impl_->kf_index.end()) ? it->second : nullptr;
}

const std::set<KeyFrame*>& Map::keyframes() const { return impl_->keyframes; }
int Map::numKeyframes() const { return (int)impl_->keyframes.size(); }

// ---------------------------------------------------------------------------
// Map points
// ---------------------------------------------------------------------------

MapPoint* Map::addMapPoint(MapPoint* mp)
{
    CV_Assert(mp);
    if (mp->id < 0)
        mp->id = impl_->next_mp_id++;
    else if (mp->id >= impl_->next_mp_id)
        impl_->next_mp_id = mp->id + 1;
    impl_->mapPoints.insert(mp);
    impl_->mp_index[mp->id] = mp;
    return mp;
}

MapPoint* Map::getMapPoint(int id) const
{
    auto it = impl_->mp_index.find(id);
    return (it != impl_->mp_index.end()) ? it->second : nullptr;
}

const std::set<MapPoint*>& Map::mapPoints() const { return impl_->mapPoints; }
int Map::numMapPoints() const { return (int)impl_->mapPoints.size(); }

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------

void Map::addObservation(KeyFrame* kf, size_t kp_idx, MapPoint* mp)
{
    CV_Assert(kf && mp);
    CV_Assert(kp_idx < kf->mappoints.size());
    if (kf->mappoints[kp_idx] != nullptr) return; // slot already occupied
    kf->mappoints[kp_idx] = mp;
    mp->observations[kf]  = kp_idx;
}

void Map::removeObservation(KeyFrame* kf, MapPoint* mp)
{
    if (!kf || !mp) return;
    auto it = mp->observations.find(kf);
    if (it == mp->observations.end()) return;
    size_t kp_idx = it->second;
    if (kp_idx < kf->mappoints.size())
        kf->mappoints[kp_idx] = nullptr;
    mp->observations.erase(it);
}

void Map::removeMapPoint(MapPoint* mp)
{
    if (!mp) return;
    for (auto& [kf, kp_idx] : mp->observations)
        if (kp_idx < kf->mappoints.size())
            kf->mappoints[kp_idx] = nullptr;
    impl_->mapPoints.erase(mp);
    impl_->mp_index.erase(mp->id);
    delete mp;
}

// ---------------------------------------------------------------------------
// Reference / current keyframes
// ---------------------------------------------------------------------------

void      Map::setRefKeyframe    (KeyFrame* kf) { impl_->ref_kf     = kf; }
KeyFrame* Map::getRefKeyframe    () const       { return impl_->ref_kf;   }

void      Map::setCurrentKeyframe(KeyFrame* kf) { impl_->current_kf = kf; }
KeyFrame* Map::getCurrentKeyframe() const       { return impl_->current_kf; }

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

void Map::appendPose(const Matx44d& T_cw)          { impl_->trajectory.push_back(T_cw); }
const std::vector<Matx44d>& Map::trajectory() const { return impl_->trajectory; }

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void Map::clear()
{
    for (KeyFrame* kf : impl_->keyframes) delete kf;
    for (MapPoint* mp : impl_->mapPoints) delete mp;
    impl_->keyframes.clear();
    impl_->mapPoints.clear();
    impl_->kf_index.clear();
    impl_->mp_index.clear();
    impl_->ref_kf     = nullptr;
    impl_->current_kf = nullptr;
    impl_->trajectory.clear();
    impl_->next_kf_id = 0;
    impl_->next_mp_id = 0;
}

}} // namespace cv::slam
