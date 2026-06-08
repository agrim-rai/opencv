// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_FRAME_HPP
#define OPENCV_SLAM_FRAME_HPP

#include "../precomp.hpp"

namespace cv {
namespace slam {

/** @brief Per-frame scratch pad.  Created for every incoming image and discarded
after the tracking step completes.  Never stored in the Map.

Tracks which keypoints have been matched to MapPoints (mappoints[] parallel
array) and which have been flagged as outliers by pose optimisation.  The
spatial grid enables fast radius-based keypoint lookups used by the
motion-model and local-map tracking stages.
*/
struct Frame
{
    Mat                    image;        //!< Greyscale copy kept for optical-flow fallback.
    std::vector<KeyPoint>  keypoints;
    Mat                    descriptors; //!< NxD matrix; row i = descriptor for keypoints[i].
    std::vector<Point2f>   undist_kpts; //!< Undistorted pixel coords, parallel to keypoints.
    Size                   imageSize;

    Matx44d                pose_cw = Matx44d::eye(); //!< World->camera pose set by tracking.
    std::vector<MapPoint*> mappoints;  //!< Parallel to keypoints; nullptr = unmatched.
    std::vector<bool>      outliers;   //!< Parallel to keypoints; true = rejected by optimiser.

    // ---- Spatial grid for fast radius-search --------------------------------

    static constexpr int GRID_ROWS = 48;
    static constexpr int GRID_COLS = 64;

    std::vector<std::vector<std::vector<size_t>>> grid; // [row][col] -> keypoint indices

    /** Populate grid from undist_kpts.  Call once after undist_kpts is ready. */
    void buildGrid()
    {
        if (imageSize.width <= 0 || imageSize.height <= 0) return;
        grid.assign(GRID_ROWS, std::vector<std::vector<size_t>>(GRID_COLS));
        const float cs = (float)GRID_COLS / (float)imageSize.width;
        const float rs = (float)GRID_ROWS / (float)imageSize.height;
        for (size_t i = 0; i < undist_kpts.size(); ++i)
        {
            int col = std::min(GRID_COLS - 1, std::max(0, (int)(undist_kpts[i].x * cs)));
            int row = std::min(GRID_ROWS - 1, std::max(0, (int)(undist_kpts[i].y * rs)));
            grid[row][col].push_back(i);
        }
    }

    /** Return indices of keypoints whose undistorted position is within
        @p r pixels of (@p x, @p y). */
    std::vector<size_t> getKeypointsInRadius(float x, float y, float r) const
    {
        std::vector<size_t> result;
        if (imageSize.width <= 0 || imageSize.height <= 0 || grid.empty()) return result;
        const float cs = (float)GRID_COLS / (float)imageSize.width;
        const float rs = (float)GRID_ROWS / (float)imageSize.height;
        const int col_min = std::max(0,             (int)((x - r) * cs));
        const int col_max = std::min(GRID_COLS - 1, (int)((x + r) * cs));
        const int row_min = std::max(0,             (int)((y - r) * rs));
        const int row_max = std::min(GRID_ROWS - 1, (int)((y + r) * rs));
        const float r2 = r * r;
        for (int row = row_min; row <= row_max; ++row)
            for (int col = col_min; col <= col_max; ++col)
                for (size_t idx : grid[row][col])
                {
                    float dx = undist_kpts[idx].x - x;
                    float dy = undist_kpts[idx].y - y;
                    if (dx * dx + dy * dy <= r2)
                        result.push_back(idx);
                }
        return result;
    }
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_FRAME_HPP
