// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {
namespace slam {

// Anchor TU for OdometryParams. Keeping the out-of-line default constructor
// here means the in-header member initializers remain the single source of
// truth for default values.
OdometryParams::OdometryParams() = default;

}} // namespace cv::slam
