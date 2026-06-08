// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Minimal driver for cv::slam::VisualOdometry.
//
// Build:
//   cmake -DBUILD_EXAMPLES=ON ...   (binary: bin/example_slam_visual_odometry_demo)
//
// Usage:
//   ./example_slam_visual_odometry_demo
//       --aliked=<path/to/aliked.onnx>
//       --lightglue=<path/to/aliked_lightglue.onnx>
//       --images=<path/to/images_dir>
//       --output=<path/to/output_dir>
//       --K=fx,fy,cx,cy
//       [--dist=k1,k2,p1,p2[,k3]]

#include <opencv2/slam.hpp>
#include <opencv2/features.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Parse a comma-separated list of doubles. Returns false on any malformed token
// or if the count does not match `expected` (when `expected` is non-zero).
bool parseCsvDoubles(const std::string& s, std::vector<double>& out, size_t expected = 0)
{
    out.clear();
    if (s.empty()) return expected == 0;

    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (tok.empty()) return false;
        try { out.push_back(std::stod(tok)); }
        catch (const std::exception&) { return false; }
    }
    return expected == 0 || out.size() == expected;
}

cv::Mat makeK(const std::vector<double>& v)
{
    return (cv::Mat_<double>(3, 3) << v[0], 0,    v[2],
                                      0,    v[1], v[3],
                                      0,    0,    1);
}

} // namespace

int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |       | print this message }"
        "{aliked         |       | path to ALIKED ONNX model (required) }"
        "{lightglue      |       | path to LightGlue ONNX model (required) }"
        "{images         |       | folder of input images (required) }"
        "{output         | vo_out| folder to write trajectory/map/log }"
        "{K              |       | intrinsics as fx,fy,cx,cy (required) }"
        "{dist           |       | distortion coefficients k1,k2,p1,p2[,k3] (optional) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("cv::slam::VisualOdometry minimal demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    const std::string aliked_path    = parser.get<std::string>("aliked");
    const std::string lightglue_path = parser.get<std::string>("lightglue");
    const std::string images_dir     = parser.get<std::string>("images");
    const std::string output_dir     = parser.get<std::string>("output");
    const std::string K_str          = parser.get<std::string>("K");
    const std::string dist_str       = parser.get<std::string>("dist");

    if (aliked_path.empty() || lightglue_path.empty() || images_dir.empty() || K_str.empty())
    {
        std::cerr << "ERROR: --aliked, --lightglue, --images and --K are required\n\n";
        parser.printMessage();
        return 1;
    }

    std::vector<double> K_vals;
    if (!parseCsvDoubles(K_str, K_vals, 4))
    {
        std::cerr << "ERROR: --K must be four comma-separated numbers: fx,fy,cx,cy\n";
        return 1;
    }
    const cv::Mat K = makeK(K_vals);

    std::vector<double> dist_vals;
    if (!parseCsvDoubles(dist_str, dist_vals))
    {
        std::cerr << "ERROR: --dist must be a comma-separated list of numbers\n";
        return 1;
    }
    const cv::Mat dist = dist_vals.empty()
                         ? cv::Mat()
                         : cv::Mat(dist_vals, true).reshape(1, (int)dist_vals.size()).clone();

    try
    {
        cv::Ptr<cv::Feature2D>         detector = cv::ALIKED::create(aliked_path);
        cv::Ptr<cv::DescriptorMatcher> matcher  = cv::LightGlueMatcher::create(lightglue_path);

        std::cout << "[demo] images: " << images_dir << "\n";
        std::cout << "[demo] output: " << output_dir << "\n";
        std::cout << "[demo] K:      " << K_str      << "\n";
        if (!dist_vals.empty())
            std::cout << "[demo] dist:   " << dist_str << "\n";

        cv::Ptr<cv::slam::VisualOdometry> vo =
            cv::slam::VisualOdometry::create(
                detector, matcher,
                images_dir, output_dir,
                K, dist);

        const bool ok = vo->run();

        const cv::slam::Map& m = vo->getMap();
        std::cout << "[demo] done."
                  << "  run="         << (ok ? "ok" : "failed")
                  << "  keyframes="   << m.numKeyframes()
                  << "  map_points="  << m.numMapPoints()
                  << "  trajectory="  << vo->getTrajectory().size()
                  << "\n";
        return ok ? 0 : 1;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "ERROR (cv::Exception): " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
