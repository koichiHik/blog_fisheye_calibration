
#ifndef FISHEYE_H_
#define FISHEYE_H_

// OpenCV
#include <opencv2/core.hpp>

double CalibrateFisheye(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const cv::Size& image_size, const int flags,
    const cv::TermCriteria criteria, cv::Matx33d& K, std::vector<double>& D,
    std::vector<cv::Vec3d>& board_rots, std::vector<cv::Vec3d>& board_trans);

#endif  // FISHEYE_H_
