
#ifndef CORNER_DETECTION_UTIL_H_
#define CORNER_DETECTION_UTIL_H_

// System
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace calibration {

enum class BoardType {
  CHECKER,
  CIRCLES_GRID,
  INVALID,
};

enum class LensModel {
  PINHOLE,
  FISHEYE,
  INVALID,
};

BoardType ConvertBoardTypeFromString(std::string& board_type_string);

LensModel ConvertLensModelFromString(std::string& board_type_string);

void DetectPoints(const cv::Mat& color_img, const cv::Size& pattern_size,
                  const BoardType type,
                  std::vector<cv::Point2f>& detected_points,
                  cv::Ptr<cv::SimpleBlobDetector> p_detector);

void SaveResult(const std::string& image_path,
                const std::vector<cv::Point2f>& detected_points,
                cv::Size pattern_size);

void ShowResult(const cv::Mat& color_img,
                const std::vector<cv::Point2f>& detected_points,
                const BoardType type, const cv::Size& pattern_size);

void LoadDetectPointsFromFile(const std::string& path,
                              std::vector<cv::Point2f>& detected_points);

void DrawResult(const std::vector<cv::Point2f>& detected_points,
                const BoardType type, const cv::Size& pattern_size,
                cv::Mat& color_img);

}  // namespace calibration

#endif  // CORNER_DETECTION_UTIL_H_
