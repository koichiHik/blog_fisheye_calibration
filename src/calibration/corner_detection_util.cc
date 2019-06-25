
// Self Header
#include <calibration/corner_detection_util.h>

// System
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

// Boost
#include <boost/algorithm/string.hpp>

// Glog
#include <glog/logging.h>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace calibration {

BoardType ConvertBoardTypeFromString(std::string& board_type_string) {
  if (board_type_string == "CHECKER") {
    return BoardType::CHECKER;
  } else if (board_type_string == "CIRCLES_GRID") {
    return BoardType::CIRCLES_GRID;
  } else {
    LOG(ERROR) << "Board Type is not supported.";
  }
  return BoardType::INVALID;
}

LensModel ConvertLensModelFromString(std::string& board_type_string) {
  if (board_type_string == "PINHOLE") {
    return LensModel::PINHOLE;
  } else if (board_type_string == "FISHEYE") {
    return LensModel::FISHEYE;
  } else {
    LOG(ERROR) << "Lens Model is not supported.";
  }
  return LensModel::INVALID;
}

void DetectPoints(const cv::Mat& color_img, const cv::Size& pattern_size,
                  const BoardType type,
                  std::vector<cv::Point2f>& detected_points,
                  cv::Ptr<cv::SimpleBlobDetector> p_detector) {
  detected_points.clear();
  if (type == BoardType::CHECKER) {
    int flag = CALIB_CB_ADAPTIVE_THRESH;
    bool result = cv::findChessboardCorners(color_img, pattern_size,
                                            detected_points, flag);

    if (result) {
      cv::TermCriteria criteria;
      {
        criteria.type = cv::TermCriteria::EPS + cv::TermCriteria::COUNT;
        criteria.maxCount = 10000000;
        criteria.epsilon = DBL_EPSILON;
      }
      cv::Mat gray_img;
      cv::cvtColor(color_img, gray_img, COLOR_BGR2GRAY);
      cv::cornerSubPix(gray_img, detected_points, cv::Size(11, 11),
                       cv::Size(-1, -1), criteria);
    }

  } else if (type == BoardType::CIRCLES_GRID) {
    LOG(INFO) << "Generate SimpleBlobDetector...";
    std::vector<cv::KeyPoint> detected_key_points;
    // detector->detect(color_img, detected_key_points, cv::noArray());
    bool result = cv::findCirclesGrid(
        color_img, pattern_size, detected_points,
        cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING, p_detector);
    std::transform(detected_key_points.begin(), detected_key_points.end(),
                   detected_points.begin(),
                   [](const cv::KeyPoint& key_point) -> cv::Point2f {
                     cv::Point2f pnt(key_point.pt);
                     return pnt;
                   });
  } else {
    LOG(FATAL) << "The specified pattern is no supported.";
  }
}

void SaveResult(const string& image_path,
                const vector<cv::Point2f>& detected_points,
                cv::Size pattern_size) {
  size_t n = image_path.rfind(".jpg");
  std::string save_filename = image_path;
  save_filename.replace(n, 4, ".txt");
  std::fstream outfile(save_filename, std::ios::out);

  if (detected_points.size() == pattern_size.height * pattern_size.width) {
    for (const auto& pnt : detected_points) {
      outfile << pnt.x << "," << pnt.y << std::endl;
    }
  }
}

void LoadDetectPointsFromFile(const std::string& path,
                              std::vector<cv::Point2f>& detected_points) {
  detected_points.clear();
  std::ifstream reading_file(path, std::ios::in);
  while (!reading_file.eof()) {
    std::string line;
    std::getline(reading_file, line);

    vector<string> results;
    boost::split(results, line, [](char c) { return c == ','; });

    if (results.size() != 2) {
      break;
    }
    detected_points.push_back(
        cv::Point2f(std::stod(results[0]), std::stod(results[1])));
  }
}

void DrawResult(const std::vector<cv::Point2f>& detected_points,
                const BoardType type, const cv::Size& pattern_size,
                cv::Mat& color_img) {
  size_t total_points = pattern_size.height * pattern_size.width;
  if (type == BoardType::CHECKER) {
    bool result = detected_points.size() == total_points;
    cv::drawChessboardCorners(color_img, pattern_size, detected_points, result);
  } else if (type == BoardType::CIRCLES_GRID) {
    for (const auto& pnt : detected_points) {
      cv::circle(color_img, pnt, 10, cv::Scalar(0, 0, 255), 3);
    }
  } else {
    LOG(ERROR) << "BoardType is not supported!";
  }
}

void ShowResult(const cv::Mat& color_img,
                const std::vector<cv::Point2f>& detected_points,
                const BoardType type, const cv::Size& pattern_size) {
  cv::Mat drawn = color_img.clone();
  DrawResult(detected_points, type, pattern_size, drawn);
  cv::imshow("Corner Detction", drawn);
  cv::waitKey(0);
}

}  // namespace calibration