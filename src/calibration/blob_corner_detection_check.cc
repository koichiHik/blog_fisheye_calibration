
// System
#include <algorithm>
#include <fstream>
#include <iostream>

// Boost
#include <boost/algorithm/string.hpp>

// Glog
#include <glog/logging.h>

// Gflags
#include <gflags/gflags.h>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Original
#include <calibration/corner_detection_util.h>
#include <fileutil/filesystem_util.h>

DEFINE_string(calib_picture_dir_path, "./",
              "Path to the directory containing the photos.");

DEFINE_string(pattern, "CHECKER", "");
DEFINE_double(resize_factor, 0.25, "");

DEFINE_bool(file_viewer, false, "");
DEFINE_bool(save_results, false, "");
DEFINE_bool(show_results, false, "");

DEFINE_double(thresholdStep, 10, "");
DEFINE_double(minThreshold, 50, "");
DEFINE_double(maxThreshold, 220, "");
DEFINE_uint64(minRepeatability, 2, "");
DEFINE_double(minDistBetweenBlobs, 10, "");

DEFINE_uint64(checker_pattern_width, 7, "");
DEFINE_uint64(checker_pattern_height, 10, "");

DEFINE_bool(filterByColor, true, "");
DEFINE_int32(blobColor, 0, "");

DEFINE_bool(filterByArea, true, "");
DEFINE_double(minArea, 25, "");
DEFINE_double(maxArea, 5000, "");

DEFINE_bool(filterByCircularity, true, "");
DEFINE_double(minCircularity, 0.8f, "");
DEFINE_double(maxCircularity, std::numeric_limits<float>::max(), "");

DEFINE_bool(filterByInertia, true, "");
DEFINE_double(minInertiaRatio, 0.1f, "");
DEFINE_double(maxInertiaRatio, std::numeric_limits<float>::max(), "");

DEFINE_bool(filterByConvexity, true, "");
DEFINE_double(minConvexity, 0.95, "");
DEFINE_double(maxConvexity, std::numeric_limits<float>::max(), "");

using namespace std;
using namespace cv;
using namespace calibration;

namespace {

cv::SimpleBlobDetector::Params CreateParams() {
  cv::SimpleBlobDetector::Params params;

  params.thresholdStep = FLAGS_thresholdStep;
  params.minThreshold = FLAGS_minThreshold;
  params.maxThreshold = FLAGS_maxThreshold;
  params.minRepeatability = FLAGS_minRepeatability;
  params.minDistBetweenBlobs = FLAGS_minDistBetweenBlobs;

  params.filterByColor = FLAGS_filterByColor;
  params.blobColor = FLAGS_blobColor;

  params.filterByArea = FLAGS_filterByArea;
  params.minArea = FLAGS_minArea;
  params.maxArea = FLAGS_maxArea;

  params.filterByConvexity = FLAGS_filterByConvexity;
  params.minConvexity = FLAGS_minConvexity;
  // params.maxConvexity = FLAGS_maxConvexity;
  params.maxConvexity = numeric_limits<float>::max();

  params.filterByInertia = FLAGS_filterByInertia;
  params.minInertiaRatio = FLAGS_minInertiaRatio;
  // params.maxInertiaRatio = FLAGS_maxInertiaRatio;
  params.maxInertiaRatio = numeric_limits<float>::max();

  params.filterByCircularity = FLAGS_filterByCircularity;
  params.minConvexity = FLAGS_minConvexity;
  // params.maxConvexity = FLAGS_maxConvexity;
  params.maxConvexity = numeric_limits<float>::max();

  return params;
}

void DetectPointMode(const std::string &dir_path, const double resize_factor,
                     const cv::Size &pattern_size, const BoardType type) {
  LOG(INFO) << "Collecting all files in the given directory...";
  std::vector<string> file_list;
  fileutil::RaiseAllFilesInDirectory(dir_path, file_list,
                                     vector<string>{".jpg"});

  cv::Ptr<cv::SimpleBlobDetector> p_blob_detector =
      cv::SimpleBlobDetector::create(CreateParams());

  for (const auto &path : file_list) {
    LOG(INFO) << "Detection against : " << path;
    cv::Mat color_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    cv::resize(color_img, color_img, cv::Size(), resize_factor, resize_factor);
    vector<cv::Point2f> detected_points;

    // Detect points sequentially.
    DetectPoints(color_img, pattern_size, type, detected_points,
                 p_blob_detector);

    // Show results.
    if (FLAGS_show_results) {
      ShowResult(color_img, detected_points, type, pattern_size);
    }

    // Save results.
    if (FLAGS_save_results) {
      SaveResult(path, detected_points, pattern_size);
    }
  }
}

void FileViewerMode(const std::string &dir_path, const cv::Size &pattern_size,
                    const BoardType type, const double resize_factor) {
  std::vector<string> image_path_list, saved_txt_list;
  fileutil::RaiseAllFilesInDirectory(FLAGS_calib_picture_dir_path,
                                     image_path_list, vector<string>{".jpg"});
  fileutil::RaiseAllFilesInDirectory(FLAGS_calib_picture_dir_path,
                                     saved_txt_list, vector<string>{".txt"});

  CHECK(image_path_list.size() == saved_txt_list.size())
      << "Please confirm one to one correspondence between image file and "
         "result txt file.";
  std::sort(image_path_list.begin(), image_path_list.end());
  std::sort(saved_txt_list.begin(), saved_txt_list.end());

  for (size_t idx = 0; idx < saved_txt_list.size(); idx++) {
    LOG(INFO) << "Showing : " << saved_txt_list[idx];
    std::vector<cv::Point2f> detected_points;
    LoadDetectPointsFromFile(saved_txt_list[idx], detected_points);
    cv::Mat color_image = cv::imread(image_path_list[idx], CV_LOAD_IMAGE_COLOR);
    DrawResult(detected_points, type, pattern_size, color_image);
    cv::resize(color_image, color_image, cv::Size(), resize_factor,
               resize_factor);
    cv::imshow("Corner Detction", color_image);
    // cv::imwrite("result_" + std::to_string(idx) + "_.jpg", color_image);
    cv::waitKey(0);
  }
}

} // namespace

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  string dir_path = FLAGS_calib_picture_dir_path;
  double resize_factor = FLAGS_resize_factor;
  BoardType type = ConvertBoardTypeFromString(FLAGS_pattern);
  cv::Size pattern_size(FLAGS_checker_pattern_width,
                        FLAGS_checker_pattern_height);
  if (FLAGS_file_viewer) {
    FileViewerMode(FLAGS_calib_picture_dir_path, pattern_size, type,
                   resize_factor);
  } else {
    DetectPointMode(FLAGS_calib_picture_dir_path, resize_factor, pattern_size,
                    type);
  }

  return 0;
}