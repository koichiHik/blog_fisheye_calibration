
// Standard
#include <iostream>
#include <string>

// Gflag
#include <gflags/gflags.h>

// Glog
#include <glog/logging.h>

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Original
#include <calibration/corner_detection_util.h>
#include <calibration/fisheye.h>
#include <fileutil/calibfile_util.h>
#include <fileutil/filesystem_util.h>

DEFINE_string(calib_picture_dir_path, "./",
              "Path to the directory containing the photos.");
DEFINE_string(calib_file_path, "", "");
DEFINE_string(camera_name, "", "");
DEFINE_string(lens_model, "PINHOLE",
              "Lens model to be calibrated. Either \"PINHOLE\" or \"FISHEYE\"");
DEFINE_int32(board_height, 2, "Height of the calibration board in mm");
DEFINE_int32(board_width, 2, "Width of the calibration board in mm");
DEFINE_double(square_size, 32.5, "Length of rectangle in mm.");
DEFINE_double(aspect_ratio, 1.0,
              "Aspect ratio of rectangle in checker board pattern.");
DEFINE_double(resize_factor_to_show, 0.25, "Resize factor for visualization.");
DEFINE_bool(show_undistorted_image, false, "");

using namespace std;
using namespace calibration;

namespace {

struct CalibData {
  string file_path;
  cv::Mat color_image;
  cv::Mat gray_image;
  vector<cv::Point2f> image_points;
  cv::Vec3d rot_vec, trans_vec;
  double rep_err;
};

struct CalibDataSets {
  LensModel lens_model;
  cv::Size board_dim, image_size;
  // cv::Mat K, new_K, dist_coeff;
  cv::Matx33d K, new_K;
  std::vector<double> dist_coeff;
  double rep_avg_err, square_length;
  vector<cv::Point3f> object_points_in_board_coord;
  vector<CalibData> data;
};

cv::TermCriteria CreateTermCriteria();

void LoadImages(const vector<string> &file_paths, vector<cv::Mat> &color_images,
                vector<cv::Mat> &gray_images) {
  color_images.clear();
  gray_images.clear();
  for (const auto &path : file_paths) {
    cv::Mat color_img = cv::imread(path, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat gray_img = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

    CHECK(color_img.data != NULL) << "Image can not be read" << endl << path;
    color_images.push_back(color_img);
    gray_images.push_back(gray_img);
  }
}

void LoadCorners(const vector<string> &file_paths,
                 vector<vector<cv::Point2f>> &corners_for_files) {
  corners_for_files.clear();
  corners_for_files.reserve(file_paths.size());
  for (const auto &path : file_paths) {
    vector<cv::Point2f> corners;
    calibration::LoadDetectPointsFromFile(path, corners);
    corners_for_files.push_back(corners);
  }
}

void LoadDataIntoCalibDataSets(const string &dir_path,
                               CalibDataSets &calib_sets) {
  vector<string> jpeg_list, txt_list;
  fileutil::RaiseAllFilesInDirectory(dir_path, jpeg_list,
                                     vector<string>{".jpg"});
  fileutil::RaiseAllFilesInDirectory(dir_path, txt_list,
                                     vector<string>{".txt"});

  CHECK(jpeg_list.size() == txt_list.size())
      << "Number of image and result files are different. Please check the "
         "input directory.";

  vector<cv::Mat> color_images, gray_images;
  vector<vector<cv::Point2f>> corners_for_files;
  LoadImages(jpeg_list, color_images, gray_images);
  LoadCorners(txt_list, corners_for_files);
  CHECK_GT(color_images.size(), 0) << "No image files are loaded.";

  calib_sets.data.resize(jpeg_list.size());
  calib_sets.image_size = cv::Size(color_images[0].cols, color_images[0].rows);

  for (size_t idx = 0; idx < color_images.size(); idx++) {
    calib_sets.data[idx].file_path = jpeg_list[idx];
    calib_sets.data[idx].color_image = color_images[idx];
    calib_sets.data[idx].gray_image = gray_images[idx];
    calib_sets.data[idx].image_points = corners_for_files[idx];

    CHECK((calib_sets.image_size.width == color_images[idx].cols &&
           calib_sets.image_size.height == color_images[idx].rows))
        << "Dimensions of all images have to be exactly same." << endl
        << "Path : " << jpeg_list[idx] << endl
        << "[Supposed] Cols : " << calib_sets.image_size.width
        << ", Rows : " << calib_sets.image_size.height
        << ", [Actual] Cols : " << color_images[idx].cols
        << ", Rows : " << color_images[idx].rows;
  }
}

vector<cv::Point3f> CalculateRelativePositionOnCalibrationBoard(
    const cv::Size &board_size, const float square_size) {
  vector<cv::Point3f> relative_position_in_board;

  for (int i = 0; i < board_size.height; i++) {
    for (int j = 0; j < board_size.width; j++) {
      relative_position_in_board.push_back(
          cv::Point3f(square_size * j, square_size * i, 0));
    }
  }

  return relative_position_in_board;
}

bool Calibrate(CalibDataSets &calib_data_sets) {
  vector<cv::Vec3d> trans_vecs, rot_vecs;
  vector<vector<cv::Point2f>> detected_points_on_images;
  vector<vector<cv::Point3f>> object_points_on_boards;
  cv::Matx33d K, new_K;
  std::vector<double> dist_coeffs;

  cv::Size image_size = calib_data_sets.image_size;
  for (const auto &calib_data : calib_data_sets.data) {
    detected_points_on_images.push_back(calib_data.image_points);
    object_points_on_boards.push_back(
        calib_data_sets.object_points_in_board_coord);
  }

  if (calib_data_sets.lens_model == LensModel::PINHOLE) {
    int flag = 0;
    // flag = cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4;
    flag = cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_PRINCIPAL_POINT;
    cv::calibrateCamera(object_points_on_boards, detected_points_on_images,
                        image_size, K, dist_coeffs, rot_vecs, trans_vecs, flag,
                        CreateTermCriteria());
    new_K = cv::getOptimalNewCameraMatrix(K, dist_coeffs, image_size, 1.0,
                                          image_size, nullptr, true);
  } else if (calib_data_sets.lens_model == LensModel::FISHEYE) {
    int flag = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
               cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_CHECK_COND;
    /*cv::fisheye::calibrate(object_points_on_boards, detected_points_on_images,
                           image_size, K, dist_coeffs, rot_vecs, trans_vecs,
                           flag, CreateTermCriteria());*/

    CalibrateFisheye(object_points_on_boards, detected_points_on_images,
                     image_size, flag, CreateTermCriteria(), K, dist_coeffs,
                     rot_vecs, trans_vecs);

    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, dist_coeffs, image_size, cv::Mat::eye(3, 3, CV_32FC1), new_K, 1.0,
        calib_data_sets.image_size);

    LOG(INFO) << "Resultant New K Matrix : " << new_K;
  } else {
    LOG(FATAL) << "The specified lens model is not supported." << endl;
  }

  // Update CalibDataSets.
  calib_data_sets.K = K;
  calib_data_sets.new_K = new_K;
  calib_data_sets.dist_coeff = dist_coeffs;
  for (size_t i = 0; i < calib_data_sets.data.size(); i++) {
    calib_data_sets.data[i].rot_vec = rot_vecs[i];
    calib_data_sets.data[i].trans_vec = trans_vecs[i];
  }

  return cv::checkRange(K) && cv::checkRange(dist_coeffs);
}

void ComputeAverageReprojectionError(CalibDataSets &calib_data_sets) {
  int total_points = 0;
  double total_errs = 0.0;

  for (auto calib_set : calib_data_sets.data) {
    if (calib_set.image_points.size() == 0) {
      continue;
    }

    vector<cv::Point2f> projected_points;

    if (calib_data_sets.lens_model == LensModel::PINHOLE) {
      cv::projectPoints(calib_data_sets.object_points_in_board_coord,
                        calib_set.rot_vec, calib_set.trans_vec,
                        calib_data_sets.K, calib_data_sets.dist_coeff,
                        projected_points);
    } else if (calib_data_sets.lens_model == LensModel::FISHEYE) {
      cv::fisheye::projectPoints(calib_data_sets.object_points_in_board_coord,
                                 projected_points, calib_set.rot_vec,
                                 calib_set.trans_vec, calib_data_sets.K,
                                 calib_data_sets.dist_coeff);
    } else {
      LOG(ERROR) << "Invalid Lens Model!";
    }

    double err =
        cv::norm(calib_set.image_points, projected_points, cv::NORM_L2);

    int point_num = projected_points.size();
    calib_set.rep_err = std::sqrt(err * err / point_num);
    total_points += point_num;
    total_errs += err * err;
  }

  calib_data_sets.rep_avg_err = std::sqrt(total_errs / total_points);
}

void ShowUncalibratedImages(CalibDataSets &calib_data_sets) {
  for (size_t idx = 0; idx < calib_data_sets.data.size(); idx++) {
    CalibData &calib_data = calib_data_sets.data[idx];

    cv::Mat undist_img_resized, undist_img_opt_resized;
    {
      cv::Mat undist_img, undist_img_opt;
      cv::Mat map1, map2, map1_opt, map2_opt;
      if (FLAGS_lens_model == "PINHOLE") {
        cv::initUndistortRectifyMap(
            calib_data_sets.K, calib_data_sets.dist_coeff,
            cv::Mat::eye(3, 3, CV_32FC1), calib_data_sets.K,
            calib_data_sets.image_size, CV_32FC1, map1, map2);
        cv::initUndistortRectifyMap(
            calib_data_sets.K, calib_data_sets.dist_coeff,
            cv::Mat::eye(3, 3, CV_32FC1), calib_data_sets.new_K,
            calib_data_sets.image_size, CV_32FC1, map1_opt, map2_opt);

      } else if (FLAGS_lens_model == "FISHEYE") {
        cv::fisheye::initUndistortRectifyMap(
            calib_data_sets.K, calib_data_sets.dist_coeff,
            cv::Mat::eye(3, 3, CV_32FC1), calib_data_sets.K,
            calib_data_sets.image_size, CV_32FC1, map1, map2);
        cv::fisheye::initUndistortRectifyMap(
            calib_data_sets.K, calib_data_sets.dist_coeff,
            cv::Mat::eye(3, 3, CV_32FC1), calib_data_sets.new_K,
            calib_data_sets.image_size, CV_32FC1, map1_opt, map2_opt);
      }

      cv::remap(calib_data.color_image, undist_img, map1, map2,
                CV_INTER_LINEAR);
      cv::resize(undist_img, undist_img_resized, cv::Size(),
                 FLAGS_resize_factor_to_show, FLAGS_resize_factor_to_show);

      cv::remap(calib_data.color_image, undist_img_opt, map1_opt, map2_opt,
                CV_INTER_LINEAR);
      cv::resize(undist_img_opt, undist_img_opt_resized, cv::Size(),
                 FLAGS_resize_factor_to_show, FLAGS_resize_factor_to_show);
    }

    cv::imshow("Undistorted Image", undist_img_resized);
    // cv::imshow("Optimal Undistorted Image", undist_img_opt_resized);
    cv::waitKey(0);
  }
}

cv::TermCriteria CreateTermCriteria() {
  cv::TermCriteria term_criteria;
  {
    term_criteria.type = cv::TermCriteria::COUNT;
    term_criteria.type += cv::TermCriteria::EPS;
    term_criteria.maxCount = 100000;
    term_criteria.epsilon = DBL_EPSILON;
  }
  return term_criteria;
}

void OutputCalibResultToYamlFile(const std::string &path,
                                 const std::string &camera_name,
                                 const std::string &lensmodel,
                                 const CalibDataSets &calib_data_sets) {
  cv::Matx33d cvK(calib_data_sets.K);
  Eigen::Matrix3d K(cvK.val);

  Eigen::Vector2i size(calib_data_sets.image_size.width,
                       calib_data_sets.image_size.height);

  Eigen::VectorXd dist_coeff =
      Eigen::VectorXd::Map(&(*calib_data_sets.dist_coeff.begin()),
                           calib_data_sets.dist_coeff.size());

  fileutil::GenerateCailbfileInYaml(path, FLAGS_camera_name, lensmodel, size, K,
                                    K, dist_coeff);
}

}  // namespace

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Loading images in the specified directory...";
  CalibDataSets calib_data_sets;
  calib_data_sets.lens_model = ConvertLensModelFromString(FLAGS_lens_model);
  calib_data_sets.square_length = FLAGS_square_size;
  calib_data_sets.board_dim = cv::Size(FLAGS_board_width, FLAGS_board_height);
  LoadDataIntoCalibDataSets(FLAGS_calib_picture_dir_path, calib_data_sets);

  // Find corners and calculate corresponding 3d points.
  calib_data_sets.object_points_in_board_coord =
      CalculateRelativePositionOnCalibrationBoard(calib_data_sets.board_dim,
                                                  FLAGS_square_size);

  // Execute calibration.
  cv::Mat K, dist_coeffs;
  vector<float> rep_err;
  LOG(INFO) << "Start calibration...";
  CHECK(Calibrate(calib_data_sets)) << "Calibration failed...";

  // Calculate reprojection error
  LOG(INFO) << "Compute statistics...";
  ComputeAverageReprojectionError(calib_data_sets);

  LOG(INFO) << "Summary of calibration..." << endl
            << "K matrix : " << endl
            << calib_data_sets.K << endl
            << "Distortion coefficient : "
            << cv::Mat(calib_data_sets.dist_coeff) << endl
            << "Average reprojection error : " << calib_data_sets.rep_avg_err;

  std::string path = FLAGS_calib_file_path == ""
                         ? FLAGS_calib_picture_dir_path + "/calib.yml"
                         : FLAGS_calib_file_path;

  // Output Result to Yaml File.
  OutputCalibResultToYamlFile(path, FLAGS_camera_name, FLAGS_lens_model,
                              calib_data_sets);

  if (FLAGS_show_undistorted_image) {
    ShowUncalibratedImages(calib_data_sets);
  }

  return 0;
}