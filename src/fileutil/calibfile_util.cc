
// Self Header
#include <fileutil/calibfile_util.h>

// STL
#include <fstream>
#include <map>

// Glog
#include <glog/logging.h>

// yaml-cpp
#include <yaml-cpp/yaml.h>

namespace fileutil {

void GenerateCailbfileInYaml(const std::string& path,
                             const std::string& camera_name,
                             const std::string& lensmodel,
                             const Eigen::Vector2i& size,
                             const Eigen::Matrix3d& K,
                             const Eigen::Matrix3d& new_K,
                             const Eigen::VectorXd& dist_coeffs) {
  YAML::Node cam;
  {
    cam["image_width"] = size(0);
    cam["image_height"] = size(1);
    cam["camera_name"] = camera_name;

    // Camera Matrix
    cam["camera_matrix"]["rows"] = 3;
    cam["camera_matrix"]["cols"] = 3;
    cam["camera_matrix"]["data"] =
        std::vector<float>(K.data(), K.data() + K.rows() * K.cols());
    cam["camera_matrix"]["data"].SetStyle(YAML::EmitterStyle::Flow);

    // Distortion Model
    cam["distortion_model"] = lensmodel;

    // Distortion Coefficient
    cam["distortion_coefficients"]["rows"] = 1;
    cam["distortion_coefficients"]["cols"] = 4;
    cam["distortion_coefficients"]["data"] = std::vector<float>(
        dist_coeffs.data(), dist_coeffs.data() + dist_coeffs.size());
    cam["distortion_coefficients"]["data"].SetStyle(YAML::EmitterStyle::Flow);

    // New Camera Matrix
    cam["new_camera_matrix"]["rows"] = 3;
    cam["new_camera_matrix"]["cols"] = 3;
    cam["new_camera_matrix"]["data"] = std::vector<float>(
        new_K.data(), new_K.data() + new_K.rows() * new_K.cols());
    cam["new_camera_matrix"]["data"].SetStyle(YAML::EmitterStyle::Flow);
  }

  // Distortion Coefficient
  YAML::Emitter out;
  out << cam;

  std::ofstream file(path);
  file << out.c_str();
  file.close();
}

void ReadCalibfileInYaml(const std::string& path, std::string& camera_name,
                         std::string& lensmodel, Eigen::Vector2i& size,
                         Eigen::Matrix3d& K, Eigen::Matrix3d& new_K,
                         Eigen::VectorXd& dist_coeffs) {
  YAML::Node cam = YAML::LoadFile(path);

  CHECK(!cam.IsNull()) << "Specified yaml file can not be read.";

  // Size
  {
    size(0) = cam["image_width"].as<int>();
    size(1) = cam["image_height"].as<int>();
    CHECK(size(0) > 0 && size(1) > 0) << "The image size has to be defined.";
  }

  // Camera Model
  {
    lensmodel = cam["distortion_model"].as<std::string>();
    CHECK(lensmodel == std::string("FISHEYE") ||
          lensmodel == std::string("PINHOLE"))
        << "Currently only FISHEYE or PINHOLE is supported.";
  }

  // Camera Matrix.
  {
    int rows = cam["camera_matrix"]["rows"].as<int>();
    int cols = cam["camera_matrix"]["cols"].as<int>();
    CHECK(rows == 3 && cols == 3) << "Size of the camera matrix has to be 3x3";

    std::vector<double> cam_mat =
        cam["camera_matrix"]["data"].as<std::vector<double>>();
    K = Eigen::Matrix3d(cam_mat.data());
  }

  // New Camera Matrix
  {
    int rows = cam["new_camera_matrix"]["rows"].as<int>();
    int cols = cam["new_camera_matrix"]["cols"].as<int>();
    CHECK(rows == 3 && cols == 3) << "Size of the camera matrix has to be 3x3";

    std::vector<double> cam_mat =
        cam["new_camera_matrix"]["data"].as<std::vector<double>>();
    new_K = Eigen::Matrix3d(cam_mat.data());
  }

  // Distortion Coefficient
  {
    std::vector<double> tmp_dist_coeffs =
        cam["distortion_coefficients"]["data"].as<std::vector<double>>();
    dist_coeffs =
        Eigen::VectorXd::Map(tmp_dist_coeffs.data(), tmp_dist_coeffs.size());
  }
}

}  // namespace fileutil

#if SAMPLE_CODE

int main() {
  std::string path = "/home/koichi/workspace/tools/tools/calib.yml";

  {
    Eigen::Vector2i size;
    size(0) = 1400;
    size(1) = 1500;

    Eigen::Matrix3d K;
    K(0, 0) = 700.11;
    K(1, 1) = 750.12;
    K(2, 2) = 1.0;
    K(0, 2) = 700.13;
    K(1, 2) = 750.14;

    Eigen::Matrix3d new_K = K;

    Eigen::VectorXd dist_coeffs(4);
    dist_coeffs(0) = 1.00;
    dist_coeffs(1) = 1.10;
    dist_coeffs(2) = 1.20;
    dist_coeffs(3) = 1.30;

    fileutil::GenerateCailbfileInYaml(path, size, K, new_K, dist_coeffs);
  }

  {
    Eigen::Vector2i size;
    Eigen::Matrix3d K, new_K;
    Eigen::VectorXd dist_coeffs;
    fileutil::ReadCalibfileInYaml(path, size, K, new_K, dist_coeffs);

    LOG(INFO) << "Size : " << std::endl << size << std::endl;
    LOG(INFO) << "K : " << std::endl << K << std::endl;
    LOG(INFO) << "new_K : " << std::endl << new_K << std::endl;
    LOG(INFO) << "dist_coeffs : " << std::endl << dist_coeffs << std::endl;
  }

  return 0;
}

#endif