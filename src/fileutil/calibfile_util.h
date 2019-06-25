
#ifndef YAMLFILE_UTIL_H_
#define YAMLFILE_UTIL_H_

// STL
#include <string>
#include <vector>

// Eigen
#include <Eigen/Core>

namespace fileutil {

void GenerateCailbfileInYaml(const std::string& path,
                             const std::string& camera_name,
                             const std::string& lensmodel,
                             const Eigen::Vector2i& size,
                             const Eigen::Matrix3d& K,
                             const Eigen::Matrix3d& new_K,
                             const Eigen::VectorXd& dist_coeffs);

void ReadCalibfileInYaml(const std::string& path, std::string& camera_name,
                         std::string& lensmodel, Eigen::Vector2i& size,
                         Eigen::Matrix3d& K, Eigen::Matrix3d& new_K,
                         Eigen::VectorXd& dist_coeffs);

}  // namespace fileutil

#endif  // YAMLFILE_UTIL_H_
