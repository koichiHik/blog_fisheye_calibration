

#ifndef FILESYSTEM_UTIL_CC_
#define FILESYSTEM_UTIL_CC_

// Self Header
#include <fileutil/filesystem_util.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

namespace fileutil {
namespace {

void RaiseAllFilesInDirectoryInternal(
    const std::string &dirpath, std::vector<std::string> &img_path_list,
    std::vector<std::string> &img_filename_list) {
  namespace fs = boost::filesystem;
  fs::path path(dirpath);
  if (path.is_relative()) {
    path = fs::absolute(path);
  }

  BOOST_FOREACH (const fs::path &p, std::make_pair(fs::directory_iterator(path),
                                                   fs::directory_iterator())) {
    if (!fs::is_directory(p)) {
      img_filename_list.push_back(p.filename().string());
      img_path_list.push_back(fs::absolute(p).string());
    }
  }

  std::sort(img_filename_list.begin(), img_filename_list.end());
  std::sort(img_path_list.begin(), img_path_list.end());
}
}  // namespace

void RaiseAllFilesInDirectory(const std::string &dirpath,
                              std::vector<std::string> &filelist,
                              const std::vector<std::string> &exts) {
  std::vector<std::string> tmp_img_path_list, tmp_file_name_list;
  RaiseAllFilesInDirectoryInternal(dirpath, tmp_img_path_list,
                                   tmp_file_name_list);

  for (size_t i = 0; i < tmp_img_path_list.size(); i++) {
    std::string abs_path = tmp_img_path_list[i];
    std::string filename = tmp_file_name_list[i];
    for (auto ext : exts) {
      if (ext.size() == 0) {
        continue;
      }

      if (abs_path.find(ext) == abs_path.size() - ext.size()) {
        filelist.push_back(abs_path);
        break;
      }
    }
  }
}

bool ExtractFilename(const std::string &path, std::string &filename) {
  filename = "";
  boost::filesystem::path p{path};
  if (!boost::filesystem::is_regular_file(p)) {
    return false;
  }
  filename = std::string(p.filename().c_str());
  return true;
}

bool ExtractAbsoluteDirectoryPathFromDirectoryPath(
    const std::string &path, std::string &directory_path) {
  directory_path = "";
  boost::filesystem::path p{path};
  if (!boost::filesystem::is_directory(p)) {
    return false;
  }

  if (!p.has_parent_path()) {
    return false;
  }
  boost::filesystem::path abs_dir =
      boost::filesystem::absolute(p.parent_path());
  directory_path = std::string(abs_dir.c_str());
  return true;
}

bool ExtractAbsoluteDirectoryPathFromFilepath(const std::string &path,
                                              std::string &directory_path) {
  directory_path = "";
  boost::filesystem::path p{path};
  if (boost::filesystem::is_directory(p)) {
    return false;
  }

  if (!p.has_parent_path()) {
    return false;
  }
  boost::filesystem::path abs_dir =
      boost::filesystem::absolute(p.parent_path());
  directory_path = std::string(abs_dir.c_str());
  return true;
}

bool CheckDirectory(const std::string &path) {
  boost::filesystem::path p(path);
  if (!boost::filesystem::is_directory(path)) {
    return false;
  }
  return true;
}

bool CreateDirectoryIfNotExist(const std::string &path) {
  boost::filesystem::path p(path);

  if (boost::filesystem::exists(p)) {
    if (boost::filesystem::is_directory(p)) {
      return true;
    } else {
      return false;
    }
  }

  return boost::filesystem::create_directories(p);
}
}  // namespace fileutil

#endif  // FILESYSTEM_UTIL_CC_
