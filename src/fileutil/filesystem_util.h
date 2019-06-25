
#ifndef FILESYSTEM_UTIL_H_
#define FILESYSTEM_UTIL_H_

// Standard
#include <string>
#include <vector>

namespace fileutil {

void RaiseAllFilesInDirectory(
    const std::string &dirpath, std::vector<std::string> &filelist,
    const std::vector<std::string> &exts = std::vector<std::string>());

bool ExtractFilename(const std::string &path, std::string &filename);

bool ExtractAbsoluteDirectoryPathFromDirectoryPath(const std::string &path,
                                                   std::string &directory_path);

bool ExtractAbsoluteDirectoryPathFromFilepath(const std::string &path,
                                              std::string &directory_path);

bool CheckDirectory(const std::string &path);

bool CreateDirectoryIfNotExist(const std::string &path);

} // namespace fileutil

#endif // FILESYSTEM_UTIL_H_
