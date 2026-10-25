#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

namespace FlowUtils {

// File I/O related
void savePointsToFile(const std::vector<cv::Point2f>& points, const std::string& filepath);
void save_edge_dat(const cv::Mat& edge_map, const std::string& filename);
cv::Mat readOpticalFlowFile(const std::string& path);
void parseMatchesFile(const std::string& matches_path,
                     std::vector<cv::Point2f>& src_points,
                     std::vector<cv::Point2f>& dst_points);

// Cache management related
bool isFileUpToDate(const std::string& targetFile, const std::string& sourceFile);
bool isPointsCacheValid(const std::string& pointsFile, const std::string& imageFile);
bool isEdgeCacheValid(const std::string& edgeFile, const std::string& imageFile);
std::vector<cv::Point2f> loadCachedPoints(const std::string& pointsFile);

} 

#endif // UTILS_H