
#include "utils.h"
#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace FlowUtils
{
void parseMatchesFile(const std::string &matches_path,
                      std::vector<cv::Point2f> &src_points,
                      std::vector<cv::Point2f> &dst_points)
{
    src_points.clear();
    dst_points.clear();

    std::ifstream file(matches_path);
    if (!file.is_open())
        return;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float x0, y0, x1, y1;
        if (iss >> x0 >> y0 >> x1 >> y1)
        {
            src_points.emplace_back(x0, y0);
            dst_points.emplace_back(x1, y1);
        }
    }
}

void save_edge_dat(const cv::Mat &edge_map, const std::string &filename)
{
    cv::Mat edge_float;
    if (edge_map.type() != CV_32F)
        edge_map.convertTo(edge_float, CV_32F);
    else
        edge_float = edge_map;

    if (edge_float.channels() > 1)
        cv::cvtColor(edge_float, edge_float, cv::COLOR_BGR2GRAY);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(edge_float.data),
              edge_float.rows * edge_float.cols * sizeof(float));
}

cv::Mat readOpticalFlowFile(const std::string &path)
{
    std::ifstream file(path.c_str(), std::ios_base::binary);
    if (!file.good())
    {
        printf("Error opening flow file: %s\n", path.c_str());
        return cv::Mat();
    }

    float magic;
    file.read((char *)&magic, sizeof(float));
    if (magic != 202021.25f)
    {
        printf("Invalid .flo file magic number\n");
        return cv::Mat();
    }

    int width, height;
    file.read((char *)&width, sizeof(int));
    file.read((char *)&height, sizeof(int));

    cv::Mat flow(height, width, CV_32FC2);
    file.read((char *)flow.data, width * height * 2 * sizeof(float));
    return flow;
}

void savePointsToFile(const std::vector<cv::Point2f> &points, const std::string &filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
        return;
    for (const auto &point : points)
        file << point.x << " " << point.y << "\n";
}

bool isFileUpToDate(const std::string &targetFile, const std::string &sourceFile)
{
    std::ifstream f(targetFile);
    if (!f.good())
        return false;
    f.close();

    struct stat targetStat, sourceStat;
    if (stat(targetFile.c_str(), &targetStat) != 0 || stat(sourceFile.c_str(), &sourceStat) != 0)
        return false;
    return targetStat.st_mtime >= sourceStat.st_mtime;
}

bool isPointsCacheValid(const std::string &pointsFile, const std::string &imageFile)
{
    return isFileUpToDate(pointsFile, imageFile);
}

bool isEdgeCacheValid(const std::string &edgeFile, const std::string &imageFile)
{
    return isFileUpToDate(edgeFile, imageFile);
}

std::vector<cv::Point2f> loadCachedPoints(const std::string &pointsFile)
{
    std::vector<cv::Point2f> points;
    std::ifstream file(pointsFile);
    if (!file.is_open())
        return points;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float x, y;
        if (iss >> x >> y)
            points.emplace_back(x, y);
    }
    return points;
}

} // namespace FlowUtils