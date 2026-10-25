#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct SparseFlowMatches
{
    std::vector<cv::Point2f> src_points; // 源图像中的特征点
    std::vector<cv::Point2f> dst_points; // 目标图像中的特征点
};

class IRaftEngine
{
public:
    virtual ~IRaftEngine() = default;

    virtual bool estimateMatchesBatch(
        const std::vector<cv::Mat> &batch_i1, // 源图像
        const std::vector<cv::Mat> &batch_i2, // 目标图像
        const std::vector<std::vector<cv::Point2f>> &batch_points, // 特征点
        std::vector<SparseFlowMatches> &batch_matches) = 0; // 匹配结果
};

class IInterpoNetEngine
{
public:
    virtual ~IInterpoNetEngine() = default;

    virtual bool densifyBatch(
        const std::vector<cv::Mat> &batch_i1, // 源图像
        const std::vector<cv::Mat> &batch_i2, // 目标图像
        const std::vector<SparseFlowMatches> &batch_matches, // 匹配结果
        std::vector<cv::Mat> &batch_flows) = 0; // 光流结果
};

