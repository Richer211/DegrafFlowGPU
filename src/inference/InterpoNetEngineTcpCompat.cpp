#include "inference/InterpoNetEngineTcpCompat.h"

#include <iostream>

bool InterpoNetEngineTcpCompat::densifyBatch(
    const std::vector<cv::Mat> &,
    const std::vector<cv::Mat> &,
    const std::vector<SparseFlowMatches> &,
    std::vector<cv::Mat> &)
{
    std::cerr << "[DEPRECATED] InterpoNet TCP compatibility engine is disabled in single-process mode." << std::endl;
    return false;
}

