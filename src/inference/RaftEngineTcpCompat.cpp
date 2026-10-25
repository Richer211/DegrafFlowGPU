#include "inference/RaftEngineTcpCompat.h"

#include <iostream>

bool RaftEngineTcpCompat::estimateMatchesBatch(
    const std::vector<cv::Mat> &,
    const std::vector<cv::Mat> &,
    const std::vector<std::vector<cv::Point2f>> &,
    std::vector<SparseFlowMatches> &)
{
    std::cerr << "[DEPRECATED] RAFT TCP compatibility engine is disabled in single-process mode." << std::endl;
    return false;
}

