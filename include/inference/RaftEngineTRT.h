#pragma once

#include "inference/IFlowInferenceEngine.h"

// NOTE:
// This class is intentionally named TRT to match the migration target.
// Current implementation is a C++ in-process fallback (no TCP/service).
class RaftEngineTRT final : public IRaftEngine
{
public:
    bool estimateMatchesBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<std::vector<cv::Point2f>> &batch_points,
        std::vector<SparseFlowMatches> &batch_matches) override;
};

