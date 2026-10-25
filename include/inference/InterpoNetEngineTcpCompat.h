#pragma once

#include "inference/IFlowInferenceEngine.h"

class InterpoNetEngineTcpCompat final : public IInterpoNetEngine
{
public:
    bool densifyBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<SparseFlowMatches> &batch_matches,
        std::vector<cv::Mat> &batch_flows) override;
};

