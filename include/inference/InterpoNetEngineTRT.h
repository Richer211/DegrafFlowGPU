#pragma once

#include "inference/IFlowInferenceEngine.h"
#include <opencv2/core.hpp>

// NOTE:
// This class is intentionally named TRT to match the migration target.
// Current implementation is a C++ in-process fallback (no TCP/service).
class InterpoNetEngineTRT final : public IInterpoNetEngine
{
public:
    InterpoNetEngineTRT(int k = 128, float sigma = 0.05f, bool use_post_proc = true,
                        float fgs_lambda = 500.0f, float fgs_sigma = 1.5f);

    bool densifyBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<SparseFlowMatches> &batch_matches,
        std::vector<cv::Mat> &batch_flows) override;

    bool densifyBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<SparseFlowMatches> &batch_matches,
        const std::vector<SparseFlowMatches> *batch_matches_ba,
        std::vector<cv::Mat> &batch_flows);

private:
    int k_;
    float sigma_;
    bool use_post_proc_;
    float fgs_lambda_;
    float fgs_sigma_;
};

