/*!
\file EvaluateSceneFlow.h
\brief Scene Flow evaluation module - Fixed version
\author Gang Wang, Durham University
*/

#pragma once

#include "SceneFlowReconstructor.h"
#include "FeatureMatcher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <map>
using namespace cv;

// Standardized scene flow evaluation indicator structure
struct SceneFlowMetrics
{
    double EPE3d; // Average 3D endpoint error (meters)
    double AccS; // Strict accuracy (EPE < 0.05m or relative error < 5%)
    double AccR; // Loose accuracy (EPE < 0.1m or relative error < 10%)
    double Outlier; // Outlier ratio (EPE > 0.3m or relative error > 10%)
    int valid_count; // Number of valid pixels
    double time_ms; // Computation time (milliseconds)

    // Constructor
    SceneFlowMetrics() : EPE3d(0.0), AccS(0.0), AccR(0.0), Outlier(0.0), valid_count(0), time_ms(0.0) {}
};

class EvaluateSceneFlow
{
public:
    EvaluateSceneFlow();

    std::vector<SceneFlowMetrics> runEvaluation(
        const std::string &method,
        bool display_images,
        const std::vector<int> &image_indices);

    SceneFlowMetrics runEvaluation(
        const std::string &method,
        bool display_images,
        int image_no);

    const std::vector<SceneFlowMetrics> &getAllResults() const { return all_results_; }

    void clearResults() { all_results_.clear(); }

    void exportSceneFlowComparisonCSV(
        const std::string& csv_path,
        const std::map<std::string, std::vector<SceneFlowMetrics>>& method_results);
    


private:
    std::vector<SceneFlowMetrics> all_results_; // 存储所有评估结果

    SceneFlowMetrics calculateStandardMetrics(const cv::Mat &pred_scene_flow,
                                              const cv::Mat &gt_scene_flow);
    SceneFlowMetrics evaluateSingleFrame(const cv::Mat &pred_scene_flow,
                                         const cv::Mat &gt_scene_flow,
                                         bool verbose = true);
    void writeMetricsToCSV(const SceneFlowMetrics &metrics,
                           const std::string &method,
                           int image_no,
                           const std::string &csv_path);
};