// Header file for the heavily modified version of the openCV optical_flow_evaluation.cpp file:
// https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_evaluation.cpp
// Adapted to be able to evaluate KITTI and Middlebury datasets.

#pragma once

#include "SaliencyDetector.h"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> 
#include "opencv2/calib3d.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>

#include <iostream>	 // standard C++ I/O
#include <string>	 // standard C++ I/O
#include <algorithm> // includes max()
#include <vector>

using namespace cv;

struct OptFlowMetrics 
{
    float EPE;
    float Fl_bg;
    float Fl_fg;
    float Fl_all;
    float std_dev;
    float R05;
    float R1;
    float R2;
    float R3;
    float R5;
    float R10;
    double time_ms;
    int image_no;
    
    OptFlowMetrics() : EPE(0.0f), Fl_bg(0.0f), Fl_fg(0.0f), Fl_all(0.0f), std_dev(0.0f), R05(0.0f), R1(0.0f), 
                       R2(0.0f), R3(0.0f), R5(0.0f), R10(0.0f), 
                       time_ms(0.0), image_no(0) {}
};
class EvaluateOptFlow
{

public:
	// Stats for a single flow computation
	std::vector<float> stats_vector;

	// List of stats over all image pairs
	std::vector<std::vector<float>> all_stats;

    std::vector<OptFlowMetrics> all_results_;

	EvaluateOptFlow();

	/*inline bool isFlowCorrect(const Point2f u);
	inline bool isFlowCorrect(const Point3f u);
	static Mat endpointError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2);
	static Mat angularError(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2);
	static float stat_RX(Mat errors, float threshold, Mat mask);
	static float stat_AX(Mat hist, int cutoff_count, float max_value);
	static void calculateStats(Mat errors, Mat mask = Mat(), bool display_images = false);
	static Mat flowToDisplay(const Mat flow);*/

	void calculateStats(Mat errors, Mat mask, bool display_images); // adding this as public so it can update the stats_vector variable

	int runEvaluation(String method, bool display, int image_no);

	std::vector<OptFlowMetrics> runEvaluation(const String &method, bool display_images, const std::vector<int> &image_indices);
    void calculateStatsForMetrics(Mat errors, Mat mask, OptFlowMetrics& metrics);
    void clearResults();
    // Internal/debug table: EPE + RX thresholds (project-local).
    void exportOpticalFlowTableCSV(
        const std::string& csv_path,
        const std::map<std::string, std::vector<OptFlowMetrics>>& method_results);
    // Paper Table I style on KITTI training: EPE + official Fl-bg/Fl-fg/Fl-all.
    void exportPaperOpticalFlowTableCSV(
        const std::string& csv_path,
        const std::map<std::string, std::vector<OptFlowMetrics>>& method_results);
};