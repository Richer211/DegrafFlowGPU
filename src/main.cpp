
#include "stdafx.h"

// FS code
#include "FeatureMatcher.h"
#include "SaliencyDetector.h"
#include "EvaluateOptFlow.h"
#include "EvaluateSceneFlow.h"
#include "vo_features.h"

// OpenCV - requires contrib modules
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <map>
#include <sstream>
#include <fstream>
#include <ctime>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace std;

static void printUsage(const char *prog)
{
	cout << "Usage: " << prog << " [options]\n"
		 << "Options:\n"
		 << "  --start <int>         Start frame index (default: 0)\n"
		 << "  --count <int>         Number of frame pairs to evaluate (default: 10)\n"
		 << "  --batch-size <int>    Batch size for interponet mode (default: 5)\n"
		 << "  --no-batch            Disable batch processing\n"
		 << "  --display             Enable visualization windows\n"
		 << "  --methods <csv>       Methods to run, e.g. degraf_flow_interponet,degraf_flow_rlof\n"
		 << "  --help                Show this help message\n";
}

static vector<string> splitCSV(const string &input)
{
	vector<string> out;
	string item;
	std::stringstream ss(input);
	while (std::getline(ss, item, ','))
	{
		if (!item.empty())
			out.push_back(item);
	}
	return out;
}

static string makeTimestamp()
{
	time_t now = time(nullptr);
	tm *lt = localtime(&now);
	char buf[32];
	strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", lt);
	return string(buf);
}

static bool copyFileBinary(const string &src, const string &dst)
{
	ifstream in(src, ios::binary);
	if (!in.is_open())
		return false;
	ofstream out(dst, ios::binary | ios::trunc);
	if (!out.is_open())
		return false;
	out << in.rdbuf();
	return true;
}

static void freezeBaselineOutputs(
	const std::map<String, std::vector<OptFlowMetrics>> &optical_method_results,
	const std::map<std::string, std::vector<SceneFlowMetrics>> &scene_method_results)
{
	const string baseline_dir = "../data/outputs/baseline";
	cv::utils::fs::createDirectories(baseline_dir);
	const string ts = makeTimestamp();

	copyFileBinary("../data/outputs/table_i_optical_flow.csv", baseline_dir + "/table_i_optical_flow_" + ts + ".csv");
	copyFileBinary("../data/outputs/table_i_optical_flow_official.csv", baseline_dir + "/table_i_optical_flow_official_" + ts + ".csv");
	copyFileBinary("../data/outputs/table_ii_scene_flow.csv", baseline_dir + "/table_ii_scene_flow_" + ts + ".csv");

	ofstream summary(baseline_dir + "/baseline_summary_" + ts + ".csv", ios::trunc);
	if (!summary.is_open())
		return;

	summary << "type,method,epe_or_epe3d,r2_or_accs,r3_or_accr,outlier,time_ms,std\n";
	for (const auto &pair : optical_method_results)
	{
		const auto &method = pair.first;
		const auto &rows = pair.second;
		if (rows.empty())
			continue;
		double avg_epe = 0, avg_r2 = 0, avg_r3 = 0, avg_time = 0, avg_std = 0;
		for (const auto &m : rows)
		{
			avg_epe += m.EPE;
			avg_r2 += m.R2;
			avg_r3 += m.R3;
			avg_time += m.time_ms;
			avg_std += m.std_dev;
		}
		const double n = static_cast<double>(rows.size());
		summary << "optical," << method << ","
				<< (avg_epe / n) << ","
				<< (avg_r2 / n) << ","
				<< (avg_r3 / n) << ","
				<< 0.0 << ","
				<< (avg_time / n) << ","
				<< (avg_std / n) << "\n";
	}

	for (const auto &pair : scene_method_results)
	{
		const auto &method = pair.first;
		const auto &rows = pair.second;
		if (rows.empty())
			continue;
		double avg_epe3d = 0, avg_accs = 0, avg_accr = 0, avg_outlier = 0, avg_time = 0;
		for (const auto &m : rows)
		{
			avg_epe3d += m.EPE3d;
			avg_accs += m.AccS;
			avg_accr += m.AccR;
			avg_outlier += m.Outlier;
			avg_time += m.time_ms;
		}
		const double n = static_cast<double>(rows.size());
		summary << "scene," << method << ","
				<< (avg_epe3d / n) << ","
				<< (avg_accs / n) << ","
				<< (avg_accr / n) << ","
				<< (avg_outlier / n) << ","
				<< (avg_time / n) << ","
				<< 0.0 << "\n";
	}
}

int main(int argc, char **argv)
{
	////////////////////////// Optical Flow (GPU pipeline) //////////////////////////
	bool use_batch_processing = true;
	int batch_size = 5;
	bool display_images = false;
	int start_image = 0;
	int total_images = 10;

	std::vector<String> optical_methods = {
		"degraf_flow_interponet",
		"degraf_flow_rlof",
	};
	std::vector<std::string> scene_methods = {
		"degraf_flow_interponet",
		"degraf_flow_rlof",
	};

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "--help")
		{
			printUsage(argv[0]);
			return 0;
		}
		else if (arg == "--start" && i + 1 < argc)
		{
			start_image = std::stoi(argv[++i]);
		}
		else if (arg == "--count" && i + 1 < argc)
		{
			total_images = std::stoi(argv[++i]);
		}
		else if (arg == "--batch-size" && i + 1 < argc)
		{
			batch_size = std::stoi(argv[++i]);
		}
		else if (arg == "--no-batch")
		{
			use_batch_processing = false;
		}
		else if (arg == "--display")
		{
			display_images = true;
		}
		else if (arg == "--methods" && i + 1 < argc)
		{
			vector<string> methods = splitCSV(argv[++i]);
			optical_methods.clear();
			scene_methods.clear();
			for (const auto &m : methods)
			{
				optical_methods.push_back(m);
				scene_methods.push_back(m);
			}
		}
		else
		{
			cerr << "Unknown/invalid argument: " << arg << "\n";
			printUsage(argv[0]);
			return 1;
		}
	}

	cout << "[Config] start=" << start_image
		 << ", count=" << total_images
		 << ", batch=" << (use_batch_processing ? "on" : "off")
		 << ", batch_size=" << batch_size
		 << ", display=" << (display_images ? "on" : "off") << endl;

	cv::utils::fs::createDirectories("../data/outputs");

	EvaluateOptFlow evaluator;
	std::map<String, std::vector<OptFlowMetrics>> optical_method_results;
	cout << "Running GPU optical flow evaluation (InterpoNet + RLOF baseline)." << endl;

	for (const auto &method : optical_methods)
	{
		cout << "\n--- Evaluating optical flow method: " << method << " ---" << endl;

		evaluator.clearResults();

		if (use_batch_processing && method == "degraf_flow_interponet")
		{
			// InterpoNet uses batch processing
			cout << "Using batch processing for " << method << " (batch_size=" << batch_size << ")" << endl;

			for (int batch_idx = 0; batch_idx < total_images; batch_idx += batch_size)
			{
				int current_batch_size = std::min(batch_size, total_images - batch_idx);
				std::vector<int> batch_indices;
				for (int i = 0; i < current_batch_size; ++i)
				{
					batch_indices.push_back(start_image + batch_idx + i);
				}

				cout << "Processing batch: frames " << batch_indices.front()
					 << "-" << batch_indices.back() << endl;

				// Call the batch version of the unified interface
				std::vector<OptFlowMetrics> batch_results =
					evaluator.runEvaluation(method, display_images, batch_indices);

				for (const auto &result : batch_results)
				{
					optical_method_results[method].push_back(result);
				}
			}
		}
		else
		{
			for (int i = start_image; i < start_image + total_images; ++i)
			{
				cout << "\nOPTICAL FLOW [" << method << "] Frame #: " << i << "\n";

				// Call the single-frame version of the unified interface
				std::vector<int> single_index = {i};
				std::vector<OptFlowMetrics> single_result =
					evaluator.runEvaluation(method, display_images, single_index);

				if (!single_result.empty())
				{
					optical_method_results[method].push_back(single_result[0]);
				}
			}
		}

		const auto &results = optical_method_results[method];
		if (!results.empty())
		{
			cout << "\n--- Optical Flow Stats [" << method << "] ---\n";
			cout << "# EPE      STD      0.5     1        2        3       5       10       time\n";

			double avg_EPE = 0, avg_std = 0, avg_R05 = 0, avg_R1 = 0, avg_R2 = 0, avg_R3 = 0, avg_R5 = 0, avg_R10 = 0, avg_time = 0;
			double avg_Fl_bg = 0, avg_Fl_fg = 0, avg_Fl_all = 0;

			for (size_t j = 0; j < results.size(); j++)
			{
				const OptFlowMetrics &metrics = results[j];

				printf("%d %.4f %.4f %.2f %.2f %.2f %.2f %.2f %.2f %.3f\n",
					   metrics.image_no, metrics.EPE, metrics.std_dev,
					   metrics.R05, metrics.R1, metrics.R2, metrics.R3,
					   metrics.R5, metrics.R10, metrics.time_ms);

				avg_EPE += metrics.EPE;
				avg_std += metrics.std_dev;
				avg_R05 += metrics.R05;
				avg_R1 += metrics.R1;
				avg_R2 += metrics.R2;
				avg_R3 += metrics.R3;
				avg_R5 += metrics.R5;
				avg_R10 += metrics.R10;
				avg_time += metrics.time_ms;
				avg_Fl_bg += metrics.Fl_bg;
				avg_Fl_fg += metrics.Fl_fg;
				avg_Fl_all += metrics.Fl_all;
			}

			size_t count = results.size();
			cout << "\nAverages for " << method << ":\n";
			cout << "Average EPE: " << avg_EPE / count << "\n";
			cout << "Average R2.0: " << avg_R2 / count << "\n";
			cout << "Average R3.0: " << avg_R3 / count << "\n";
			cout << "Average Time: " << avg_time / count << " ms\n";
			cout << "Average STD: " << avg_std / count << "\n";
			cout << "Official Fl-bg: " << avg_Fl_bg / count << "%\n";
			cout << "Official Fl-fg: " << avg_Fl_fg / count << "%\n";
			cout << "Official Fl-all: " << avg_Fl_all / count << "%\n";
			cout << "--------------------------------------------\n";
		}
	}
	evaluator.exportOpticalFlowTableCSV("../data/outputs/table_i_optical_flow.csv", optical_method_results);
	evaluator.exportPaperOpticalFlowTableCSV("../data/outputs/table_i_optical_flow_official.csv", optical_method_results);

	////////////////////////// 3D Scene Flow evaluation //////////////////////////
	EvaluateSceneFlow evaluatorscene;
	std::map<std::string, std::vector<SceneFlowMetrics>> scene_method_results;

	for (const auto &method : scene_methods)
	{
		cout << "\n--- Evaluating scene flow method: " << method << " ---" << endl;
		evaluatorscene.clearResults();

		if (use_batch_processing && method == "degraf_flow_interponet")
		{
			cout << "Using batch processing for " << method << " (batch_size=" << batch_size << ")" << endl;
			for (int batch_idx = 0; batch_idx < total_images; batch_idx += batch_size)
			{
				int current_batch_size = std::min(batch_size, total_images - batch_idx);
				std::vector<int> batch_indices;
				for (int i = 0; i < current_batch_size; ++i)
				{
					batch_indices.push_back(start_image + batch_idx + i);
				}
				std::vector<SceneFlowMetrics> batch_results =
					evaluatorscene.runEvaluation(method, display_images, batch_indices);

				scene_method_results[method].insert(
					scene_method_results[method].end(),
					batch_results.begin(),
					batch_results.end());
			}
		}
		else
		{
			for (int i = start_image; i < start_image + total_images; ++i)
			{
				SceneFlowMetrics result = evaluatorscene.runEvaluation(method, display_images, i);
				scene_method_results[method].push_back(result);
			}
		}

		const auto &results = scene_method_results[method];
		if (!results.empty())
		{
			double avg_EPE3d = 0, avg_AccS = 0, avg_AccR = 0, avg_Outlier = 0, avg_time = 0;
			for (const auto &metrics : results)
			{
				avg_EPE3d += metrics.EPE3d;
				avg_AccS += metrics.AccS;
				avg_AccR += metrics.AccR;
				avg_Outlier += metrics.Outlier;
				avg_time += metrics.time_ms;
			}
			size_t count = results.size();
			cout << "SceneFlow averages for " << method
				 << " | EPE3d: " << avg_EPE3d / count
				 << " | AccS: " << avg_AccS / count
				 << " | AccR: " << avg_AccR / count
				 << " | Outlier: " << avg_Outlier / count
				 << " | Time: " << avg_time / count << " ms" << endl;
		}
	}
	evaluatorscene.exportSceneFlowComparisonCSV("../data/outputs/table_ii_scene_flow.csv", scene_method_results);
	freezeBaselineOutputs(optical_method_results, scene_method_results);

	return 0;
}