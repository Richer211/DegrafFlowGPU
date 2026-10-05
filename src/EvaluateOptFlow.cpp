// Heavily modified version of the openCV optical_flow_evaluation.cpp file:
// https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_evaluation.cpp
// Adapted to be able to evaluate KITTI and Middlebury datasets.

#include "stdafx.h"

#include "EvaluateOptFlow.h"
#include "FeatureMatcher.h"
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::optflow;
using namespace std;

static String getDataSceneFlowRoot() // 获取数据集根目录
{
    const char *env_path = std::getenv("DEGRAF_DATA_PATH"); // 获取环境变量DEGRAF_DATA_PATH的值
    if (env_path && std::string(env_path).size() > 0) // 如果环境变量DEGRAF_DATA_PATH的值不为空且长度大于0
        return String(env_path); // 将环境变量DEGRAF_DATA_PATH的值转换为String类型，并返回
    return String("/root/autodl-tmp/data/kitti/data_scene_flow"); // 如果环境变量DEGRAF_DATA_PATH的值为空或长度为0，则返回默认值"/root/autodl-tmp/data/kitti/data_scene_flow"
}

static std::string getKittiProtocol()
{
    const char *env_protocol = std::getenv("DEGRAF_KITTI_PROTOCOL");
    if (!env_protocol)
        return "occ";
    std::string protocol(env_protocol);
    if (protocol == "noc")
        return "noc";
    return "occ";
}

static bool fileExists(const std::string &path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

EvaluateOptFlow::EvaluateOptFlow()
{
}

inline bool isFlowCorrect(const Point2f u)
{
	return !std::isnan(u.x) && !std::isnan(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect(const Point3f u)
{
	return !std::isnan(u.x) && !std::isnan(u.y) && !std::isnan(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9) && (fabs(u.z) < 1e9);
}

// Converts middlebury 2 channel flow image to 3 channel (CV_16UC3) KITTI format
/*!
\param flow 2 channel optical flow Mat, middlebury format
\return 3 channel KITTI optical flow Mat
*/
static Mat convertToKittiFlow(const Mat_<Point2f> &flow)
{

	Mat kittiFlow = cv::Mat::ones(flow.rows, flow.cols, CV_16UC3); // type 18

	int width = flow.cols;
	int height = flow.rows;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{

			const Point2f pt = flow(i, j);
			float u = pt.x;
			float v = pt.y;

			kittiFlow.at<Vec3s>(i, j)[1] = (uint16_t)max(min(v * 64.0f + 32768.0f, 65535.0f), 0.0f);

			kittiFlow.at<Vec3s>(i, j)[2] = (uint16_t)max(min(u * 64.0f + 32768.0f, 65535.0f), 0.0f);
		}
	}
	return kittiFlow;
}

// Converts KITTI ground truth image into a middlebury 2 channel image for evaluation
/*!
\param ground_truth_path string specifying the path to the KITTI 3 channel ground truth image
\return 2 channel Mat, ground truth optical flow
*/
static Mat readKittiGroundTruth(String ground_truth_path)
{

	String path = ground_truth_path;
	// NB opencv has order BGR => valid , v , u
	Mat image = imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

	Mat gt = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
	int width = image.cols;
	int height = image.rows;

	for (int32_t v = 0; v < height; v++)
	{
		for (int32_t u = 0; u < width; u++)
		{
			Vec3s val = image.at<Vec3s>(v, u);
			if (val[0] > 0)
			{
				Vec2f flow;
				if (val[2] > 0)
				{
					flow[0] = ((float)val[2] - 32768.0f) / 64.0f;
				}
				else
				{
					flow[0] = ((float)val[2] + 32768.0f) / 64.0f;
				}

				if (val[1] > 0)
				{
					flow[1] = ((float)val[1] - 32768.0f) / 64.0f;
				}
				else
				{
					flow[1] = ((float)val[1] + 32768.0f) / 64.0f;
				}
				gt.at<Vec2f>(v, u) = flow;
			}
			else
			{
				Vec2f flow;
				flow[0] = std::numeric_limits<float>::quiet_NaN();
				flow[1] = std::numeric_limits<float>::quiet_NaN();
				gt.at<Vec2f>(v, u) = flow;
			}
		}
	}
	return gt;
}

// Replacement for cv::optflow::readOpticalFlow - reads .flo files
static Mat readOpticalFlowFile(const String &path)
{
	std::ifstream file(path.c_str(), std::ios_base::binary);
	if (!file.good())
	{
		printf("Error opening flow file: %s\n", path.c_str());
		return Mat();
	}

	float magic;
	file.read((char *)&magic, sizeof(float));
	if (magic != 202021.25f) // .flo file magic number
	{
		printf("Invalid .flo file magic number\n");
		return Mat();
	}

	int width, height;
	file.read((char *)&width, sizeof(int));
	file.read((char *)&height, sizeof(int));

	Mat flow(height, width, CV_32FC2);
	file.read((char *)flow.data, width * height * 2 * sizeof(float));
	file.close();

	return flow;
}

static Mat endpointError(const Mat_<Point2f> &flow1, const Mat_<Point2f> &flow2)
{
	Mat result(flow1.size(), CV_32FC1);

	for (int i = 0; i < flow1.rows; ++i)
	{
		for (int j = 0; j < flow1.cols; ++j)
		{
			const Point2f u1 = flow1(i, j);
			const Point2f u2 = flow2(i, j);

			if (isFlowCorrect(u1) && isFlowCorrect(u2))
			{
				const Point2f diff = u1 - u2;
				result.at<float>(i, j) = sqrt((float)diff.ddot(diff)); // distance
			}
			else
				result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
		}
	}
	return result;
}

static Mat angularError(const Mat_<Point2f> &flow1, const Mat_<Point2f> &flow2)
{
	Mat result(flow1.size(), CV_32FC1);

	for (int i = 0; i < flow1.rows; ++i)
	{
		for (int j = 0; j < flow1.cols; ++j)
		{
			const Point2f u1_2d = flow1(i, j);
			const Point2f u2_2d = flow2(i, j);
			const Point3f u1(u1_2d.x, u1_2d.y, 1);
			const Point3f u2(u2_2d.x, u2_2d.y, 1);

			if (isFlowCorrect(u1) && isFlowCorrect(u2))
				result.at<float>(i, j) = acos((float)(u1.ddot(u2) / norm(u1) * norm(u2)));
			else
				result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
		}
	}
	return result;
}

// what fraction of pixels have errors higher than given threshold?
static float stat_RX(Mat errors, float threshold, Mat mask)
{
	CV_Assert(errors.size() == mask.size());
	CV_Assert(mask.depth() == CV_8U);

	int count = 0, all = 0;
	for (int i = 0; i < errors.rows; ++i)
	{
		for (int j = 0; j < errors.cols; ++j)
		{
			if (mask.at<char>(i, j) != 0)
			{
				++all;
				if (errors.at<float>(i, j) > threshold)
					++count;
			}
		}
	}
	return (float)count / all;
}

static float stat_AX(Mat hist, int cutoff_count, float max_value)
{
	int counter = 0;
	int bin = 0;
	int bin_count = hist.rows;
	while (bin < bin_count && counter < cutoff_count)
	{
		counter += (int)hist.at<float>(bin, 0);
		++bin;
	}
	return (float)bin / bin_count * max_value;
}

void EvaluateOptFlow::calculateStats(Mat errors, Mat mask = Mat(), bool display_images = false)
{
	float R_thresholds[] = {0.5f, 1.f, 2.f, 3.f, 5.f, 10.f};
	float A_thresholds[] = {0.5f, 0.75f, 0.95f};
	if (mask.empty())
		mask = Mat::ones(errors.size(), CV_8U);
	CV_Assert(errors.size() == mask.size());
	CV_Assert(mask.depth() == CV_8U);

	// mean and std computation
	Scalar s_mean, s_std;
	float mean, std;
	meanStdDev(errors, s_mean, s_std, mask);
	mean = (float)s_mean[0];
	std = (float)s_std[0];
	printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

	// FS added to collect stats (printed out in main.cpp)
	stats_vector.push_back(mean);
	stats_vector.push_back(std);

	// RX stats - displayed in percent
	float R;
	int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
	for (int i = 0; i < R_thresholds_count; ++i)
	{
		R = stat_RX(errors, R_thresholds[i], mask);
		printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
		stats_vector.push_back(R * 100); // FS added to collect stats
	}

	// AX stats
	double max_value;
	minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

	Mat hist;
	const int n_images = 1;
	const int channels[] = {0};
	const int n_dimensions = 1;
	const int hist_bins[] = {1024};
	const float iranges[] = {0, (float)max_value};
	const float *ranges[] = {iranges};
	const bool uniform = true;
	const bool accumulate = false;
	calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
			 accumulate);
	int all_pixels = countNonZero(mask);
	int cutoff_count;
	float A;
	int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
	for (int i = 0; i < A_thresholds_count; ++i)
	{
		cutoff_count = (int)(floor(A_thresholds[i] * all_pixels + 0.5f));
		A = stat_AX(hist, cutoff_count, (float)max_value);
		// printf("A%.2f: %.2f\n", A_thresholds[i], A);
	}
}

static Mat flowToDisplay(const Mat flow)
{
	// Used to show gt for kitti data with NaN values
	Mat flow_copy = flow;
	for (int32_t v = 0; v < flow.rows; v++)
	{
		for (int32_t u = 0; u < flow.cols; u++)
		{
			if (std::isnan(flow.at<Vec2f>(v, u)[0]))
			{
				Vec2f f;
				f[0] = 0.0f;
				f[1] = 0.0f;
				flow_copy.at<Vec2f>(v, u) = f;
			}
		}
	}

	Mat flow_split[2];
	Mat magnitude, angle;
	Mat hsv_split[3], hsv, rgb;
	split(flow_copy, flow_split);

	///////////////////-- New colour display/////////////////////////
	Mat flowHSV, flowRGB;
	vector<cv::Mat> flowVec(3);

	cv::cartToPolar(flow_split[0], flow_split[1], flowVec[1], flowVec[0], true);
	flowVec[2] = cv::Mat::ones(flowVec[0].size(), flowVec[0].type()) * 255;
	cv::threshold(flowVec[1], flowVec[1], 4, 4, cv::THRESH_TRUNC);
	flowVec[1] = flowVec[1] * 0.25f;
	cv::merge(flowVec, flowHSV);
	cv::cvtColor(flowHSV, flowRGB, cv::COLOR_HSV2BGR);
	flowRGB.convertTo(flowRGB, CV_8UC3);
	return flowRGB;
}

static Mat flowToDisplay2(const Mat flow)
{
	Mat flow_split[2];
	Mat magnitude, angle;
	Mat hsv_split[3], hsv, rgb;
	split(flow, flow_split);
	cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
	normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
	hsv_split[0] = angle; // already in degrees - no normalization needed
	hsv_split[1] = Mat::ones(angle.size(), angle.type());
	hsv_split[2] = magnitude;
	merge(hsv_split, 3, hsv);
	cvtColor(hsv, rgb, COLOR_HSV2BGR);
	return rgb;
}

// Converts an error flow map into a linear heatmap blue -> red
// N.B set range of colour map in the function where specified
/*!
\param flow 2 channel mat of End Point Error vectors
\param mask Mask specifiying the pixels at which flow was recovered
\return error heat map from blue (low error) to red (high error)
*/
static Mat errorHeatMap(const Mat_<Point2f> &flow, Mat mask)
{

	Mat flow_magnitudes(flow.size(), CV_32FC1);
	Mat mask_copy;
	mask.copyTo(mask_copy);

	for (int i = 0; i < flow.rows; i++)
	{
		for (int j = 0; j < flow.cols; j++)
		{

			if (mask_copy.at<char>(i, j) != 0)
			{
				float flow_mag = sqrt(flow(i, j).x * flow(i, j).x + flow(i, j).y * flow(i, j).y);

				// Set max error to adjust the colour mapping
				if (flow_mag > 3)
				{
					flow_magnitudes.at<float>(i, j) = 3; // Adjust this range depending on the error range you want to see.
				}
				else
				{
					flow_magnitudes.at<float>(i, j) = flow_mag;
				}
			}
			else
			{
				flow_magnitudes.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
			}
		}
	}

	// Convert to 8 bit
	flow_magnitudes.convertTo(flow_magnitudes, CV_8U);

	Mat norm;
	// Normailse to get contrast
	normalize(flow_magnitudes, norm, 255, 0, NORM_INF);
	applyColorMap(norm, norm, COLORMAP_JET);
	bitwise_not(mask_copy, mask_copy);
	norm.setTo(Scalar(0, 0, 0), mask_copy);
	// Show the result:
	return norm;
}


// Unified batch/single frame evaluation interface
std::vector<OptFlowMetrics> EvaluateOptFlow::runEvaluation(const String &method, bool display_images, const std::vector<int> &image_indices)
{
    std::vector<OptFlowMetrics> results;
    
    // Determine whether batch processing is supported
    bool is_batch_capable = (method == "degraf_flow_interponet");
    
    // =====================================================
    // Step 1: Data Preparation
    // =====================================================
    struct ImagePairData {//定义一个结构体，用于存储图像对数据
        Mat i1, i2; // 存储第一张和第二张图像
        String i1_path, i2_path, groundtruth_path; // 存储第一张和第二张图像的路径，以及地面truth路径
        String num_str; // 存储图像编号
        int image_no; // 存储图像索引
    };
    
    std::vector<ImagePairData> batch_data; // 存储图像对数据
    batch_data.reserve(image_indices.size()); // 预留空间，避免频繁扩容
    
    // load all data
    for (int image_no : image_indices) { // image_indices表示要处理的图像索引列表, for循环遍历每个图像索引
        ImagePairData data; // 创建一个ImagePairData对象，用于存储图像对数据
        data.image_no = image_no; // 设置图像索引
        
        char num[7]; // 创建一个字符数组，用于存储图像编号
        sprintf(num, "%06d", image_no); // 将图像索引转换为字符串，并存储到num中
        data.num_str = String(num); // 将num转换为String类型，并存储到data.num_str中
        
        String data_root = getDataSceneFlowRoot(); // 获取数据集根目录
        String base_dir = data_root + "/training/";
        data.i1_path = base_dir + "image_2/" + data.num_str + "_10.png";
        data.i2_path = base_dir + "image_2/" + data.num_str + "_11.png";
        const std::string protocol = getKittiProtocol();
        const std::string flow_pref = std::string("flow_") + protocol + "/" + std::string(data.num_str) + "_10.png";
        const std::string flow_alt = std::string("flow_") + (protocol == "occ" ? "noc" : "occ") + "/" + std::string(data.num_str) + "_10.png";
        std::string flow_path = std::string(base_dir) + flow_pref;
        if (!fileExists(flow_path))
            flow_path = std::string(base_dir) + flow_alt;
        data.groundtruth_path = flow_path;
        

        data.i1 = imread(data.i1_path, 1); // 读取第一张图像
        data.i2 = imread(data.i2_path, 1); // 读取第二张图像
        
        if (!data.i1.data || !data.i2.data || data.i1.empty() || data.i2.empty()) {
            printf("No image data \n");
            continue;
        }
        if (data.i1.size() != data.i2.size() || data.i1.channels() != data.i2.channels()) {
            printf("Dimension mismatch between input images\n");
            continue;
        }
        

        if (data.i1.depth() != CV_8U) data.i1.convertTo(data.i1, CV_8U);
        if (data.i2.depth() != CV_8U) data.i2.convertTo(data.i2, CV_8U);
        
        batch_data.push_back(data);
    }
    
    if (batch_data.empty()) {
        return results;
    }
    
    // =====================================================
    // Step 2: Optical flow calculation (batch or frame by frame)
    // =====================================================
    std::vector<Mat> batch_flows;
    std::vector<double> individual_times;
    std::vector<vector<Point2f>> batch_points1, batch_points2; 
    
    if (is_batch_capable && batch_data.size() > 1) {
        // Batch processing: degraf_flow_interponet
        std::vector<Mat> batch_i1, batch_i2;
        std::vector<String> batch_num_strs;
        
        for (const auto& data : batch_data) {
            batch_i1.push_back(data.i1);
            batch_i2.push_back(data.i2);
            batch_num_strs.push_back(data.num_str);
        }
        
        double batch_start = getTickCount();
        FeatureMatcher matcher;
        
        // Call the batch version
        std::vector<std::vector<Point2f>> batch_points, batch_dst_points;
        batch_flows = matcher.degraf_flow_InterpoNet(
            batch_i1, batch_i2, batch_num_strs,
            display_images ? &batch_points : nullptr,
            display_images ? &batch_dst_points : nullptr
        );
        
        double total_time_ms = (getTickCount() - batch_start) / getTickFrequency() * 1000.0;
        
        // Evenly distribute time
        for (size_t i = 0; i < batch_flows.size(); ++i) {
            individual_times.push_back(total_time_ms / batch_flows.size());
        }
        
        //Store feature points for visualization
        if (display_images && !batch_points.empty()) {
            batch_points1 = batch_points;
            batch_points2 = batch_dst_points;
        }
        
    } else {
        // Frame-by-frame processing: all other methods
        batch_flows.resize(batch_data.size());
        individual_times.resize(batch_data.size());
        batch_points1.resize(batch_data.size());
        batch_points2.resize(batch_data.size());
        
        for (size_t i = 0; i < batch_data.size(); ++i) {
            const auto& data = batch_data[i];
            Mat flow;
            Mat i1 = data.i1, i2 = data.i2;
            
            // Image preprocessing
            if ((method == "farneback" || method == "tvl1" || method == "deepflow" || 
                 method == "DISflow_ultrafast" || method == "DISflow_fast" || method == "DISflow_medium") 
                 && i1.channels() == 3) {
                cvtColor(i1, i1, COLOR_BGR2GRAY);
                cvtColor(i2, i2, COLOR_BGR2GRAY);
            }
            else if (method == "simpleflow" && i1.channels() == 1) {
                cvtColor(i1, i1, COLOR_GRAY2BGR);
                cvtColor(i2, i2, COLOR_GRAY2BGR);
            }
            
            double startTick = getTickCount();
            
            Ptr<DenseOpticalFlow> algorithm;
            
            if (method == "farneback")
                algorithm = cv::optflow::createOptFlow_Farneback();
            else if (method == "simpleflow")
                algorithm = cv::optflow::createOptFlow_SimpleFlow();
            else if (method == "tvl1")
                algorithm = cv::optflow::createOptFlow_DualTVL1();
            else if (method == "deepflow")
                algorithm = cv::optflow::createOptFlow_DeepFlow();
            else if (method == "sparsetodenseflow")
                algorithm = cv::optflow::createOptFlow_SparseToDense();
            else if (method == "pcaflow")
                algorithm = cv::optflow::createOptFlow_PCAFlow();
            else if (method == "DISflow_ultrafast")
                algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_ULTRAFAST);
            else if (method == "DISflow_fast")
                algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
            else if (method == "DISflow_medium")
                algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
            else if (method == "degraf_flow_lk") {
                FeatureMatcher algo;
                algo.degraf_flow_LK(data.i1, data.i2, flow, 60, (0.05000000075F), true, (500.0F), (1.5F), data.num_str);
                if (display_images) {
                    batch_points1[i] = algo.points_filtered;
                    batch_points2[i] = algo.dst_points_filtered;
                }
            }
            else if (method == "degraf_flow_rlof") {
                FeatureMatcher algo;
                algo.degraf_flow_RLOF(data.i1, data.i2, flow, 127, (0.05000000075F), true, (500.0F), (1.5F), data.num_str);
                if (display_images) {
                    batch_points1[i] = algo.points_filtered;
                    batch_points2[i] = algo.dst_points_filtered;
                }
            }
            else if (method == "degraf_flow_interponet") {
                // Single frame InterpoNet redirects to RLOF
                FeatureMatcher algo;
                algo.degraf_flow_RLOF(data.i1, data.i2, flow, 127, (0.05000000075F), true, (500.0F), (1.5F), data.num_str);
                if (display_images) {
                    batch_points1[i] = algo.points_filtered;
                    batch_points2[i] = algo.dst_points_filtered;
                }
            }
            else {
                printf("Wrong method!\n");
                continue;
            }
            
            if (algorithm.get()) {
                algorithm->calc(i1, i2, flow);
            }
            
            double time = ((double)getTickCount() - startTick) / getTickFrequency();
            
            batch_flows[i] = flow;
            individual_times[i] = time * 1000.0; 
        }
    }
    
    // =====================================================
    // Step 3: Evaluate the calculation
    // =====================================================
    for (size_t i = 0; i < batch_data.size(); ++i) {
        const auto& data = batch_data[i];
        const Mat& flow = batch_flows[i];
        
        if (flow.empty()) {
            printf("Optical flow calculation failed for %06d\n", data.image_no);
            continue;
        }
        
        OptFlowMetrics metrics;
        metrics.image_no = data.image_no;
        metrics.time_ms = individual_times[i];

		Mat kittiFlow = convertToKittiFlow(flow);
		String output_path = "../data/outputs/"+ method+ "/" + data.num_str + "_10.png";
		imwrite(output_path, kittiFlow);
		printf("Saved KITTI flow to: %s\n", output_path.c_str());
		
        
        if (!data.groundtruth_path.empty()) {
            Mat ground_truth = readKittiGroundTruth(data.groundtruth_path);
            
            if (!ground_truth.empty() && 
                flow.size() == ground_truth.size() && 
                flow.channels() == 2 && ground_truth.channels() == 2) {
                
                String error_measure = "endpoint";
                String region = "all";
                
                // Error calculation
                Mat computed_errors;
                if (error_measure == "endpoint")
                    computed_errors = endpointError(flow, ground_truth);
                else if (error_measure == "angular")
                    computed_errors = angularError(flow, ground_truth);
                
                // mask region
                Mat mask;
                if (region == "all")
                    mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
                else if (region == "discontinuities")
                {
                    Mat truth_merged, grad_x, grad_y, gradient;
                    vector<Mat> truth_split;
                    split(ground_truth, truth_split);
                    truth_merged = truth_split[0] + truth_split[1];

                    Sobel(truth_merged, grad_x, CV_16S, 1, 0, -1, 1, 0, BORDER_REPLICATE);
                    grad_x = abs(grad_x);
                    Sobel(truth_merged, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_REPLICATE);
                    grad_y = abs(grad_y);
                    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); // approximation!

                    Scalar s_mean;
                    s_mean = mean(gradient);
                    double threshold = s_mean[0]; // threshold value arbitrary
                    mask = gradient > threshold;
                    dilate(mask, mask, Mat::ones(9, 9, CV_8U));
                }
                else if (region == "untextured")
                {
                    Mat i1_grayscale, grad_x, grad_y, gradient;
                    if (data.i1.channels() == 3)
                        cvtColor(data.i1, i1_grayscale, COLOR_BGR2GRAY);
                    else
                        i1_grayscale = data.i1;
                    Sobel(i1_grayscale, grad_x, CV_16S, 1, 0, 7);
                    grad_x = abs(grad_x);
                    Sobel(i1_grayscale, grad_y, CV_16S, 0, 1, 7);
                    grad_y = abs(grad_y);
                    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); // approximation!
                    GaussianBlur(gradient, gradient, Size(5, 5), 1, 1);

                    Scalar s_mean;
                    s_mean = mean(gradient);
                    // arbitrary threshold value used - could be determined statistically from the image?
                    double threshold = 1000;
                    mask = gradient < threshold;
                    dilate(mask, mask, Mat::ones(3, 3, CV_8U));
                }
                else
                {
                    printf("Invalid region selected! Available options: all, discontinuities, untextured");
                    continue;
                }
                
                // masking out NaNs and incorrect GT values 
                Mat truth_split[2];
                split(ground_truth, truth_split);
                Mat abs_mask = Mat((abs(truth_split[0]) < 1e9) & (abs(truth_split[1]) < 1e9));
                Mat nan_mask = Mat((truth_split[0] == truth_split[0]) & (truth_split[1] == truth_split[1]));
                bitwise_and(abs_mask, nan_mask, nan_mask);
                bitwise_and(nan_mask, mask, mask);
                
                // Calculate statistical indicators
                printf("Using %s error measure\n", error_measure.c_str());
                calculateStats(computed_errors, mask, display_images);
                
                // At the same time, extract indicators to the metrics structure
                calculateStatsForMetrics(computed_errors, mask, metrics);
                

                if (display_images) {
                    Mat im1 = data.i1, im2 = data.i2;
                    vector<Point2f> points1, points2;
                    if (i < batch_points1.size()) {
                        points1 = batch_points1[i];
                        points2 = batch_points2[i];
                    }
                    
                    Mat difference = ground_truth - flow;
                    Mat masked_difference;
                    difference.copyTo(masked_difference, mask);
                    Mat error = flowToDisplay(masked_difference);
                    Mat heatmap = errorHeatMap(difference, mask);
                    Mat ground = flowToDisplay(ground_truth);
                    Mat flow_image = flowToDisplay(flow);

                    // Display all useful output images in one frame 
                    Mat win_mat(Size(data.i1.cols * 2, data.i1.rows * 3), CV_8UC3);

                    imwrite("../data/outputs/010_01.png", data.i1); // 保存第一张图像
                    imwrite("../data/outputs/010_11.png", data.i2); // 保存第二张图像
                    
					// Generate DeGraF feature points visualization 
					if (method == "degraf_flow_lk" || method == "degraf_flow_rlof" || method == "degraf_flow_interponet") {
						Mat feature_points;
						data.i1.copyTo(feature_points); 
						
						// Draw detected feature points (points1) as small circles
						for (int i = 0; i < points1.size(); i++) {
							Scalar color = Scalar(255, 255, 255);				
							Mat overlay;
							feature_points.copyTo(overlay);
							circle(overlay, points1[i], 2, color, -1, cv::LINE_AA, 0);
							addWeighted(feature_points, 0.3, overlay, 0.7, 0, feature_points);
							
						}
						
						// Save feature points visualization
						imwrite("../data/outputs/feature_points.png", feature_points);
						cout << "Detected " << points1.size() << " DeGraF feature points" << endl;
					}
                    // Draw sparse flow field from degraf flow
                    if (method == "degraf_flow_lk" || method == "degraf_flow_rlof" || method == "degraf_flow_interponet") {
                        // Read RAFT dense optical flow as background
						Mat raft_background = imread("../data/outputs/000010_raft_flow.png", IMREAD_COLOR);
						Mat sparse;
						
						if (!raft_background.empty()) {
							// If the size does not match, resize to the original image size
							if (raft_background.size() != data.i1.size()) {
								resize(raft_background, raft_background, data.i1.size());
								cout << "Resized RAFT background to match original image" << endl;
							}
							raft_background.copyTo(sparse);
							cout << "Using RAFT background successfully" << endl;
						} else {
							cout << "RAFT background not found, using fallback" << endl;
							data.i2.copyTo(sparse);
						}

						for (int j = 0; j < points1.size(); j += 4) {
							Mat overlay;
							sparse.copyTo(overlay);
							
							arrowedLine(overlay, points1[j], points2[j], Scalar(255, 255, 255), 2, 8, 0, 0.2);
							addWeighted(sparse, 0.3, overlay, 0.7, 0, sparse);
						}
						
						imwrite("../data/outputs/sparse.png", sparse);
						sparse.copyTo(win_mat(Rect(0, data.i1.rows, data.i1.cols, data.i1.rows)));
                    }
                    else {
                        Mat placeholder = Mat::zeros(data.i1.rows, data.i1.cols, CV_8UC3);
                        bitwise_not(placeholder, placeholder);
                        line(placeholder, Point2f(0, 0), Point2f(data.i1.cols, data.i1.rows), Scalar(0, 0, 0), 2, 8);
                        line(placeholder, Point2f(data.i1.cols, 0), Point2f(0, data.i1.rows), Scalar(0, 0, 0), 2, 8);
                        placeholder.copyTo(win_mat(Rect(0, data.i1.rows, data.i1.cols, data.i1.rows)));
                    }
                    
                    imwrite("../data/outputs/error.png", error);
                    imwrite("../data/outputs/heatmap.png", heatmap);
                    imwrite("../data/outputs/ground.png", ground);
                    imwrite("../data/outputs/flow.png", flow_image);
                    im1.copyTo(win_mat(Rect(0, 0, data.i1.cols, data.i1.rows)));
                    im2.copyTo(win_mat(Rect(data.i1.cols, 0, data.i1.cols, data.i1.rows)));
                    flow_image.copyTo(win_mat(Rect(data.i1.cols, data.i1.rows, data.i1.cols, data.i1.rows)));
                    ground.copyTo(win_mat(Rect(0, data.i1.rows * 2, data.i1.cols, data.i1.rows)));
                    heatmap.copyTo(win_mat(Rect(data.i1.cols, data.i1.rows * 2, data.i1.cols, data.i1.rows)));

                    // Shrink to fit all image on the screen
                    resize(win_mat, win_mat, Size(1325, 600));
                    imshow("Results", win_mat);
                    
                    printf("Using %s error measure\n", error_measure.c_str());
                    calculateStats(computed_errors, mask, display_images);

                    if (display_images)
                        waitKey(0);
                }
            }
        }
        
        results.push_back(metrics);
        all_results_.push_back(metrics);
    }
    
    return results;
}


// Metric calculation functions extracted from calculateStats
void EvaluateOptFlow::calculateStatsForMetrics(Mat errors, Mat mask, OptFlowMetrics& metrics) 
{
    float R_thresholds[] = {0.5f, 1.f, 2.f, 3.f, 5.f, 10.f};
    
    if (mask.empty())
        mask = Mat::ones(errors.size(), CV_8U);
    
    // Calculate mean and standard deviation
    Scalar s_mean, s_std;
    meanStdDev(errors, s_mean, s_std, mask);
    metrics.EPE = (float)s_mean[0];
    metrics.std_dev = (float)s_std[0];
    
    // Calculate RX statistics
    float* R_values[] = {&metrics.R05, &metrics.R1, &metrics.R2, &metrics.R3, &metrics.R5, &metrics.R10};
    for (int i = 0; i < 6; ++i) {
        float R = stat_RX(errors, R_thresholds[i], mask);
        *(R_values[i]) = R * 100;
    }
}


// Clear result function
void EvaluateOptFlow::clearResults() 
{
    all_results_.clear();
    all_stats.clear();
}

void EvaluateOptFlow::exportOpticalFlowTableCSV(
    const std::string &csv_path,
    const std::map<std::string, std::vector<OptFlowMetrics>> &method_results)
{
    std::ofstream file(csv_path, std::ios::trunc);
    if (!file.is_open())
        return;

    // Export only metrics that are actually computed in this project.
    file << "Method,Average EPE,Average R2.0,Average R3.0,Average Time(ms),Average STD\n";

    for (const auto &method_pair : method_results)
    {
        const std::string &method_name = method_pair.first;
        const std::vector<OptFlowMetrics> &results = method_pair.second;
        if (results.empty())
            continue;

        double avg_epe = 0.0, avg_r2 = 0.0, avg_r3 = 0.0, avg_time = 0.0, avg_std = 0.0;
        for (const auto &m : results)
        {
            avg_epe += m.EPE;
            avg_r2 += m.R2;
            avg_r3 += m.R3;
            avg_time += m.time_ms;
            avg_std += m.std_dev;
        }

        const double n = static_cast<double>(results.size());
        file << method_name << ","
             << (avg_epe / n) << ","
             << (avg_r2 / n) << ","
             << (avg_r3 / n) << ","
             << (avg_time / n) << ","
             << (avg_std / n) << "\n";
    }
}


// Backwards compatible with the original runEvaluation function
int EvaluateOptFlow::runEvaluation(String method, bool display_images, int image_no) 
{
    std::vector<int> indices = {image_no};
    std::vector<OptFlowMetrics> results = runEvaluation(method, display_images, indices);
    
    if (!results.empty()) {
        OptFlowMetrics result = results[0];
        
        stats_vector.clear();
        stats_vector.push_back(image_no);
        stats_vector.push_back(result.EPE);
        stats_vector.push_back(result.std_dev);
        stats_vector.push_back(result.R05);
        stats_vector.push_back(result.R1);
        stats_vector.push_back(result.R2);
        stats_vector.push_back(result.R3);
        stats_vector.push_back(result.R5);
        stats_vector.push_back(result.R10);
        stats_vector.push_back(result.time_ms);
        
        all_stats.push_back(stats_vector);
        stats_vector.clear();
    }
    
    return 0;
}