#include "EvaluateSceneFlow.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <cstdlib>

using namespace cv;
using namespace std;

static std::string getDataSceneFlowRoot()
{
    const char *env_path = std::getenv("DEGRAF_DATA_PATH");
    if (env_path && std::string(env_path).size() > 0)
        return std::string(env_path);
    return "/root/autodl-tmp/data/kitti/data_scene_flow";
}

static std::string getDataSceneFlowCalibRoot()
{
    const char *env_path = std::getenv("DEGRAF_CALIB_PATH");
    if (env_path && std::string(env_path).size() > 0)
        return std::string(env_path);
    return "/root/autodl-tmp/data/kitti/data_scene_flow_calib";
}

EvaluateSceneFlow::EvaluateSceneFlow() {}


/**
 * Function: Parse fx, fy, cx, cy, baseline from KITTI calib_cam_to_cam/*.txt files
 * Input parameters:
 * calib_file: file path
 * fx, fy, cx, cy, baseline: reference variables, write parsed values
 * Output: true means successful parsing, false means failure or incorrect format
 */
static bool loadCameraIntrinsics(const std::string &calib_file, float &fx, float &fy, float &cx, float &cy, float &baseline)
{
    std::ifstream file(calib_file);
    if (!file.is_open())
    {
        return false;
    }

    std::string line;
    std::vector<float> P2_values;
    std::vector<float> P3_values;

    while (std::getline(file, line))
    {
        if (line.find("P_rect_02:") != std::string::npos)
        {
            std::istringstream iss(line.substr(11)); // Skip label
            float val;
            while (iss >> val)
            {
                P2_values.push_back(val);
            }
        }
        else if (line.find("P_rect_03:") != std::string::npos)
        {
            std::istringstream iss(line.substr(11));
            float val;
            while (iss >> val)
            {
                P3_values.push_back(val);
            }
        }
    }
    file.close();

    if (P2_values.size() < 12 || P3_values.size() < 12)
    {
        return false;
    }

    // From P2 parse intrinsic parameters
    fx = P2_values[0]; // P2[0][0]
    fy = P2_values[5]; // P2[1][1]
    cx = P2_values[2]; // P2[0][2]
    cy = P2_values[6]; // P2[1][2]

    // From P2[0][3] 和 P3[0][3] calculate baseline
    float Tx2 = P2_values[3]; // P2[0][3]
    float Tx3 = P3_values[3]; // P3[0][3]
    baseline = -(Tx3 - Tx2) / fx;

    return true;
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

/**
 * @brief Read disparity GT file according to KITTI official standard
 * @param path disparity file path (such as disp_noc_0/000000_10.png)
 * @return CV_32FC1 disparity map, invalid pixels are NaN
 */
static cv::Mat readKITTIDisparity(const std::string &path)
{
    // 1. Read the original uint16 image
    cv::Mat disp_raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (disp_raw.empty())
    {
        printf("❌ Cannot open disparity file: %s\n", path.c_str());
        return cv::Mat();
    }

    // 2. Check type: KITTI disparity should be CV_16UC1
    if (disp_raw.type() != CV_16UC1)
    {
        printf("❌ Invalid disparity image type (expect CV_16UC1, got %d)\n", disp_raw.type());
        return cv::Mat();
    }

    // 3. Convert according to KITTI standard
    cv::Mat disp_f32(disp_raw.size(), CV_32F);
    for (int y = 0; y < disp_raw.rows; ++y)
    {
        for (int x = 0; x < disp_raw.cols; ++x)
        {
            uint16_t raw_val = disp_raw.at<uint16_t>(y, x);
            if (raw_val == 0)
            {
                //  KITTI standard: 0 value indicates invalid pixel
                disp_f32.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                //  KITTI standard: divide by 256 to get true disparity value
                disp_f32.at<float>(y, x) = static_cast<float>(raw_val) / 256.0f;
            }
        }
    }
    return disp_f32;
}

/**
 * @brief Read optical flow GT file according to KITTI official standard
 * @param path Optical flow file path (such as flow_noc/000000_10.png or flow_occ/000000_10.png)
 * @return CV_32FC2 optical flow map, invalid pixels are NaN
 */
static cv::Mat readKITTIFlowGT(const std::string &ground_truth_path)
{
    String path = ground_truth_path;
    // NB opencv has order BGR => valid , v , u
    Mat image = imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    Mat gt = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
    int width = image.cols;
    int height = image.rows;
    int valid_pixels = 0; // Add statistics

    for (int32_t v = 0; v < height; v++)
    {
        for (int32_t u = 0; u < width; u++)
        {
            Vec3s val = image.at<Vec3s>(v, u);
            if (val[0] > 0) // validity check
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
                valid_pixels++; // Count valid pixels
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

    // Add statistics output
    printf("Flow GT statistics: %d/%d (%.2f%%) valid pixels\n",
           valid_pixels, width * height, 100.0f * valid_pixels / (width * height));

    return gt;
}


// Standard indicator calculation
SceneFlowMetrics EvaluateSceneFlow::calculateStandardMetrics(const cv::Mat &pred_scene_flow,
                                                             const cv::Mat &gt_scene_flow)
{
    SceneFlowMetrics metrics;

    int total_gt_valid = 0;
    int evaluable_points = 0;

    // Add failure area analysis variables
    std::vector<float> missing_region_magnitudes;
    std::vector<float> covered_region_magnitudes;

    double total_epe = 0.0;
    int acc_strict_count = 0;
    int acc_relax_count = 0;
    int outlier_count = 0;

    for (int v = 0; v < pred_scene_flow.rows; ++v)
    {
        for (int u = 0; u < pred_scene_flow.cols; ++u)
        {
            cv::Vec3f pred = pred_scene_flow.at<cv::Vec3f>(v, u);
            cv::Vec3f gt = gt_scene_flow.at<cv::Vec3f>(v, u);

            // Check GT validity
            bool gt_valid = (!std::isnan(gt[0]) && !std::isnan(gt[1]) && !std::isnan(gt[2]) &&
                             !(gt[0] == 0 && gt[1] == 0 && gt[2] == 0));

            if (gt_valid)
            {
                total_gt_valid++;

                // Calculate GT amplitude
                float gt_magnitude = cv::norm(gt);

                // Check the validity of pred
                bool pred_valid = (!std::isnan(pred[0]) && !std::isnan(pred[1]) && !std::isnan(pred[2]) &&
                                   !(pred[0] == 0 && pred[1] == 0 && pred[2] == 0));

                if (pred_valid)
                {
                    evaluable_points++;
                    covered_region_magnitudes.push_back(gt_magnitude); //Record the coverage area amplitude

                    float dx = pred[0] - gt[0];
                    float dy = pred[1] - gt[1];
                    float dz = pred[2] - gt[2];
                    float epe = std::sqrt(dx * dx + dy * dy + dz * dz);

                    total_epe += epe;

                    float gt_norm = cv::norm(gt);
                    float rel_err = epe / (gt_norm + 1e-6f);

                    if (epe < 0.05f || rel_err < 0.05f)
                        acc_strict_count++;
                    if (epe < 0.1f || rel_err < 0.1f)
                        acc_relax_count++;
                    if (epe > 0.3f || rel_err > 0.1f)
                        outlier_count++;
                }
                else
                {
                    missing_region_magnitudes.push_back(gt_magnitude); //Record the amplitude of the failure area
                }
            }
        }
    }

    // Failure area analysis
    if (!missing_region_magnitudes.empty() && !covered_region_magnitudes.empty())
    {
        float avg_missing = std::accumulate(missing_region_magnitudes.begin(),
                                            missing_region_magnitudes.end(), 0.0f) /
                            missing_region_magnitudes.size();
        float avg_covered = std::accumulate(covered_region_magnitudes.begin(),
                                            covered_region_magnitudes.end(), 0.0f) /
                            covered_region_magnitudes.size();

        printf("=== Regional Analysis ===\n");
        printf("Coverage area: %zu px, Average GT amplitude: %.4f m\n",
               covered_region_magnitudes.size(), avg_covered);
        printf("Failure area: %zu px, Average GT amplitude: %.4f m\n",
               missing_region_magnitudes.size(), avg_missing);

        if (avg_missing > avg_covered)
        {
            printf("Failure area is more difficult than coverage area (large movement)\n");
        }
        else
        {
            printf("Failure area is simpler than coverage area (small movement)\n");
        }
        printf("==================\n");
    }

    if (evaluable_points > 0)
    {
        metrics.EPE3d = total_epe / evaluable_points;
        metrics.AccS = 100.0 * acc_strict_count / evaluable_points;
        metrics.AccR = 100.0 * acc_relax_count / evaluable_points;
        metrics.Outlier = 100.0 * outlier_count / evaluable_points;
        metrics.valid_count = evaluable_points;
    }

    return metrics;
}

// Core single frame evaluation function
SceneFlowMetrics EvaluateSceneFlow::evaluateSingleFrame(const cv::Mat &pred_scene_flow,
                                                        const cv::Mat &gt_scene_flow,
                                                        bool verbose)
{
    SceneFlowMetrics metrics = calculateStandardMetrics(pred_scene_flow, gt_scene_flow);

    if (verbose)
    {
        printf("EPE3d: %.4f | AccS: %.2f%% | AccR: %.2f%% | Outlier: %.2f%% | Valid: %d\n",
               metrics.EPE3d, metrics.AccS, metrics.AccR, metrics.Outlier, metrics.valid_count);
    }

    return metrics;
}

// Simplifying CSV writing
void EvaluateSceneFlow::writeMetricsToCSV(const SceneFlowMetrics &metrics,
                                          const std::string &method,
                                          int image_no,
                                          const std::string &csv_path)
{
    std::ofstream file(csv_path, std::ios::app);
    if (!file.is_open())
    {
        printf("❌ Failed to open CSV: %s\n", csv_path.c_str());
        return;
    }

    // Write in standard format
    file << image_no << "," << method << ","
         << metrics.EPE3d << "," << metrics.AccS << "," << metrics.AccR << ","
         << metrics.Outlier << "," << metrics.valid_count << "," << metrics.time_ms << "\n";

    file.close();
}



void EvaluateSceneFlow::exportSceneFlowComparisonCSV(
    const std::string& csv_path,
    const std::map<std::string, std::vector<SceneFlowMetrics>>& method_results)
{
    std::ofstream file(csv_path, std::ios::trunc);
    if (!file.is_open()) return;
    
    file << "Method,EPE3d(m),AccS(%),AccR(%),Outlier(%),Runtime(ms)\n";
    
    
    for (const auto& method_pair : method_results) {
        const std::string& method_name = method_pair.first;
        const std::vector<SceneFlowMetrics>& results = method_pair.second;
        
        if (!results.empty()) {
            // Calculate the average
            double avg_EPE3d = 0, avg_AccS = 0, avg_AccR = 0, avg_Outlier = 0, avg_time = 0;
            for (const auto& metrics : results) {
                avg_EPE3d += metrics.EPE3d;
                avg_AccS += metrics.AccS;
                avg_AccR += metrics.AccR;
                avg_Outlier += metrics.Outlier;
                avg_time += metrics.time_ms;
            }
            size_t count = results.size();
            
            // Write the average value of this method
            file << method_name << "+Disp," << avg_EPE3d/count << "," << avg_AccS/count << "," 
                 << avg_AccR/count << "," << avg_Outlier/count << "," << avg_time/count << "\n";
        }
    }
    file.close();
}

/**
* @brief Unified scene flow evaluation entry point function - replaces the original runEvaluation and runEvaluationBatch
* @param method Optical flow method name
* @param display_images Display visualization function
* @param image_indices Image index array (for single frame, pass {i}; for batch, pass {i1,i2,...})
* @return Evaluation result array
*/
std::vector<SceneFlowMetrics> EvaluateSceneFlow::runEvaluation(
    const std::string &method,
    bool display_images,
    const std::vector<int> &image_indices)
{
    std::vector<SceneFlowMetrics> results;
    
    // Determine whether true batch processing is supported
    bool is_batch_capable = (method == "degraf_flow_interponet");
    
    // =====================================================
    // Step 1: Data preparation
    // =====================================================
    struct ImagePairData {
        cv::Mat i1, i2;
        cv::Mat gray1, gray2;
        std::string i1_path, i2_path;
        std::string disp0_path, disp1_path;
        std::string flow_gt_path;
        std::string calib_path;
        std::string num_str;
        float fx, fy, cx, cy, baseline;
        int image_no;
    };
    
    std::vector<ImagePairData> batch_data;
    batch_data.reserve(image_indices.size());
    
    // load all data
    for (int image_no : image_indices) {
        ImagePairData data;
        data.image_no = image_no;
        
        // Path construction
        char num[7];
        sprintf(num, "%06d", image_no);
        data.num_str = std::string(num);
        
        std::string base_dir = getDataSceneFlowRoot() + "/training/";
        data.i1_path = base_dir + "image_2/" + data.num_str + "_10.png";
        data.i2_path = base_dir + "image_2/" + data.num_str + "_11.png";
        data.disp0_path = base_dir + "disp_noc_0/" + data.num_str + "_10.png";
        data.disp1_path = base_dir + "disp_noc_1/" + data.num_str + "_10.png";
        data.flow_gt_path = base_dir + "flow_noc/" + data.num_str + "_10.png";
        data.calib_path = getDataSceneFlowCalibRoot() + "/training/calib_cam_to_cam/" + data.num_str + ".txt";
        
        // load the image
        data.i1 = cv::imread(data.i1_path, 1);
        data.i2 = cv::imread(data.i2_path, 1);
        cv::Mat disp0 = cv::imread(data.disp0_path, cv::IMREAD_UNCHANGED);
        cv::Mat disp1 = cv::imread(data.disp1_path, cv::IMREAD_UNCHANGED);
        cv::Mat flow_gt = cv::imread(data.flow_gt_path, cv::IMREAD_UNCHANGED);
        
        // Verify data
        if (data.i1.empty() || data.i2.empty() || disp0.empty() || disp1.empty() || flow_gt.empty()) {
            printf("❌ Input missing %06d\n", image_no);
            continue;
        }
        if (data.i1.size() != data.i2.size() || data.i1.size() != disp0.size() || 
            disp0.size() != disp1.size() || data.i1.size() != flow_gt.size()) {
            printf("❌ Size mismatch %06d\n", image_no);
            continue;
        }
        
        // Preprocess grayscale image
        if (data.i1.channels() == 3)
            cv::cvtColor(data.i1, data.gray1, cv::COLOR_BGR2GRAY);
        else
            data.gray1 = data.i1.clone();
        
        if (data.i2.channels() == 3)
            cv::cvtColor(data.i2, data.gray2, cv::COLOR_BGR2GRAY);
        else
            data.gray2 = data.i2.clone();
        
        // Load calibration parameters
        if (!loadCameraIntrinsics(data.calib_path, data.fx, data.fy, data.cx, data.cy, data.baseline)) {
            data.fx = 721.5377f;
            data.fy = 721.5377f;
            data.cx = 609.5593f;
            data.cy = 172.8540f;
            data.baseline = 0.5371f;
            printf("Using default KITTI camera parameters for %06d\n", image_no);
        }
        
        batch_data.push_back(data);
    }
    
    if (batch_data.empty()) {
        return results;
    }
    
    // =====================================================
    // Step 2: Optical flow calculation (select batch or frame by frame depending on the method)
    // =====================================================
    std::vector<cv::Mat> batch_flows;
    std::vector<double> individual_times;
    
    if (is_batch_capable && batch_data.size() > 1) {
        // InterpoNet batch processing
        std::vector<cv::Mat> batch_i1, batch_i2;
        std::vector<std::string> batch_num_strs;
        
        for (const auto& data : batch_data) {
            batch_i1.push_back(data.i1);
            batch_i2.push_back(data.i2);
            batch_num_strs.push_back(data.num_str);
        }
        
        double batch_start = cv::getTickCount();
        FeatureMatcher matcher;
        
        // Call the batch version to get feature points for visualization (if necessary)
        std::vector<std::vector<cv::Point2f>> batch_points, batch_dst_points;
        batch_flows = matcher.degraf_flow_InterpoNet(
            batch_i1, batch_i2, batch_num_strs,
            display_images ? &batch_points : nullptr,      
            display_images ? &batch_dst_points : nullptr
        );
        
        double total_time_ms = (cv::getTickCount() - batch_start) / cv::getTickFrequency() * 1000.0;
        
        // When batch processing, the time per frame is the average time
        for (size_t i = 0; i < batch_flows.size(); ++i) {
            individual_times.push_back(total_time_ms / batch_flows.size());
        }
    } else {
        // Process frame by frame (can be parallelized)
        batch_flows.resize(batch_data.size());
        individual_times.resize(batch_data.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < batch_data.size(); ++i) {
            const auto& data = batch_data[i];
            cv::Mat flow;
            
            double single_start = cv::getTickCount();
            
            // Optical flow estimation
            if (method == "farneback")
                cv::optflow::createOptFlow_Farneback()->calc(data.gray1, data.gray2, flow);
            else if (method == "tvl1")
                cv::optflow::createOptFlow_DualTVL1()->calc(data.gray1, data.gray2, flow);
            else if (method == "deepflow")
                cv::optflow::createOptFlow_DeepFlow()->calc(data.gray1, data.gray2, flow);
            else if (method == "DISflow_fast")
                cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST)->calc(data.gray1, data.gray2, flow);
            else if (method == "DISflow_medium")
                cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM)->calc(data.gray1, data.gray2, flow);
            else if (method == "degraf_flow_rlof") {
                FeatureMatcher matcher;
                matcher.degraf_flow_RLOF(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            }
            else if (method == "degraf_flow_lk") {
                FeatureMatcher matcher;
                matcher.degraf_flow_LK(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            }
            else if (method == "degraf_flow_interponet") {
                printf("Note: Single frame InterpoNet redirected to RLOF for better efficiency\n");
                FeatureMatcher matcher;
                matcher.degraf_flow_RLOF(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            }
            else {
                printf("❌ Unknown optical flow method: %s\n", method.c_str());
            }
            
            double single_time = (cv::getTickCount() - single_start) / cv::getTickFrequency() * 1000.0;
            
            #pragma omp critical
            {
                batch_flows[i] = flow;
                individual_times[i] = single_time;
            }
        }
    }
    
    // =====================================================
    // Step 3: Scene flow reconstruction and evaluation
    // =====================================================
    for (size_t i = 0; i < batch_data.size(); ++i) {
        const auto& data = batch_data[i];
        const cv::Mat& flow = batch_flows[i];
        
        if (flow.empty()) {
            printf("❌ Optical flow calculation failed %06d\n", data.image_no);
            continue;
        }
        
        // Scene flow reconstruction
        cv::Mat disp0_f32 = readKITTIDisparity(data.disp0_path);
        cv::Mat disp1_f32 = readKITTIDisparity(data.disp1_path);
        
        SceneFlowReconstructor reconstructor(data.fx, data.fy, data.cx, data.cy, data.baseline);
        cv::Mat scene_flow = reconstructor.computeSceneFlow(flow, disp0_f32, disp1_f32);
        
        if (scene_flow.empty()) {
            printf("Scene flow calculation failed %06d\n", data.image_no);
            continue;
        }
        
        // Read GT
        cv::Mat flow_gt_processed = readKITTIFlowGT(data.flow_gt_path);
        cv::Mat gt_scene_flow = reconstructor.computeSceneFlow(flow_gt_processed, disp0_f32, disp1_f32);
        
        if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3) {
            printf("GT Scene Flow reading failed %06d\n", data.image_no);
            continue;
        }
        

        SceneFlowMetrics metrics = evaluateSingleFrame(scene_flow, gt_scene_flow, true);
        metrics.time_ms = individual_times[i];

        results.push_back(metrics);
        all_results_.push_back(metrics);
        
        printf("Frame %06d evaluated successfully\n", data.image_no);
    }

    return results;
}

/**
* @brief Convenience overload - single-frame interface for backward compatibility
*/
SceneFlowMetrics EvaluateSceneFlow::runEvaluation(
    const std::string &method,
    bool display_images,
    int image_no)
{
    std::vector<int> indices = {image_no};
    std::vector<SceneFlowMetrics> results = runEvaluation(method, display_images, indices);
    
    if (!results.empty()) {
        return results[0];
    } else {
        return SceneFlowMetrics();  // Returns an empty result of the default construction
    }
}