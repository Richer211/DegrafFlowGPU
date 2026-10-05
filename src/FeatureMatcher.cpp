/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow and InterpoNet Optical flow 
\author Gang Wang
*/

#include "stdafx.h"
#include "FeatureMatcher.h"
#include <chrono>         
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "inference/RaftEngineTRT.h"
#include "inference/InterpoNetEngineTRT.h"
#if USE_CUDA
#include "degraf_detector.h"
#include <cuda_runtime.h>
#endif

namespace {
inline bool envInt(const char *name, int &out_value)
{
    const char *value = std::getenv(name);
    if (!value || std::string(value).empty())
        return false;
    try
    {
        out_value = std::stoi(value);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

inline std::string envOrDefault(const char *name, const std::string &fallback)
{
    const char *value = std::getenv(name);
    if (value && std::string(value).size() > 0)
        return std::string(value);
    return fallback;
}

inline bool envEnabled(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    std::string v(value);
    for (char &c : v)
        c = static_cast<char>(std::tolower(c));
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

inline bool loadPointsFile(const std::string &path, std::vector<cv::Point2f> &points)
{
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open())
        return false;
    points.clear();
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        float x = 0.0f, y = 0.0f;
        if (!(iss >> x >> y))
            continue;
        points.emplace_back(x, y);
    }
    return !points.empty();
}

inline bool loadMatchesFile(const std::string &path, SparseFlowMatches &matches)
{
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open())
        return false;
    matches.src_points.clear();
    matches.dst_points.clear();
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
        if (!(iss >> x1 >> y1 >> x2 >> y2))
            continue;
        matches.src_points.emplace_back(x1, y1);
        matches.dst_points.emplace_back(x2, y2);
    }
    return !matches.src_points.empty() && matches.src_points.size() == matches.dst_points.size();
}

inline std::string buildIndexedPath(const std::string &dir, const std::string &num_str, const std::string &suffix)
{
    if (dir.empty())
        return "";
    return cv::format("%s/%s%s", dir.c_str(), num_str.c_str(), suffix.c_str());
}

inline void detectDegrafPointsForInterpoNet(
    const cv::Mat &img,
    int saliency_kernel,
    int degraf_window_w,
    int degraf_window_h,
    int degraf_step_x,
    int degraf_step_y,
    std::vector<cv::Point2f> &points)
{
    points.clear();
    cv::Size s = img.size();

    IplImage *fromIpl = cvCreateImageHeader(cvSize(img.cols, img.rows), IPL_DEPTH_8U, img.channels());
    cvSetData(fromIpl, const_cast<uchar *>(img.data), img.step);
    cv::Mat dogMat = cv::Mat::zeros(s.height, s.width, CV_8UC3);
    IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels()); // 创建dogIpl图像
    cvSetData(dogIpl, dogMat.data, dogMat.step);
    /**
	 * saliency_mat 是送进检测器的那张图，尺寸和第 10 帧一样，但格子里不是照片的原始亮度。
	 * DoGoS_Saliency 先把第 10 帧转成灰度，再和它的模糊版本做差。
	 * 结构清楚的地方在这张图上更亮，平坦的地方更暗。窗口里算质心时用的「亮度」，就是这张图上的数。
	 */
    SaliencyDetector saliency_detector;
    saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, saliency_kernel, true, true);
    saliency_detector.Release();

    cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);
#if USE_CUDA 
    CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
    gpu_gradient_detector->CudaDetectGradients( 
        saliency_mat, degraf_window_w, degraf_window_h, degraf_step_x, degraf_step_y);
    cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
    delete gpu_gradient_detector;
#else
    GradientDetector *cpu_gradient_detector = new GradientDetector();
    cpu_gradient_detector->DetectGradients(
        dogIpl, degraf_window_w, degraf_window_h, degraf_step_x, degraf_step_y);
    cv::KeyPoint::convert(cpu_gradient_detector->keypoints, points);
    delete cpu_gradient_detector;
#endif

    const int MAX_POINTS = 50000;
    if (points.size() > MAX_POINTS)
    {
        std::vector<cv::Point2f> subsampled_points;
        subsampled_points.reserve(MAX_POINTS);
        int step_sample = static_cast<int>(points.size()) / MAX_POINTS;
        step_sample = std::max(1, step_sample);
        for (size_t i = 0; i < points.size(); i += static_cast<size_t>(step_sample))
        {
            subsampled_points.push_back(points[i]);
            if (subsampled_points.size() >= MAX_POINTS)
                break;
        }
        points = std::move(subsampled_points);
    }

    cvReleaseImageHeader(&fromIpl);
    cvReleaseImageHeader(&dogIpl);
}

inline cv::Ptr<cv::Feature2D> createSiftDetector(int nfeatures, int nOctaveLayers,
                                                  double contrastThreshold, double edgeThreshold,
                                                  double sigma)
{
#if defined(CV_VERSION_MAJOR) && ((CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 4))
	return cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
	return cv::ORB::create(std::max(nfeatures, 500), 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
#endif
}
}

// Constructor
FeatureMatcher::FeatureMatcher()
{
}

// DeGraF-Flow using lucas-kanade point tracking
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/

void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));

	int64 timeStart0 = getTickCount();
	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	vector<Point2f> dst_points;
	vector<unsigned char> status;
	vector<float> err;

	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0)
	{
		cv::Size s = from.size();

		// Convert cv::Mat to IplImage for legacy SaliencyDetector
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 5, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();

		int status_1 = gradient_detector_1->DetectGradients(dogIpl, 3, 3, 7, 7);

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);

		// Clean up IplImage headers
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
	}
	// Using other point detectors
	else if (point == 1)
	{
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints, 2, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(0, 2, 0.01, 10.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(7000, 3, 0.00, 100.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4)
	{
		vector<KeyPoint> keypoints;
		Ptr<ORB> detector = ORB::create(5400, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 5)
	{
		vector<KeyPoint> keypoints;
		Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(5, true, cv::AgastFeatureDetector::OAST_9_16);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 6)
	{

		for (int i = 0; i < prev.rows - 7; i = i + 7)
		{
			for (int j = 0; j < prev.cols - 7; j = j + 7)
			{
				points.push_back(Point2f(j, i));
			}
		}
	}

	// Lucas-Kanade point tracking
	cv::calcOpticalFlowPyrLK(prev_grayscale, cur_grayscale, points, dst_points, status, err, Size(11, 11), 4);

	// Set max vector length allowed (in pixels) N.B change max vector length for different data sets.
	int max_flow_length = 100;

	// long double execTime1 = (getTickCount()*1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	for (unsigned int i = 0; i < points.size(); i++)
	{
		if (status[i] != 0 &&
			sqrt(pow(points[i].x - dst_points[i].x, 2) + pow(points[i].y - dst_points[i].y, 2)) < max_flow_length &&
			dst_points[i].x >= 0 && dst_points[i].x < cur.cols && dst_points[i].y < cur.rows && dst_points[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(dst_points[i]);
		}
	}

	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();

	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k);
	gd->setSigma(sigma);
	gd->setUsePostProcessing(use_post_proc);
	gd->setFGSLambda(fgs_lambda);
	gd->setFGSSigma(fgs_sigma);

	if (points_filtered.size() > SHRT_MAX)
	{
		cout << "Too many points to interpolate";
	}

	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);

}

// DeGraF-Flow using Robust Local Optical Flow point tracking, requires RLOF code found at https://github.com/tsenst/RLOFLib
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/
void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));

	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	// vector<unsigned char> status;
	vector<float> err;
	int64 timeStart0 = getTickCount();
	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0)
	{
		cv::Size s = from.size();

		// Convert cv::Mat to IplImage for legacy SaliencyDetector
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);
		std::cout << "About to call cvSetData for image_8u" << std::endl;
		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();
		int status_1 = gradient_detector_1->DetectGradients(dogIpl, 3, 3, 9, 9); // DeGraF params specified here

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);

		std::cout << "CPU Detection completed successfully!" << std::endl;
		std::cout << "CPU Keypoints detected: " << points.size() << std::endl;
		
		// Limit the number of feature points to avoid EPIC interpolator overflow
		const int MAX_POINTS = 50000;  
		if (points.size() > MAX_POINTS) {
			std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS 
					<< " for EPIC compatibility" << std::endl;
			
			// Uniform subsampling strategy
			std::vector<cv::Point2f> subsampled_points;
			subsampled_points.reserve(MAX_POINTS);
			
			int step_sample = points.size() / MAX_POINTS;
			for (size_t i = 0; i < points.size(); i += step_sample) {
				subsampled_points.push_back(points[i]);
				if (subsampled_points.size() >= MAX_POINTS) break;
			}
			
			points = subsampled_points;
			std::cout << "Final points for RLOF: " << points.size() << std::endl;
		}
		// Clean up IplImage headers
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
	}
	// Using other point detectors
	else if (point == 1)
	{
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints, 10, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(0, 2, 0.01, 10.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3)
	{
		vector<KeyPoint> keypoints;
		Ptr<ORB> detector = ORB::create(5400, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4)
	{
		vector<KeyPoint> keypoints;
		Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(16, true, cv::AgastFeatureDetector::OAST_9_16);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}

	else if (point == 5)
	{
		for (int i = 0; i < prev.rows - 6; i = i + 6)
		{
			for (int j = 0; j < prev.cols - 6; j = j + 6)
			{
				points.push_back(Point2f(j, i));
			}
		}
	}
	else if (point == 6)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(5400, 3, 0.00, 100.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 7) {
#if USE_CUDA
		// GPU-accelerated DeGraF detection
		cv::Size s = from.size();
		
		// Convert cv::Mat to IplImage for legacy SaliencyDetector (same as CPU version)
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);
		
		std::cout << "About to call GPU DeGraF saliency detection" << std::endl;
		
		// Saliency detection (same as CPU version)
		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();
		
		// Convert IplImage back to cv::Mat for GPU detector
		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);
		
		std::cout << "Starting GPU DeGraF gradient detection..." << std::endl;
		auto gpu_start = std::chrono::high_resolution_clock::now();
		
		// GPU DeGraF detection
		CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		
		try {
			int gpu_status = gpu_gradient_detector->CudaDetectGradients(saliency_mat, 3, 3, 9, 9); // Same DeGraF params
			
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			
			if (gpu_status) {
				// Convert GPU keypoints to Point2f (same interface as CPU version)
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);

				// Limit the number of feature points to avoid EPIC interpolator overflow
				const int MAX_POINTS = 50000;  
				if (points.size() > MAX_POINTS) {
					std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS 
							<< " for EPIC compatibility" << std::endl;
					
					// Uniform subsampling strategy
					std::vector<cv::Point2f> subsampled_points;
					subsampled_points.reserve(MAX_POINTS);
					
					int step_sample = points.size() / MAX_POINTS;
					for (size_t i = 0; i < points.size(); i += step_sample) {
						subsampled_points.push_back(points[i]);
						if (subsampled_points.size() >= MAX_POINTS) break;
					}
					
					points = subsampled_points;
					std::cout << "Final points for RLOF: " << points.size() << std::endl;
				}
				
			}
			
		} catch (const std::exception& e) {
			std::cerr << "GPU DeGraF exception: " << e.what() << std::endl;
			std::cerr << "Falling back to CPU version..." << std::endl;
			
			// Fallback to CPU version
			GradientDetector *cpu_gradient_detector = new GradientDetector();
			int cpu_status = cpu_gradient_detector->DetectGradients(dogIpl, 3, 3, 9, 9);
			
			if (cpu_status) {
				cv::KeyPoint::convert(cpu_gradient_detector->keypoints, points);
				std::cout << "CPU fallback successful, keypoints: " << points.size() << std::endl;
			}
			
			delete cpu_gradient_detector;
		}
		
		// Clean up GPU detector
		delete gpu_gradient_detector;
		
		// Clean up IplImage headers (same as CPU version)
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
		
		std::cout << "GPU DeGraF detection phase completed" << std::endl;
#else
		std::cout << "GPU DeGraF path skipped: USE_CUDA=0" << std::endl;
#endif
	}

	long double execTime0 = (getTickCount() * 1.0000 - timeStart0) / (getTickFrequency() * 1.0000);
	std::cout << "Time to compute DeGraF points = " << execTime0 << "\n\n";

	//////////////////////////////// RLOF ////////////////////////////////////////////////////////////////

	int64 timeStart1 = getTickCount();

	
	Mat img0 = prev;
	Mat img1 = cur;


	std::vector<Point2f> prevPoints, currPoints;
	for (int r = 0; r < points.size(); r++)
	{
		prevPoints.push_back(Point2f(points[r].x, points[r].y));
	}


	currPoints.resize(prevPoints.size()); 
	std::vector<uchar> status(prevPoints.size()); 
	std::vector<float> error(prevPoints.size()); 

	// Wrap the parameter structure with a pointer
	cv::Ptr<cv::optflow::RLOFOpticalFlowParameter> rlof_param = 
		cv::makePtr<cv::optflow::RLOFOpticalFlowParameter>();

	rlof_param->useIlluminationModel = true; // 使用光照模型
	rlof_param->useGlobalMotionPrior = true; // 使用全局运动先验
	rlof_param->smallWinSize = 10; // 小窗口大小
	rlof_param->largeWinSize = 11; // 大窗口大小
	rlof_param->maxLevel = 4; // 最大金字塔层数
	rlof_param->maxIteration = 30; // 最大迭代次数
	rlof_param->supportRegionType = cv::optflow::SR_FIXED;
	// Create RLOF optical flow estimator
	cv::Ptr<cv::optflow::SparseRLOFOpticalFlow> proc =
		cv::optflow::SparseRLOFOpticalFlow::create(rlof_param);

	std::cout << "points size: " << points.size() << std::endl;
	std::cout << "prevPoints size: " << prevPoints.size() << std::endl;
	std::cout << "currPoints size before calc: " << currPoints.size() << std::endl;
	// Check the number of input image channels
	std::cout << "Input images - prev channels: " << prev_grayscale.channels() 
			<< ", cur channels: " << cur_grayscale.channels() << std::endl;
	try
	{
		proc->calc(prev_grayscale, cur_grayscale, prevPoints, currPoints, status, error);
		std::cout << "RLOF calculation completed successfully" << std::endl;
	}
	catch (cv::Exception &e)
	{
		std::cout << "OpenCV RLOF Error: " << e.what() << std::endl;
		return; 
	}

	dst_points_filtered.clear();
	points_filtered.clear();
	dst_points_filtered.shrink_to_fit(); 
	points_filtered.shrink_to_fit();

	int max_flow_length = 100;
	for (unsigned int i = 0; i < points.size() && i < currPoints.size(); i++)
	{
		// points[i] 是起点，currPoints[i] 是 RLOF 找到的终点。只有 status 通过、位移没超过 100 像素、而且两点都在图里，才加进去：
		if (status[i] && 
			sqrt(pow(points[i].x - currPoints[i].x, 2) + pow(points[i].y - currPoints[i].y, 2)) < max_flow_length &&
			currPoints[i].x >= 0 && currPoints[i].x < cur.cols && currPoints[i].y < cur.rows && currPoints[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			// points_filtered 就是起点名单，dst_points_filtered 就是终点名单。
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(currPoints[i]); 
		}
	}

	long double execTime1 = (getTickCount() * 1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	std::cout << "Time to run RLOF = " << execTime1 << "\n\n";
	std::cout << "Filtered points: " << points_filtered.size() << " out of " << points.size() << std::endl;
	////////////////////////////////   Interpolation  //////////////////////////////////////////////////////////////////

	int64 timeStart2 = getTickCount();

	if (points_filtered.size() > SHRT_MAX)
	{
		cout << "Too many points to interpolate";
	}

	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();

	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k); // 找多少个已经有起点和终点的点，默认是 100
	gd->setSigma(sigma); // 高斯模糊的sigma，默认是 1.0
	gd->setUsePostProcessing(use_post_proc); // 是否使用后处理，默认是 true
	gd->setFGSLambda(fgs_lambda); // 是否使用FGS，默认是 true
	gd->setFGSSigma(fgs_sigma); // FGS的sigma，默认是 1.0

	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);

	long double execTime2 = (getTickCount() * 1.0000 - timeStart2) / (getTickFrequency() * 1.0000);
	std::cout << "Time to interpolate = " << execTime2 << "\n";
}




// =====================================================
// Batch InterpoNet function (supports feature point export)
// =====================================================

std::vector<cv::Mat> FeatureMatcher::degraf_flow_InterpoNet(
    const std::vector<cv::Mat>& batch_i1,
    const std::vector<cv::Mat>& batch_i2,
    const std::vector<std::string>& batch_num_strs,
    std::vector<std::vector<cv::Point2f>>* out_points_filtered,   
    std::vector<std::vector<cv::Point2f>>* out_dst_points_filtered)
{
    const bool profile_stages = true;
    int saliency_kernel = 3;
    envInt("DEGRAF_SALIENCY_KERNEL", saliency_kernel);
    int degraf_window_w = 3;
    int degraf_window_h = 3;
    int degraf_step_x = 9;
    int degraf_step_y = 9;
    envInt("DEGRAF_WINDOW_W", degraf_window_w);
    envInt("DEGRAF_WINDOW_H", degraf_window_h);
    envInt("DEGRAF_STEP_X", degraf_step_x);
    envInt("DEGRAF_STEP_Y", degraf_step_y);
    saliency_kernel = std::max(1, saliency_kernel);
    degraf_window_w = std::max(1, degraf_window_w);
    degraf_window_h = std::max(1, degraf_window_h);
    degraf_step_x = std::max(1, degraf_step_x);
    degraf_step_y = std::max(1, degraf_step_y);
    const std::string points_input_dir = envOrDefault("DEGRAF_POINTS_INPUT_DIR", "");
    const std::string matches_input_dir = envOrDefault("DEGRAF_MATCHES_INPUT_DIR", "");
    const bool use_ba_matches = envEnabled("DEGRAF_INTERPONET_USE_BA_MATCHES", false);
#if USE_CUDA
    const char* degraf_backend = "cuda_degraf";
#else
    const char* degraf_backend = "cpu_degraf";
#endif
    std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] degraf_backend="
              << degraf_backend << std::endl;
    std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] saliency_kernel="
              << saliency_kernel
              << " window=" << degraf_window_w << "x" << degraf_window_h
              << " step=" << degraf_step_x << "x" << degraf_step_y
              << std::endl;
    if (!points_input_dir.empty())
        std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] points_input_dir=" << points_input_dir << std::endl;
    if (!matches_input_dir.empty())
        std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] matches_input_dir=" << matches_input_dir << std::endl;
    std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] use_ba_matches="
              << (use_ba_matches ? "1" : "0") << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> batch_flows;
    batch_flows.reserve(batch_i1.size());

    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_num_strs.size())
        return batch_flows;

    // =====================================================
    // Step 1: DeGraF feature detection in-process
    // =====================================================
    std::vector<std::vector<cv::Point2f>> batch_points;
    batch_points.reserve(batch_i1.size());

    for (size_t idx = 0; idx < batch_i1.size(); ++idx) {
        const cv::Mat& prev = batch_i1[idx];
        std::vector<cv::Point2f> points;
        if (!points_input_dir.empty())
        {
            const std::string p1 = buildIndexedPath(points_input_dir, batch_num_strs[idx], "_points.txt");
            const std::string p2 = buildIndexedPath(points_input_dir, batch_num_strs[idx], ".txt");
            if (loadPointsFile(p1, points) || loadPointsFile(p2, points))
            {
                batch_points.push_back(points);
                continue;
            }
            std::cerr << "[WARN][FeatureMatcher::degraf_flow_InterpoNet] points file missing/invalid for "
                      << batch_num_strs[idx] << ", fallback to online DeGraF detect." << std::endl;
        }
        detectDegrafPointsForInterpoNet(
            prev, saliency_kernel, degraf_window_w, degraf_window_h, degraf_step_x, degraf_step_y, points);
        batch_points.push_back(points);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<cv::Point2f>> batch_points_ba;
    if (use_ba_matches)
    {
        batch_points_ba.reserve(batch_i2.size());
        for (size_t idx = 0; idx < batch_i2.size(); ++idx)
        {
            std::vector<cv::Point2f> points_ba;
            detectDegrafPointsForInterpoNet(
                batch_i2[idx], saliency_kernel, degraf_window_w, degraf_window_h, degraf_step_x, degraf_step_y, points_ba);
            batch_points_ba.push_back(std::move(points_ba));
        }
    }

    // =====================================================
    // Step 2: RAFT in-process sparse matching
    // =====================================================
    std::vector<SparseFlowMatches> batch_matches; // 用于稀疏匹配的匹配结果
    std::vector<SparseFlowMatches> batch_matches_ba; // 用于BA匹配的匹配结果
    bool loaded_all_matches = false;
    if (!matches_input_dir.empty())
    {
        loaded_all_matches = true;
        batch_matches.reserve(batch_i1.size());
        for (size_t idx = 0; idx < batch_i1.size(); ++idx)
        {
            SparseFlowMatches m;
            const std::string p1 = buildIndexedPath(matches_input_dir, batch_num_strs[idx], "_matches.txt");
            const std::string p2 = buildIndexedPath(matches_input_dir, batch_num_strs[idx], ".txt");
            if (!(loadMatchesFile(p1, m) || loadMatchesFile(p2, m)))
            {
                std::cerr << "[WARN][FeatureMatcher::degraf_flow_InterpoNet] matches file missing/invalid for "
                          << batch_num_strs[idx] << ", fallback to RAFT TRT." << std::endl;
                loaded_all_matches = false;
                batch_matches.clear();
                break;
            }
            batch_matches.push_back(m);
        }
    }

    if (!loaded_all_matches)
    {
        RaftEngineTRT raft_engine;
        if (!raft_engine.estimateMatchesBatch(batch_i1, batch_i2, batch_points, batch_matches))
        {
            return batch_flows;
        }
        if (use_ba_matches)
        {
            if (!raft_engine.estimateMatchesBatch(batch_i2, batch_i1, batch_points_ba, batch_matches_ba))
            {
                return batch_flows;
            }
        }
    }
    else if (use_ba_matches)
    {
        RaftEngineTRT raft_engine;
        if (!raft_engine.estimateMatchesBatch(batch_i2, batch_i1, batch_points_ba, batch_matches_ba))
        {
            return batch_flows;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // Optional visualization outputs for evaluation UI.
    if (out_points_filtered && out_dst_points_filtered) {
        out_points_filtered->clear();
        out_dst_points_filtered->clear();
        out_points_filtered->reserve(batch_i1.size());
        out_dst_points_filtered->reserve(batch_i1.size());

        for (const auto &m : batch_matches)
        {
            out_points_filtered->push_back(m.src_points);
            out_dst_points_filtered->push_back(m.dst_points);
        }
    }

    // =====================================================
    // Step 3: InterpoNet in-process dense interpolation
    // =====================================================
    // IMPORTANT:
    // InterpoNetEngineTRT resolves edges files by:
    //   frame_id = DEGRAF_INTERPONET_EDGES_START_INDEX + local_batch_i
    // where local_batch_i is 0..(batch_size-1).
    // In batched evaluation, if START_INDEX is left at a constant value (e.g. 0),
    // later batches would repeatedly read 000000..00000N edges and silently hurt quality.
    // Here we align START_INDEX to the current batch's first global frame id.
    int batch_start_index = 0;
    if (!batch_num_strs.empty())
    {
        try
        {
            batch_start_index = std::stoi(batch_num_strs.front());
            setenv("DEGRAF_INTERPONET_EDGES_START_INDEX", std::to_string(batch_start_index).c_str(), 1);
            setenv("DEGRAF_INTERPONET_DEBUG_START_INDEX", std::to_string(batch_start_index).c_str(), 1);
        }
        catch (...)
        {
            // Keep existing env values if frame id parsing fails.
        }
    }

    InterpoNetEngineTRT interponet_engine(127, 0.05f, true, 500.0f, 1.5f);
    const std::vector<SparseFlowMatches> *batch_matches_ba_ptr =
        (use_ba_matches && batch_matches_ba.size() == batch_i1.size()) ? &batch_matches_ba : nullptr;
    if (!interponet_engine.densifyBatch(batch_i1, batch_i2, batch_matches, batch_matches_ba_ptr, batch_flows))
    {
        batch_flows.clear();
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    if (profile_stages)
    {
        const double degraf_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        const double sparse_match_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
        const double dense_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t3 - t2).count();
        std::cout << "[PROFILE][FeatureMatcher::degraf_flow_InterpoNet] frames=" << batch_i1.size()
                  << " degraf_ms=" << degraf_ms
                  << " sparse_match_ms=" << sparse_match_ms
                  << " dense_refine_ms=" << dense_ms
                  << " total_ms=" << (degraf_ms + sparse_match_ms + dense_ms)
                  << std::endl;
    }
    return batch_flows;
}

