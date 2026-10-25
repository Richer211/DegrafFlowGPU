/*!
 * Enhanced CUDA Implementation of Dense Gradient-based Features (DeGraF) Detector
 * Header file for high-performance GPU-accelerated gradient detection with quality filtering
 * By Gang Wang, Durham University 2025
 */

 #pragma once

 #include <opencv2/opencv.hpp>
 #include <opencv2/features2d.hpp>
 #include <vector>
 #include <cuda_runtime.h>
 
 //! High-performance CUDA-accelerated Dense Gradient-based Features (DeGraF) detector
 class CudaGradientDetector {
 private:
     // Optimized GPU memory layout
     float* d_image_data;                    // Input image data on GPU
     float* d_keypoint_x;                    // Keypoint X coordinates on GPU
     float* d_keypoint_y;                    // Keypoint Y coordinates on GPU
     float* d_keypoint_response;             // Keypoint response values on GPU
     
     // Host memory for results (optimized)
     std::vector<float> h_keypoint_x;
     std::vector<float> h_keypoint_y;
     std::vector<float> h_keypoint_response;
     
     // CUDA stream for async operations
     cudaStream_t stream;
     
     // Pre-allocated memory pools
     int max_image_size;
     int max_matrix_size;
     
     // Configuration parameters
     bool init_flag;
     cv::Size image_size, window_size, matrix_size;
     int step_x, step_y;
     
     void warmupGPU();
 
 public:
     // Public variables (compatible with original interface)
     std::vector<cv::KeyPoint> keypoints;
 
     // Constructor and destructor
     CudaGradientDetector();
     ~CudaGradientDetector();
     
     /*!
      * High-performance gradient detection function using enhanced CUDA with quality filtering
      * \param src_image Input image (grayscale or color)
      * \param p_window_width Width of gradient window (default: 3)
      * \param p_window_height Height of gradient window (default: 3)
      * \param p_step_x Step in X direction (default: 9)
      * \param p_step_y Step in Y direction (default: 9)
      * \return 1 on success, 0 on failure
      */
     int CudaDetectGradients(const cv::Mat& src_image, 
                        int p_window_width, int p_window_height, 
                        int p_step_x, int p_step_y);
     
     /*!
      * Release all GPU and host memory
      */
     void Release();
     
     /*!
      * Get detected keypoints
      * \return Reference to keypoints vector
      */
     const std::vector<cv::KeyPoint>& GetKeypoints() const;
     
     /*!
      * Get memory usage information
      * \param total_bytes Total GPU memory used
      * \param gradient_bytes Memory used for gradient storage (deprecated)
      * \param keypoint_bytes Memory used for keypoint storage
      */
     void GetMemoryUsage(size_t& total_bytes, size_t& gradient_bytes, size_t& keypoint_bytes) const;
     
     /*!
      * Check if detector is initialized
      * \return true if initialized, false otherwise
      */
     bool IsInitialized() const { return init_flag; }
 };