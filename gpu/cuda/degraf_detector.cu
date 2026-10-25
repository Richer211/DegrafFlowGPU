#include "degraf_detector.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// Enhanced CUDA kernel with quality filtering and sub-pixel precision
__global__ void computeGradientsEnhancedKernel(
    const float *__restrict__ image_data,  // 输入图像数据 32位浮点数
    int image_width,                       // 图像宽度 图像的宽度
    int image_height,                      // 图像高度 图像的高度
    int window_width,                      // 窗口宽度 窗口的宽度
    int window_height,                     // 窗口高度 窗口的高度
    int step_x,                            // X方向步长
    int step_y,                            // Y方向步长
    int matrix_width,                      // 矩阵宽度 矩阵的意义：一共有多少个窗口位置要处理，每个线程处理一个网格点。
    int matrix_height,                     // 矩阵高度 矩阵的意义：一共有多少个窗口位置要处理，每个线程处理一个网格点。
    float magnitude_threshold,             // 梯度幅度阈值
    float contrast_threshold,              // 局部对比度阈值
    float edge_response_threshold,         // 边缘响应阈值
    float *__restrict__ keypoint_x,        // 关键点X坐标 GPU内存
    float *__restrict__ keypoint_y,        // 关键点Y坐标 GPU内存
    float *__restrict__ keypoint_response) // 关键点响应值 GPU内存
{
    __shared__ float window_data[49]; // 共享内存 49个浮点数 用于存储窗口内的像素值

    /**
     * 变量来源（都是 CUDA 内置变量，来自 CUDA 运行时）
     * blockDim.x 和 blockDim.y 线程块的尺寸（每个 block 在 x/y 方向有多少线程）
     * threadIdx.x 和 threadIdx.y 线程在当前 block 内的索引（该线程在 block 里的坐标）
     * blockIdx：当前线程块在网格中的索引（dim3 类型）
     * blockDim：线程块维度（每个 block 内线程数量，dim3）
     * threadIdx：线程在当前 block 内的索引（dim3）
     * 它们由 CUDA 在 kernel 启动时自动设置，不需要你手动定义。
     * 作用：
     * x、y：把二维网格坐标计算出来，用于定位当前线程负责的“矩阵位置”（即窗口网格中的一个点）
     * tid：把 block 内二维线程索引映射成一维索引，常用于共享内存访问或局部数组索引
     */

    /**
     * 这里blockIdx.x * blockDim.x 表示当前block在全局x 方向的起始线程索引， threadIdx.x 表示当前线程在本 block 内的 x 方向索引（从 0 开始）。
     * 简单来说blockIdx.x 表示当前的block在X方向的索引，而blockDim.x 表示每个block在X方向的线程数量。
     * 两者相乘得到当前block在全局x 方向的起始线程索引。
     * 再加上线程在本block方向的偏移，就得到了线程在全局x 方向的索引。
     */
    int x = blockIdx.x * blockDim.x + threadIdx.x;    // 线程在矩阵中的X坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // 线程在矩阵中的Y坐标
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 线程在矩阵中的线程ID

    if (x >= matrix_width || y >= matrix_height) // 如果线程在矩阵中的X坐标或Y坐标超出矩阵范围，则返回
        return;

    int idx = y * matrix_width + x;  // 把二维网格坐标转换成一维索引，方便访问 keypoint_x/y/response 这样的线性数组
    int window_start_x = x * step_x; // 当前网格点对应的窗口左上角在图像中的坐标，因为每个网格点间隔是 step_x / step_y，所以第 x 列窗口的起点是 x * step_x
    int window_start_y = y * step_y; // 当前网格点对应的窗口左上角在图像中的坐标，因为每个网格点间隔是 step_x / step_y，所以第 y 行窗口的起点是 y * step_y

    if (window_start_x + window_width + 1 >= image_width ||
        window_start_y + window_height + 1 >= image_height) // 这段是在做边界检查，避免窗口越界访问图像。
    {
        keypoint_x[idx] = -1.0f;       // 如果窗口越界，则将关键点坐标设置为无效值，-1.0f 表示无效
        keypoint_y[idx] = -1.0f;       // 如果窗口越界，则将关键点坐标设置为无效值，-1.0f 表示无效
        keypoint_response[idx] = 0.0f; // 如果窗口越界，则将关键点响应值设置为0
        return;                        // 如果窗口越界，则返回，不进行后续计算
    }

    /**
     * 这是为了让后续遍历时能正确更新 max/min：
     * max_value 从 0 开始：因为像素值通常是非负（0~255 或浮点非负），后面遇到任何更大的值都会更新最大值。
     * min_value 从一个很大的数开始（1e6）：确保第一次读取的像素值一定更小，从而更新最小值。
     */
    int window_size = window_width * window_height;
    float max_value = 0.0f;
    float min_value = 1e6f;

    // First pass: find max/min values and load data
    for (int i = 0; i <= window_height; i++)
    {
        for (int j = 0; j <= window_width; j++)
        {
            int img_y = window_start_y + i;                              // 当前像素在整张图中的Y坐标
            int img_x = window_start_x + j;                              // 当前像素在整张图中的X坐标
            float pixel_value = image_data[img_y * image_width + img_x]; // 这里的img_y * image_width + img_x 是全局内存的索引，因为全局内存是按一维数组存储的，所以需要转换成一维索引。

            if (window_size <= 49 && tid == 0) // 如果窗口大小小于等于49，并且当前线程是第一个线程，则将像素值存储到共享内存中
            {
                // window_data[i * window_width + j] 只是用二维索引映射成一维索引来访问这个共享数组 ， window_data是一个共享内存，在最开始就被定义了。
                window_data[i * window_width + j] = pixel_value; // 将像素值存储到共享内存中
            }
            // fmaxf 和 fminf 是 CUDA 内置函数，用于计算浮点数的最大值和最小值。
            max_value = fmaxf(max_value, pixel_value); // 更新最大值
            min_value = fminf(min_value, pixel_value); // 更新最小值
        }
    }

    // Calculate local contrast for quality assessment
    float local_contrast = max_value - min_value; // 计算局部对比度

    // Early exit for low contrast regions
    // if (local_contrast < contrast_threshold) {
    //     keypoint_x[idx] = -1.0f;
    //     keypoint_y[idx] = -1.0f;
    //     keypoint_response[idx] = 0.0f;
    //     return;
    // }

    // Initialize accumulators
    float divident_high_x = 0.0f; // 质心X方向的分子
    float divident_high_y = 0.0f; // 质心Y方向的分子
    float divisor_high = 0.0f;    // 质心方向的分母, 也就是亮度总和
    int counter = 0;              // 计数器 用于统计窗口内像素的总数

    // Gradient computation for edge response calculation
    float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f; // 梯度X方向的分子，梯度Y方向的分子，梯度XY方向的分子

    // Second pass: compute centroids and gradients
    if (window_size <= 49) // 如果窗口大小小于等于49，则使用共享内存
    {
        __syncthreads();                        // 同步线程，确保所有线程都执行到这个点
        for (int i = 0; i < window_height; i++) // 遍历窗口内所有像素
        {
            for (int j = 0; j < window_width; j++) // 遍历窗口内所有像素
            {
                float pixel_value = window_data[i * window_width + j]; // 这里的i * window_width + j 是共享内存的索引，因为共享内存是按一维数组存储的，所以需要转换成一维索引。

                /**
                 * 这里是用来计算x坐标的像素值加权，window_start_x + j 是当前像素在整张图中的X坐标，pixel_value 表示该像素的强度，相乘后累加到divident_high_x中，用于之后计算加权平均（质心）
                 * float是为了保证乘法用浮点运算，并避免整型乘法导致精度丢失
                 */
                divident_high_x += (float)(window_start_x + j) * pixel_value;
                divident_high_y += (float)(window_start_y + i) * pixel_value;
                divisor_high += pixel_value;
                counter++;

                // Compute gradients for edge response (simplified)
                if (i > 0 && i < window_height - 1 && j > 0 && j < window_width - 1)
                { // 只在窗口内部像素计算梯度，避免访问边界越界。

                    /**
                     * gx：当前像素在 x 方向的梯度，右边像素 I(x+1) 减左边像素 I(x-1)
                     * gy：当前像素在 y 方向的梯度，下边像素 I(y+1) 减上边像素 I(y-1)
                     * 也就是用局部邻域差分来估计边缘变化强度，用于后面计算角点响应。
                     */
                    float gx = window_data[i * window_width + (j + 1)] - window_data[i * window_width + (j - 1)];
                    float gy = window_data[(i + 1) * window_width + j] - window_data[(i - 1) * window_width + j];

                    // 这三行是在累加结构张量（second-moment matrix）的分量
                    Ixx += gx * gx; // x 方向梯度平方的累加
                    Iyy += gy * gy; // y 方向梯度平方的累加
                    Ixy += gx * gy; // x 方向和 y 方向梯度乘积的累加
                    // 这些量用于后面计算角点响应（如 Harris 角点），判断该窗口是角点、边缘还是平坦区域。
                }
            }
        }
    }
    else // 如果窗口大小大于49，则直接从全局内存读取像素值
    {
        /**
         *  全局内存 vs 共享内存的核心区别：
         *  全局内存（global memory）：GPU 上的“主存”，所有线程/所有 block 都能访问，但访问延迟高、带宽受限。
         *  在代码里就是 image_data[...] 这种指针访问（image_data 指向全局内存）。
         *  共享内存（shared memory）：每个 block 内共享、速度快、容量小，只能被同一个 block里的线程访问，生命周期随 block 结束而结束。
         *  在代码里就是 __shared__ float window_data[49];，用 window_data[...] 访问。
         *  不同之处：
         *  1，数据来源不同：image_data[...] → 直接读全局内存， window_data[...] → 先从全局读到共享，再从共享读
         *  2，目的：用共享内存缓存小窗口数据，减少重复的全局内存访问，让多个线程（同一 block 内）复用同一块数据
         *  全局内存 = 大而慢；共享内存 = 小而快（但只在 block 内共享）
         */
        for (int i = 0; i < window_height; i++)
        {
            for (int j = 0; j < window_width; j++)
            {
                int img_y = window_start_y + i;                              // 当前像素在整张图中的Y坐标, 也就是行坐标
                int img_x = window_start_x + j;                              // 当前像素在整张图中的X坐标, 也就是列坐标
                float pixel_value = image_data[img_y * image_width + img_x]; // 当前像素的值, 也就是亮度值

                divident_high_x += (float)img_x * pixel_value; // 计算x坐标的像素值加权, 也就是列坐标 * 亮度值
                divident_high_y += (float)img_y * pixel_value; // 计算y坐标的像素值加权, 也就是行坐标 * 亮度值
                divisor_high += pixel_value;                   // 计算像素值的总和，也就是亮度总和
                counter++;                                     // 计数器加1

                // Compute gradients for edge response
                if (i > 0 && i < window_height - 1 && j > 0 && j < window_width - 1)
                {
                    float gx = image_data[img_y * image_width + (img_x + 1)] -
                               image_data[img_y * image_width + (img_x - 1)];
                    float gy = image_data[(img_y + 1) * image_width + img_x] -
                               image_data[(img_y - 1) * image_width + img_x];

                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
        }
    }

    // Calculate edge response (Harris-like)
    // float det = Ixx * Iyy - Ixy * Ixy;
    // float trace = Ixx + Iyy;
    // float edge_response = (trace > 1e-6f) ? (trace * trace / det) : 1e6f;

    // Edge response filtering
    // if (edge_response > edge_response_threshold) {
    //     keypoint_x[idx] = -1.0f;
    //     keypoint_y[idx] = -1.0f;
    //     keypoint_response[idx] = 0.0f;
    //     return;
    // }

    // Calculate final results with sub-pixel precision
    if (divisor_high > 1e-6f) // 如果像素值的总和大于1e-6，则进行后续计算
    {
        float centre_x = (float)window_start_x + ((float)window_width / 2.0f);  // 窗口中心的X坐标
        float centre_y = (float)window_start_y + ((float)window_height / 2.0f); // 窗口中心的Y坐标

        float centroid_x = divident_high_x / divisor_high; // 计算质心X坐标
        float centroid_y = divident_high_y / divisor_high; // 计算质心Y坐标

        // 乘以 2.0f 是放大偏移，把质心偏离中心的幅度加大，用于后续阈值判断或响应计算时更敏感
        float dx = 2.0f * (centroid_x - centre_x); // 计算质心相对窗口中心偏移X坐标
        float dy = 2.0f * (centroid_y - centre_y); // 计算质心相对窗口中心偏移Y坐标

        float magnitude = sqrtf(dx * dx + dy * dy); // 表示质心偏移的大小（偏移幅度），是二维向量的欧式长度

        // Magnitude-based quality filtering
        // if (magnitude < magnitude_threshold) {
        //     keypoint_x[idx] = -1.0f;
        //     keypoint_y[idx] = -1.0f;
        //     keypoint_response[idx] = 0.0f;
        //     return;
        // }

        // Sub-pixel refinement using local intensity distribution
        float sub_pixel_offset_x = 0.0f; // 亚像素偏移X
        float sub_pixel_offset_y = 0.0f; // 亚像素偏移Y

        // 亚像素精化
        if (window_size <= 49) // 如果窗口大小小于等于49
        {
            // Simple sub-pixel refinement for small windows
            float weight_sum = 0.0f;
            for (int i = 0; i < window_height; i++)
            {
                for (int j = 0; j < window_width; j++)
                {
                    float pixel_value = window_data[i * window_width + j]; // 窗口像素
                    float weight = pixel_value / divisor_high;             // 归一化权重 divisor_high 是像素值的总和
                    // j - window_width / 2.0f 是当前像素相对于窗口中心的X坐标偏移，* 0.5f 是缩小偏移幅度（经验系数）
                    sub_pixel_offset_x += weight * (j - window_width / 2.0f) * 0.5f;  // 亚像素偏移X
                    sub_pixel_offset_y += weight * (i - window_height / 2.0f) * 0.5f; // 亚像素偏移Y
                    weight_sum += weight;                                             // 权重总和
                }
            }
            if (weight_sum > 1e-6f) // 如果权重总和大于1e-6 是为了避免除以 0 或极小数导致数值不稳定。1e-6f 是一个安全阈值，表示“权重几乎为零时就不做归一化”。
            {
                /**
                 * 这是在做归一化，把前面累加的“加权偏移和”变成“加权平均偏移”
                 * sub_pixel_offset_x/y 之前累加的是 weight * 偏移
                 * weight_sum 是所有权重之和
                 * 所以要除以 weight_sum 才得到真正的平均偏移量
                 */
                sub_pixel_offset_x /= weight_sum; // 亚像素偏移X
                sub_pixel_offset_y /= weight_sum; // 亚像素偏移Y
            }
        }

        // Store results with sub-pixel precision
        /**
         * keypoint_x/y[idx]：最终关键点坐标
         * centroid_x/y：窗口内加权质心
         * dx/dy：质心相对中心的放大偏移
         * sub_pixel_offset_*：亚像素精细修正
         */
        keypoint_x[idx] = centroid_x + dx + sub_pixel_offset_x; // 最终关键点X坐标
        keypoint_y[idx] = centroid_y + dy + sub_pixel_offset_y; // 最终关键点Y坐标
        keypoint_response[idx] = magnitude * local_contrast;    // 最终质量分数：响应值，用 偏移幅度(magnitude) × 局部对比度(local_contrast) 作为质量分数
    }
    else
    {
        keypoint_x[idx] = -1.0f;       // 无效关键点坐标
        keypoint_y[idx] = -1.0f;       // 无效关键点坐标
        keypoint_response[idx] = 0.0f; // 无效质量分数
    }
}

// Optimized image conversion kernel
__global__ void convertToFloat32OptimizedKernel(
    const unsigned char *__restrict__ src,
    float *__restrict__ dst,
    int width,
    int height,
    int channels)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程处理的像素索引（一维展开）
    int total_pixels = width * height;               // 总像素数

    if (idx >= total_pixels) // 越界保护
        return;              // 如果当前线程处理的像素索引越界，则直接返回，不进行后续计算

    if (channels == 1) // 如果图像是单通道灰度图
    {
        dst[idx] = (float)src[idx] + 1.0f; // 将像素值转换为浮点数并加上1.0f
    }
    else if (channels == 3) // 如果图像是三通道彩色图
    {
        int src_idx = idx * 3; // 当前像素的索引 * 3 得到彩色图的索引
        /**
         * 这是标准的 RGB → 灰度亮度公式（NTSC/BT.601），用人眼对不同颜色敏感度的加权：
         * 绿色最敏感 → 权重最大 0.587
         * 红色次之 → 0.299
         * 蓝色最不敏感 → 0.114
         * 所以用这三个系数加权，可以得到更符合人眼感知的亮度灰度值。
         */
        float gray = 0.299f * src[src_idx] + 0.587f * src[src_idx + 1] + 0.114f * src[src_idx + 2]; // 将彩色图转换为灰度图
        dst[idx] = gray + 1.0f;                                                                     // 将灰度图转换为浮点数并加上1.0f 让像素值整体偏移，避免后续计算出现 0（可能用于除法或权重）
    }
}

// Constructor with pre-allocation and warmup
// 构造函数：预分配内存并预热GPU
CudaGradientDetector::CudaGradientDetector()
    : d_image_data(nullptr), d_keypoint_x(nullptr), d_keypoint_y(nullptr),
      d_keypoint_response(nullptr), init_flag(false), stream(nullptr)
{

    CUDA_CHECK(cudaStreamCreate(&stream)); // 创建CUDA流

    max_image_size = 2048 * 2048;              // 最大图像大小
    max_matrix_size = (2048 / 3) * (2048 / 3); // 最大矩阵大小

    size_t image_bytes = max_image_size * sizeof(float);     // 图像字节数
    size_t keypoint_bytes = max_matrix_size * sizeof(float); // 关键点字节数

    CUDA_CHECK(cudaMalloc(&d_image_data, image_bytes));           // 分配图像内存
    CUDA_CHECK(cudaMalloc(&d_keypoint_x, keypoint_bytes));        // 分配关键点X内存
    CUDA_CHECK(cudaMalloc(&d_keypoint_y, keypoint_bytes));        // 分配关键点Y内存
    CUDA_CHECK(cudaMalloc(&d_keypoint_response, keypoint_bytes)); // 分配关键点响应值内存

    warmupGPU();      // 预热GPU
    init_flag = true; // 初始化标志
}

// 预热GPU
void CudaGradientDetector::warmupGPU()
{
    try
    {
        cv::Mat test_img = cv::Mat::ones(64, 64, CV_8UC1);
        cv::Size test_image_size = test_img.size();
        cv::Size test_window_size(3, 3);
        int test_step_x = 9, test_step_y = 9;

        cv::Size test_matrix_size;
        test_matrix_size.width = (test_image_size.width - test_window_size.width) / test_step_x;
        test_matrix_size.height = (test_image_size.height - test_window_size.height) / test_step_y;

        int test_current_image_size = test_image_size.width * test_image_size.height;

        unsigned char *d_test_input;
        size_t test_input_bytes = test_img.rows * test_img.cols;
        CUDA_CHECK(cudaMalloc(&d_test_input, test_input_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_test_input, test_img.data, test_input_bytes,
                                   cudaMemcpyHostToDevice, stream));

        dim3 conv_block(256);
        dim3 conv_grid((test_current_image_size + conv_block.x - 1) / conv_block.x);

        convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
            d_test_input, d_image_data,
            test_img.cols, test_img.rows, test_img.channels());

        dim3 grad_block(16, 16);
        dim3 grad_grid((test_matrix_size.width + grad_block.x - 1) / grad_block.x,
                       (test_matrix_size.height + grad_block.y - 1) / grad_block.y);

        computeGradientsEnhancedKernel<<<grad_grid, grad_block, 0, stream>>>(
            d_image_data,
            test_image_size.width, test_image_size.height,
            test_window_size.width, test_window_size.height,
            test_step_x, test_step_y,
            test_matrix_size.width, test_matrix_size.height,
            0.1f, 5.0f, 10.0f, // Default quality thresholds
            d_keypoint_x, d_keypoint_y, d_keypoint_response);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_test_input));
    }
    catch (const std::exception &e)
    {
        std::cerr << "GPU warmup failed: " << e.what() << std::endl;
    }
}

// 析构函数：释放资源
CudaGradientDetector::~CudaGradientDetector()
{
    Release(); // 释放资源
    if (stream)
    {
        cudaStreamDestroy(stream); // 销毁CUDA流
    }
}

// 检测关键点：输入图像、窗口大小、步长
// 返回值：1成功，0失败
int CudaGradientDetector::CudaDetectGradients(const cv::Mat &src_image, int p_window_width,
                                              int p_window_height, int p_step_x, int p_step_y)
{

    if (src_image.empty())
    { // 如果输入图像为空
        std::cerr << "Error: Input image is empty!" << std::endl;
        return 0;
    }

    image_size = src_image.size(); // 图像大小 这是 二维尺寸（宽、高），用在矩阵/窗口/网格计算里。
    window_size = cv::Size(p_window_width, p_window_height);
    step_x = p_step_x; // X方向步长
    step_y = p_step_y; // Y方向步长

    matrix_size.width = (image_size.width - window_size.width) / p_step_x;    // 矩阵宽度
    matrix_size.height = (image_size.height - window_size.height) / p_step_y; // 矩阵高度

    std::cout << "CUDA matrix_size: " << matrix_size.width << "x" << matrix_size.height
              << " = " << matrix_size.width * matrix_size.height << std::endl;
    std::cout << "CUDA image_size: " << image_size.width << "x" << image_size.height << std::endl;
    std::cout << "CUDA window_size: " << window_size.width << "x" << window_size.height << std::endl;
    std::cout << "CUDA step: " << step_x << "x" << step_y << std::endl;

    int current_image_size = image_size.width * image_size.height;    // 当前图像大小 这是 像素总数（一维数量），用在内存分配、kernel grid 计算里。
    int current_matrix_size = matrix_size.width * matrix_size.height; // 当前矩阵大小 矩阵网格的总数量（一维总数），也就是窗口总数/候选关键点总数。

    if (current_image_size > max_image_size || current_matrix_size > max_matrix_size)
    { // 如果当前图像大小或矩阵大小大于最大图像大小或矩阵大小
        std::cerr << "Error: Image too large for pre-allocated GPU memory!" << std::endl;
        return 0;
    }

    cv::Mat image_8u; // 8位无符号字符图像
    if (src_image.channels() == 3)
    {
        cv::cvtColor(src_image, image_8u, cv::COLOR_BGR2GRAY); // 将彩色图像转换为灰度图像
    }
    else
    {
        image_8u = src_image.clone(); // 复制图像
    }

    if (image_8u.depth() != CV_8U)
    {                                        // 如果图像深度不为8位无符号字符
        image_8u.convertTo(image_8u, CV_8U); // 将图像转换为8位无符号字符
    }

    unsigned char *d_input_image;                        // 输入图像
    size_t input_bytes = image_8u.rows * image_8u.cols;  // 输入图像字节数
    CUDA_CHECK(cudaMalloc(&d_input_image, input_bytes)); // 分配输入图像内存 这里的 & 是取地址符 cudaError_t cudaMalloc(void** devPtr, size_t size);
    CUDA_CHECK(cudaMemcpyAsync(d_input_image, image_8u.data, input_bytes,
                               cudaMemcpyHostToDevice, stream)); // 将8位无符号字符图像复制到GPU内存

    dim3 conv_block(256);                                                   // 卷积块大小 dim3表示三维线程块
    dim3 conv_grid((current_image_size + conv_block.x - 1) / conv_block.x); // 卷积网格大小 dim3表示三维线程网格

    convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
        d_input_image, d_image_data,
        image_8u.cols, image_8u.rows, image_8u.channels()); // 将8位无符号字符图像转换为32位浮点图像

    dim3 grad_block(16, 16); // 梯度块大小 dim3表示三维线程块
    dim3 grad_grid((matrix_size.width + grad_block.x - 1) / grad_block.x,
                   (matrix_size.height + grad_block.y - 1) / grad_block.y); // 梯度网格大小 dim3表示三维线程网格

    // Enhanced kernel with quality filtering
    // 增强内核：质量过滤
    float magnitude_threshold = 0.3f;      // Minimum gradient magnitude 梯度幅度阈值
    float contrast_threshold = 8.0f;       // Minimum local contrast 局部对比度阈值
    float edge_response_threshold = 15.0f; // Maximum edge response (lower = more corner-like) 边缘响应阈值 越小越像角点 越大越像边缘

    computeGradientsEnhancedKernel<<<grad_grid, grad_block, 0, stream>>>(
        d_image_data,                                                     // 输入图像数据
        image_size.width, image_size.height,                              // 图像宽度 图像高度
        window_size.width, window_size.height,                            // 窗口宽度 窗口高度
        step_x, step_y,                                                   // X方向步长 Y方向步长
        matrix_size.width, matrix_size.height,                            // 矩阵宽度 矩阵高度 矩阵的意义：一共有多少个窗口位置要处理，每个线程处理一个网格点。
        magnitude_threshold, contrast_threshold, edge_response_threshold, // 梯度幅度阈值 局部对比度阈值 边缘响应阈值
        d_keypoint_x, d_keypoint_y, d_keypoint_response);                 // 关键点X坐标 GPU内存 关键点Y坐标 GPU内存 关键点响应值 GPU内存

    // cudaGetLastError，cudaStreamSynchronize来自于cuda_runtime.h
    CUDA_CHECK(cudaGetLastError());            // 获取CUDA错误
    CUDA_CHECK(cudaStreamSynchronize(stream)); // 同步CUDA流，等待所有CUDA操作完成

    // 扩容或者缩容来适配当前矩阵大小
    h_keypoint_x.resize(current_matrix_size);        // 关键点X坐标 主机内存
    h_keypoint_y.resize(current_matrix_size);        // 关键点Y坐标 主机内存
    h_keypoint_response.resize(current_matrix_size); // 关键点响应值 主机内存

    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_x.data(), d_keypoint_x,
                               current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream)); // 将关键点X坐标从GPU内存复制到主机内存
    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_y.data(), d_keypoint_y,
                               current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream)); // 将关键点Y坐标从GPU内存复制到主机内存
    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_response.data(), d_keypoint_response,
                               current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream)); // 将关键点响应值从GPU内存复制到主机内存

    CUDA_CHECK(cudaStreamSynchronize(stream)); // 同步CUDA流，等待所有CUDA操作完成
    CUDA_CHECK(cudaFree(d_input_image));       // 释放输入图像内存

    keypoints.clear();                          // 清空关键点
    keypoints.reserve(current_matrix_size / 4); // 预留关键点空间 这里是给 keypoints 预留容量，减少后续 emplace_back 时的动态扩容次数。

    for (int i = 0; i < current_matrix_size; i++)
    {
        float x = h_keypoint_x[i]; // 关键点X坐标
        float y = h_keypoint_y[i]; // 关键点Y坐标

        // Use the correct variable name in the kernel parameter
        // 使用正确的变量名在内核参数中
        // isnan 和 isinf 是 math.h 中的函数，用于检查浮点数是否为 NaN 或无穷大。
        // 如果坐标是有效数值且在图像范围内，则创建关键点
        if (!isnan(x) && !isnan(y) && !isinf(x) && !isinf(y) &&
            x >= 0 && x < image_size.width && y >= 0 && y < image_size.height)
        {
            // 创建关键点 这里是给 keypoints 添加关键点，每个关键点是一个 cv::KeyPoint 对象。
            // emplace_back 是 std::vector 的接口，用来直接在容器尾部构造一个元素，避免先构造临时对象再拷贝/移动，效率更高。
            keypoints.emplace_back(cv::Point2f(x, y),                                      // 关键点坐标
                                   (float)std::min(window_size.width, window_size.height), // 关键点大小
                                   -1,                                                     // 关键点角度 ，-1表示未知
                                   h_keypoint_response[i]);                                // 关键点响应值 这里使用关键点响应值作为关键点质量分数。
            // 整体意思是把计算出的关键点坐标和响应值封装成一个 cv::KeyPoint，添加到结果列表中。
        }
        else
        {
            // Replace invalid point with window center
            // 将无效点替换为窗口中心
            int matrix_x = i % matrix_size.width;                                          // 把一维索引 i 转成二维网格的列索引
            int matrix_y = i / matrix_size.width;                                          // 把一维索引 i 转成二维网格的行索引
            float center_x = matrix_x * step_x + window_size.width / 2.0f;                 // 计算窗口中心X坐标
            float center_y = matrix_y * step_y + window_size.height / 2.0f;                // 计算窗口中心Y坐标
            keypoints.emplace_back(cv::Point2f(center_x, center_y),                        // 窗口中心坐标
                                   (float)std::min(window_size.width, window_size.height), // 窗口大小
                                   -1,                                                     // 关键点角度 ，-1表示未知
                                   h_keypoint_response[i]);                                // 关键点响应值 这里使用关键点响应值作为关键点质量分数。
            // 当 GPU 输出的坐标无效时，退回到“窗口中心点”作为替代关键点，避免后续流程出现空值或越界。
        }
    }

    return 1; // 返回1表示成功
}
// 释放资源
void CudaGradientDetector::Release()
{
    if (d_image_data)
    {
        CUDA_CHECK(cudaFree(d_image_data)); // 释放图像内存
        d_image_data = nullptr;
    }
    if (d_keypoint_x)
    {
        CUDA_CHECK(cudaFree(d_keypoint_x)); // 释放关键点X内存
        d_keypoint_x = nullptr;
    }
    if (d_keypoint_y)
    {
        CUDA_CHECK(cudaFree(d_keypoint_y)); // 释放关键点Y内存
        d_keypoint_y = nullptr;
    }
    if (d_keypoint_response)
    {
        CUDA_CHECK(cudaFree(d_keypoint_response)); // 释放关键点响应值内存
        d_keypoint_response = nullptr;
    }

    h_keypoint_x.clear();        // 清空关键点X坐标 主机内存 h_keypoint_x 是一个 std::vector<float>
    h_keypoint_y.clear();        // 清空关键点Y坐标 主机内存 h_keypoint_y 是一个 std::vector<float>
    h_keypoint_response.clear(); // 清空关键点响应值 主机内存 h_keypoint_response 是一个 std::vector<float>
    keypoints.clear();           // 清空关键点 主机内存 keypoints 是一个 std::vector<cv::KeyPoint>
    // 初始化标志
    init_flag = false; // 初始化标志为 false
}

// 获取关键点
// 这是一个getter，用来把类里保存的关键点结果返回给外部调用者。
const std::vector<cv::KeyPoint> &CudaGradientDetector::GetKeypoints() const
{
    return keypoints;
}

// 获取内存使用情况
// 这是一个查询显存占用的辅助函数，把当前类预分配的 GPU 内存规模估算出来并返回给调用者。
// & 表示引用，调用者传进来的变量会被直接修改，所以调用完后这些变量就有值了。
void CudaGradientDetector::GetMemoryUsage(size_t &total_bytes, size_t &gradient_bytes, size_t &keypoint_bytes) const
{
    /**
     * image_bytes：预分配的输入图像缓冲区大小
     * keypoint_bytes：预分配的关键点输出缓冲区大小（x/y/response 三个 float 数组）
     * gradient_bytes：当前实现没有单独的梯度缓冲区，所以填 0
     * total_bytes：总的显存占用估计值
     */
    size_t image_bytes = max_image_size * sizeof(float);  // 图像字节数
    keypoint_bytes = max_matrix_size * sizeof(float) * 3; // 关键点字节数
    gradient_bytes = 0;                                   // 梯度字节数
    total_bytes = image_bytes + keypoint_bytes;           // 总字节数
}