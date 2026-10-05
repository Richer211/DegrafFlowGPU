#include "inference/RaftEngineTRT.h"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#if DEGRAF_HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#endif

namespace
{
std::string envOrDefault(const char *name, const std::string &fallback)
{
    const char *value = std::getenv(name);
    if (value && std::string(value).size() > 0)
        return std::string(value);
    return fallback;
}

bool envEnabled(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    std::string v(value);
    for (char &c : v)
        c = static_cast<char>(std::tolower(c));
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

bool envInt(const char *name, int &out_value)
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

cv::Mat adaptImageToStaticDims(const cv::Mat &src, int dst_h, int dst_w)
{
    if (src.empty() || dst_h <= 0 || dst_w <= 0)
        return cv::Mat();
    if (src.rows == dst_h && src.cols == dst_w)
        return src;

    const int copy_h = std::min(src.rows, dst_h);
    const int copy_w = std::min(src.cols, dst_w);
    cv::Mat cropped = src(cv::Rect(0, 0, copy_w, copy_h));
    if (copy_h == dst_h && copy_w == dst_w)
        return cropped.clone();

    cv::Mat out;
    cv::copyMakeBorder(cropped, out,
                       0, dst_h - copy_h,
                       0, dst_w - copy_w,
                       cv::BORDER_REPLICATE);
    return out;
}

std::string toLower(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(c)); // 将字符串转换为小写
    return s; // 返回转换后的字符串
}

bool dumpMatchesToFile(const std::string &path, const SparseFlowMatches &matches)
{
    std::ofstream ofs(path);
    if (!ofs.is_open())
        return false;
    ofs << "# src_x src_y dst_x dst_y\n";
    const size_t n = std::min(matches.src_points.size(), matches.dst_points.size());
    for (size_t i = 0; i < n; ++i)
    {
        const cv::Point2f &s = matches.src_points[i];
        const cv::Point2f &d = matches.dst_points[i];
        ofs << s.x << " " << s.y << " " << d.x << " " << d.y << "\n";
    }
    return true;
}

#if DEGRAF_HAVE_TENSORRT
size_t tensorElementSize(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kINT32:
        return 4;
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
    case nvinfer1::DataType::kINT64:
        return 8;
    case nvinfer1::DataType::kBOOL:
        return 1;
#endif
    default:
        return 0;
    }
}
#endif

float halfToFloat(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = h & 0x03FFu;

    uint32_t fbits = 0;
    if (exp == 0)
    {
        if (mant == 0)
        {
            fbits = sign;
        }
        else
        {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0)
            {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            fbits = sign | (exp << 23) | (mant << 13);
        }
    }
    else if (exp == 0x1Fu)
    {
        fbits = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        exp = exp + (127 - 15);
        fbits = sign | (exp << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}
/**
 * chw 整张光流表，channel, height, width
 * channel 通道数，height 高度，width 宽度
 * h 高度，w 宽度
 * y 高度，x 宽度
 * zero_padding 是否填充0, 点落到图外时，这一格按 0 算
 */
float bilinearSample(const std::vector<float> &chw,int channel,int h, int w,float y, float x,bool zero_padding) 
{ 
    const int x0 = static_cast<int>(std::floor(x));// 计算x的整数部分
    const int x1 = x0 + 1; // 计算x的下一个整数部分，1表示x的下一个整数部分
    const int y0 = static_cast<int>(std::floor(y)); // 计算y的整数部分
    const int y1 = y0 + 1; // 计算y的下一个整数部分，1表示y的下一个整数部分

    const float wx = x - static_cast<float>(x0); // 计算x的权重
    const float wy = y - static_cast<float>(y0); // 计算y的权重
    // channel = 0 时，idx 落在前半张表，读到 du。channel = 1 时，跳过前半张，读到同一个 (x, y) 上的 dv。
    // 特征点还是那一个，只是这个位置上的两个数要各读一次。
    const auto at = [&](int yy, int xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w)
            return zero_padding ? 0.0f : chw[static_cast<size_t>(channel) * h * w +
                                               static_cast<size_t>(std::max(0, std::min(h - 1, yy))) * w +
                                               std::max(0, std::min(w - 1, xx))];
        const size_t idx = static_cast<size_t>(channel) * h * w + static_cast<size_t>(yy) * w + xx;
        return chw[idx]; // 返回通道为channel，坐标为(xx, yy)的像素值
    };
    // 四个角的名字只有 v00、v01、v10、v11，没有 v02、v03。下标第一位是行，第二位是列：0 是左或上，1 是右或下。v00表示左上角，v01表示右上角，v10表示左下角，v11表示右下角。
    const float v00 = at(y0, x0); // 左上，坐标为(x0, y0)的像素值
    const float v01 = at(y0, x1); // 右上，坐标为(x1, y0)的像素值
    const float v10 = at(y1, x0); // 左下，坐标为(x0, y1)的像素值
    const float v11 = at(y1, x1); // 右下，坐标为(x1, y1)的像素值
    const float v0 = v00 * (1.0f - wx) + v01 * wx; // 上面这一横
    const float v1 = v10 * (1.0f - wx) + v11 * wx; // 下面这一横
    return v0 * (1.0f - wy) + v1 * wy; // 上下再掺成一个数  v0 是上面左右两格掺完的一个数。v1 是下面左右两格掺完的一个数
}

bool runLkFallback(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<std::vector<cv::Point2f>> &batch_points,
    std::vector<SparseFlowMatches> &batch_matches)
{
    batch_matches.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_points.size())
        return false;

    std::cout << "[PROFILE][RaftEngineTRT] backend=lk_fallback"
              << " frames=" << batch_i1.size() << std::endl;

    batch_matches.reserve(batch_i1.size());
    double total_lk_ms = 0.0;
    int max_flow_length = 100;
    envInt("DEGRAF_LK_MAX_FLOW_LENGTH", max_flow_length);

    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        cv::Mat gray1, gray2;
        if (batch_i1[i].channels() == 3)
            cv::cvtColor(batch_i1[i], gray1, cv::COLOR_BGR2GRAY);
        else
            gray1 = batch_i1[i];

        if (batch_i2[i].channels() == 3)
            cv::cvtColor(batch_i2[i], gray2, cv::COLOR_BGR2GRAY);
        else
            gray2 = batch_i2[i];

        const auto &src = batch_points[i];
        std::vector<cv::Point2f> dst;
        std::vector<unsigned char> status;
        std::vector<float> err;

        auto lk_start = std::chrono::high_resolution_clock::now();
        if (!src.empty())
        {
            cv::calcOpticalFlowPyrLK(
                gray1,
                gray2,
                src,
                dst,
                status,
                err,
                cv::Size(11, 11),
                4);
        }
        auto lk_end = std::chrono::high_resolution_clock::now();
        const double lk_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(lk_end - lk_start).count();
        total_lk_ms += lk_ms;

        SparseFlowMatches matches;
        matches.src_points.reserve(src.size());
        matches.dst_points.reserve(src.size());

        for (size_t p = 0; p < src.size(); ++p)
        {
            if (p >= dst.size() || p >= status.size() || !status[p])
                continue;

            const cv::Point2f &s = src[p];
            const cv::Point2f &d = dst[p];
            const float dx = s.x - d.x;
            const float dy = s.y - d.y;
            if (max_flow_length > 0 &&
                std::sqrt(dx * dx + dy * dy) >= static_cast<float>(max_flow_length))
                continue;

            if (d.x < 0 || d.y < 0 || d.x >= gray2.cols || d.y >= gray2.rows)
                continue;
            if (s.x < 0 || s.y < 0 || s.x >= gray1.cols || s.y >= gray1.rows)
                continue;

            matches.src_points.push_back(s);
            matches.dst_points.push_back(d);
        }

        std::cout << "[PROFILE][RaftEngineTRT][Frame " << i << "] src_points=" << src.size()
                  << " matched_points=" << matches.dst_points.size()
                  << " lk_ms=" << lk_ms
                  << std::endl;

        batch_matches.push_back(std::move(matches));
    }

    std::cout << "[PROFILE][RaftEngineTRT] backend=lk_fallback"
              << " total_lk_ms=" << total_lk_ms << std::endl;
    return true;
}

#if DEGRAF_HAVE_TENSORRT
class TrtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > Severity::kWARNING)
            return;
        std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

class RaftTrtRunner
{
public:
    bool init(const std::string &engine_path)
    {
        if (initialized_ && engine_path_ == engine_path)
            return true;

        reset();
        engine_path_ = engine_path;

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open())
            return false;
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size <= 0)
            return false;

        std::vector<char> engine_data(static_cast<size_t>(size));
        file.read(engine_data.data(), size); // 读取引擎数据engine
        if (!file.good())
            return false;

        runtime_.reset(nvinfer1::createInferRuntime(logger_)); // 创建推理运行时
        if (!runtime_)
            return false;
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size())); // 反序列化CUDA引擎
        if (!engine_)
            return false;
        context_.reset(engine_->createExecutionContext()); // 创建执行上下文
        if (!context_)
            return false;

        if (!resolveBindings()) // 解析绑定
            return false;

        initialized_ = true; // 初始化成功
        return true;
    }

    bool inferDenseFlow(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &flow_hw2)
    {
        if (!initialized_)
            return false;
        if (img1.empty() || img2.empty() || img1.size() != img2.size())
            return false;

        const int src_h = img1.rows;
        const int src_w = img1.cols;
        const int pad_h = (8 - (src_h % 8)) % 8;
        const int pad_w = (8 - (src_w % 8)) % 8;
        const int pad_left = pad_w / 2;
        const int pad_right = pad_w - pad_left;
        const int pad_top = 0;
        const int pad_bottom = pad_h;

        cv::Mat img1_pad = img1;
        cv::Mat img2_pad = img2;
        if (pad_h > 0 || pad_w > 0)
        {
            // Match RAFT Python InputPadder(mode='kitti'): symmetric left/right, pad only bottom on height.
            cv::copyMakeBorder(img1, img1_pad, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_REPLICATE);
            cv::copyMakeBorder(img2, img2_pad, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_REPLICATE);
        }

        const int h = img1_pad.rows;
        const int w = img1_pad.cols;
        int runtime_h = h;
        int runtime_w = w;

        // If the engine has static input dims (e.g. 1x3x376x1248), adapt padded input
        // by right/bottom pad or crop to avoid TRT setInputShape mismatch on variable-size frames.
        const nvinfer1::Dims model_in_dims = engine_->getTensorShape(input0_name_.c_str());
        if (model_in_dims.nbDims == 4 && model_in_dims.d[2] > 0 && model_in_dims.d[3] > 0)
        {
            const int model_h = model_in_dims.d[2];
            const int model_w = model_in_dims.d[3];
            if (model_h != runtime_h || model_w != runtime_w)
            {
                cv::Mat img1_adapt = adaptImageToStaticDims(img1_pad, model_h, model_w);
                cv::Mat img2_adapt = adaptImageToStaticDims(img2_pad, model_h, model_w);
                if (img1_adapt.empty() || img2_adapt.empty())
                    return false;
                img1_pad = img1_adapt;
                img2_pad = img2_adapt;
                runtime_h = model_h;
                runtime_w = model_w;
            }
        }

        nvinfer1::Dims4 input_dims(1, 3, runtime_h, runtime_w);
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
        if (!context_->setInputShape(input0_name_.c_str(), input_dims) ||
            !context_->setInputShape(input1_name_.c_str(), input_dims))
            return false;

        // Prefer resolved primary output (flow_up). Only fallback to max-area output
        // when the preferred tensor is unavailable/invalid at runtime.
        std::string selected_output_name = primary_output_name_;
        nvinfer1::Dims selected_output_dims = context_->getTensorShape(selected_output_name.c_str());
        if (selected_output_dims.nbDims != 4 || selected_output_dims.d[1] < 2 ||
            selected_output_dims.d[2] <= 0 || selected_output_dims.d[3] <= 0)
        {
            int best_area = -1;
            selected_output_name.clear();
            for (const auto &name : output_names_)
            {
                const nvinfer1::Dims od = context_->getTensorShape(name.c_str());
                if (od.nbDims != 4 || od.d[1] < 2 || od.d[2] <= 0 || od.d[3] <= 0)
                    continue;
                const int area = od.d[2] * od.d[3];
                if (area > best_area)
                {
                    best_area = area;
                    selected_output_name = name;
                    selected_output_dims = od;
                }
            }
            if (selected_output_name.empty())
                return false;
        }
        const nvinfer1::Dims out_dims = selected_output_dims;
#else
        if (!context_->setBindingDimensions(input0_idx_, input_dims) ||
            !context_->setBindingDimensions(input1_idx_, input_dims))
            return false;
        if (!context_->allInputDimensionsSpecified())
            return false;
        const nvinfer1::Dims out_dims = context_->getBindingDimensions(output_idx_);
#endif
        if (out_dims.nbDims != 4 || out_dims.d[1] < 2)
            return false;
        const int out_h = out_dims.d[2];
        const int out_w = out_dims.d[3];
        if (out_h <= 0 || out_w <= 0)
            return false;

        const size_t in_elems = static_cast<size_t>(3) * runtime_h * runtime_w;
        const size_t out_channels = static_cast<size_t>(out_dims.d[1]);
        const size_t out_elems = out_channels * out_h * out_w;
        const nvinfer1::DataType primary_dtype = engine_->getTensorDataType(selected_output_name.c_str());
        const size_t primary_elem_bytes = tensorElementSize(primary_dtype);
        if (primary_elem_bytes == 0)
            return false;

        std::vector<float> in0(in_elems);
        std::vector<float> in1(in_elems);
        std::vector<float> out(out_elems, 0.0f);
        toCHWFloat(img1_pad, in0);
        toCHWFloat(img2_pad, in1);

#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
        void *in0_dev = nullptr; 
        void *in1_dev = nullptr; 
        void *out_dev = nullptr;
        if (!allocDeviceByName(input0_name_, in_elems * sizeof(float), in0_dev) ||
            !allocDeviceByName(input1_name_, in_elems * sizeof(float), in1_dev))
            return false;
        if (!allocDeviceByName(selected_output_name, out_elems * primary_elem_bytes, out_dev))
            return false;

        // TRT10 enqueueV3 requires all outputs to have an address (or output allocator).
        for (const auto &name : output_names_)
        {
            const nvinfer1::Dims od = context_->getTensorShape(name.c_str());
            if (od.nbDims != 4 || od.d[1] <= 0 || od.d[2] <= 0 || od.d[3] <= 0)
                return false;
            const nvinfer1::DataType odt = engine_->getTensorDataType(name.c_str());
            const size_t od_elem_bytes = tensorElementSize(odt);
            if (od_elem_bytes == 0)
                return false;
            const size_t od_elems = static_cast<size_t>(od.d[1]) * od.d[2] * od.d[3];
            void *tmp = nullptr;
            if (!allocDeviceByName(name, od_elems * od_elem_bytes, tmp))
                return false;
        }
        // 将输入数据从主机内存复制到设备内存
        // cudaMemcpy(in0_dev, in0.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) 表示将 in0 从主机内存复制到设备内存

        if (cudaMemcpy(in0_dev, in0.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(in1_dev, in1.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return false;

        if (!context_->setTensorAddress(input0_name_.c_str(), in0_dev) ||
            !context_->setTensorAddress(input1_name_.c_str(), in1_dev))
            return false;
        for (const auto &name : output_names_)
        {
            void *ptr = nullptr;
            if (!getDevicePtrByName(name, ptr))
                return false;
            if (!context_->setTensorAddress(name.c_str(), ptr))
                return false;
        }
        if (!context_->enqueueV3(0)) // 返回失败，或 engine 文件打不开
            return false;           // 返回 false

        std::cout << "[PROFILE][RaftEngineTRT][infer] selected_output=" << selected_output_name
                  << " dtype=" << static_cast<int>(primary_dtype)
                  << " output_hw=" << out_h << "x" << out_w
                  << " input_hw=" << src_h << "x" << src_w
                  << " runtime_hw=" << runtime_h << "x" << runtime_w
                  << " pad(lrtb)=" << pad_left << "," << pad_right << "," << pad_top << "," << pad_bottom
                  << std::endl;

        if (primary_dtype == nvinfer1::DataType::kFLOAT)
        {
            if (cudaMemcpy(out.data(), out_dev, out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
        }
        else if (primary_dtype == nvinfer1::DataType::kHALF)
        {
            std::vector<uint16_t> out_half(out_elems, 0);
            if (cudaMemcpy(out_half.data(), out_dev, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
            for (size_t i = 0; i < out_elems; ++i)
                out[i] = halfToFloat(out_half[i]);
        }
        else
        {
            std::cerr << "[ERROR][RaftEngineTRT] Unsupported output dtype for flow tensor." << std::endl;
            return false;
        }
#else
        void *bindings[3] = {nullptr, nullptr, nullptr};
        if (!allocDevice(input0_idx_, in_elems * sizeof(float), bindings[input0_idx_]) ||
            !allocDevice(input1_idx_, in_elems * sizeof(float), bindings[input1_idx_]) ||
            !allocDevice(output_idx_, out_elems * sizeof(float), bindings[output_idx_]))
            return false;

        if (cudaMemcpy(bindings[input0_idx_], in0.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(bindings[input1_idx_], in1.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return false;

        if (!context_->enqueueV2(bindings, 0, nullptr))
            return false;

        if (cudaMemcpy(out.data(), bindings[output_idx_], out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
#endif

        flow_hw2 = cv::Mat(out_h, out_w, CV_32FC2, cv::Scalar(0, 0));
        for (int yy = 0; yy < out_h; ++yy)
        {
            cv::Vec2f *row = flow_hw2.ptr<cv::Vec2f>(yy);
            for (int xx = 0; xx < out_w; ++xx)
            {
                const size_t base = static_cast<size_t>(yy) * out_w + xx;
                row[xx][0] = out[base];
                row[xx][1] = out[static_cast<size_t>(out_h) * out_w + base];
            }
        }

        // Keep RAFT dense flow on the original padded grid (before static-dim adaptation)
        // so downstream sparse sampling scale remains consistent with prior behavior.
        if (flow_hw2.rows != h || flow_hw2.cols != w)
        {
            const int copy_h = std::min(flow_hw2.rows, h);
            const int copy_w = std::min(flow_hw2.cols, w);
            cv::Mat flow_target(h, w, CV_32FC2, cv::Scalar(0, 0));
            flow_hw2(cv::Rect(0, 0, copy_w, copy_h)).copyTo(flow_target(cv::Rect(0, 0, copy_w, copy_h)));
            flow_hw2 = flow_target;
        }
        if (src_h > flow_hw2.rows || src_w > flow_hw2.cols)
            return false;
        // Keep padded flow resolution for sparse sampling to match legacy Python RAFT matcher behavior.
        return true;
    }

private:
    struct TRTDestroy
    {
        template <typename T>
        void operator()(T *obj) const
        {
            if (obj)
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
                delete obj;
#else
                obj->destroy();
#endif
        }
    };

    void reset()
    {
        if (input0_dev_)
            cudaFree(input0_dev_);
        if (input1_dev_)
            cudaFree(input1_dev_);
        input0_dev_ = input1_dev_ = nullptr;
        input0_bytes_ = input1_bytes_ = 0;
        for (void *ptr : output_devs_)
        {
            if (ptr)
                cudaFree(ptr);
        }
        output_devs_.clear();
        output_bytes_.clear();
        context_.reset();
        engine_.reset();
        runtime_.reset();
        initialized_ = false;
        input0_name_.clear();
        input1_name_.clear();
        output_names_.clear();
        primary_output_name_.clear();
#if !defined(NV_TENSORRT_MAJOR) || (NV_TENSORRT_MAJOR < 10)
        input0_idx_ = input1_idx_ = output_idx_ = -1;
#endif
    }

    bool resolveBindings()
    {
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
        const int nb = static_cast<int>(engine_->getNbIOTensors());
        if (nb < 3)
            return false;
        for (int i = 0; i < nb; ++i)
        {
            const char *tname = engine_->getIOTensorName(i);
            if (!tname)
                continue;
            const std::string name = toLower(tname);
            const auto mode = engine_->getTensorIOMode(tname);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                if (name.find("image1") != std::string::npos || (input0_name_.empty() && name.find("input") != std::string::npos))
                    input0_name_ = tname;
                else if (name.find("image2") != std::string::npos || input1_name_.empty())
                    input1_name_ = tname;
            }
            else
            {
                output_names_.push_back(tname);
            }
        }
        if (output_names_.empty())
            return false;
        primary_output_name_ = output_names_.front();
        for (const auto &n : output_names_)
        {
            if (toLower(n).find("flow_up") != std::string::npos)
            {
                primary_output_name_ = n;
                break;
            }
        }
        output_devs_.assign(output_names_.size(), nullptr);
        output_bytes_.assign(output_names_.size(), 0);
        return !input0_name_.empty() && !input1_name_.empty();
#else
        const int nb = engine_->getNbBindings();
        if (nb < 3)
            return false;

        for (int i = 0; i < nb; ++i)
        {
            const std::string name = toLower(engine_->getBindingName(i));
            if (engine_->bindingIsInput(i))
            {
                if (name.find("image1") != std::string::npos || (input0_idx_ < 0 && name.find("input") != std::string::npos))
                    input0_idx_ = i;
                else if (name.find("image2") != std::string::npos || input1_idx_ < 0)
                    input1_idx_ = i;
            }
            else
            {
                if (name.find("flow_up") != std::string::npos || output_idx_ < 0)
                    output_idx_ = i;
            }
        }

        return input0_idx_ >= 0 && input1_idx_ >= 0 && output_idx_ >= 0;
#endif
    }

#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
    bool allocDeviceByName(const std::string &name, size_t bytes, void *&out_ptr)
    {
        void **slot = nullptr;
        size_t *allocated = nullptr;
        if (name == input0_name_)
        {
            slot = &input0_dev_;
            allocated = &input0_bytes_;
        }
        else if (name == input1_name_)
        {
            slot = &input1_dev_;
            allocated = &input1_bytes_;
        }
        else
        {
            for (size_t i = 0; i < output_names_.size(); ++i)
            {
                if (name == output_names_[i])
                {
                    if (output_devs_[i] && output_bytes_[i] >= bytes)
                    {
                        out_ptr = output_devs_[i];
                        return true;
                    }
                    if (output_devs_[i])
                        cudaFree(output_devs_[i]);
                    void *ptr = nullptr;
                    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
                        return false;
                    output_devs_[i] = ptr;
                    output_bytes_[i] = bytes;
                    out_ptr = ptr;
                    return true;
                }
            }
            return false;
        }

        if (*slot && *allocated >= bytes)
        {
            out_ptr = *slot;
            return true;
        }

        if (*slot)
            cudaFree(*slot);

        void *ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
            return false;

        *slot = ptr;
        *allocated = bytes;
        out_ptr = ptr;
        return true;
    }

    bool getDevicePtrByName(const std::string &name, void *&out_ptr)
    {
        if (name == input0_name_)
        {
            out_ptr = input0_dev_;
            return out_ptr != nullptr;
        }
        if (name == input1_name_)
        {
            out_ptr = input1_dev_;
            return out_ptr != nullptr;
        }
        for (size_t i = 0; i < output_names_.size(); ++i)
        {
            if (name == output_names_[i])
            {
                out_ptr = output_devs_[i];
                return out_ptr != nullptr;
            }
        }
        return false;
    }
#else
    bool allocDevice(int binding_idx, size_t bytes, void *&out_ptr)
    {
        if (binding_idx < 0 || static_cast<size_t>(binding_idx) >= device_buffers_.size())
            return false;
        if (device_buffers_[binding_idx] && allocated_bytes_[binding_idx] >= bytes)
        {
            out_ptr = device_buffers_[binding_idx];
            return true;
        }

        if (device_buffers_[binding_idx])
            cudaFree(device_buffers_[binding_idx]);

        void *ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
            return false;

        device_buffers_[binding_idx] = ptr;
        allocated_bytes_[binding_idx] = bytes;
        out_ptr = ptr;
        return true;
    }
#endif

    static void toCHWFloat(const cv::Mat &img, std::vector<float> &out)
    {
        cv::Mat rgb;
        if (img.channels() == 3)
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        else
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);

        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3);

        const int h = f32.rows;
        const int w = f32.cols;
        out.resize(static_cast<size_t>(3) * h * w);
        for (int y = 0; y < h; ++y)
        {
            const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; ++x)
            {
                const size_t hw = static_cast<size_t>(y) * w + x;
                out[hw] = row[x][0];
                out[static_cast<size_t>(h) * w + hw] = row[x][1];
                out[2ull * h * w + hw] = row[x][2];
            }
        }
    }

private:
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_;

    std::string input0_name_;
    std::string input1_name_;
    std::vector<std::string> output_names_;
    std::string primary_output_name_;
    void *input0_dev_ = nullptr;
    void *input1_dev_ = nullptr;
    size_t input0_bytes_ = 0;
    size_t input1_bytes_ = 0;
    std::vector<void *> output_devs_;
    std::vector<size_t> output_bytes_;

    std::string engine_path_;
    bool initialized_ = false;
#if !defined(NV_TENSORRT_MAJOR) || (NV_TENSORRT_MAJOR < 10)
    int input0_idx_ = -1;
    int input1_idx_ = -1;
    int output_idx_ = -1;
#endif
};
#endif
} // namespace

bool RaftEngineTRT::estimateMatchesBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<std::vector<cv::Point2f>> &batch_points,
    std::vector<SparseFlowMatches> &batch_matches)
{
    const std::string backend = envOrDefault("DEGRAF_RAFT_BACKEND", "trt");
    const bool allow_lk_fallback = envEnabled("DEGRAF_ALLOW_LK_FALLBACK", false);
    const std::string sample_mode = toLower(envOrDefault("DEGRAF_RAFT_SAMPLE_MODE", "legacy"));
    const bool pad_aware_sampling = (sample_mode == "pad_aware");
    const bool require_dst_in_bounds = envEnabled("DEGRAF_RAFT_REQUIRE_DST_IN_BOUNDS", false);
    const bool require_sample_in_bounds = envEnabled("DEGRAF_RAFT_REQUIRE_SAMPLE_IN_BOUNDS", false);
    const std::string sample_padding = toLower(envOrDefault("DEGRAF_RAFT_SAMPLE_PADDING", "zeros"));
    const bool sample_zero_padding = (sample_padding != "replicate");

    if (backend == "lk" || backend == "lk_fallback")
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);

    if (backend != "trt")
    {
        std::cerr << "[ERROR][RaftEngineTRT] Unsupported backend: " << backend
                  << " (expected trt or lk)" << std::endl;
        return false;
    }

    const std::string engine_path = envOrDefault("DEGRAF_RAFT_ENGINE_PATH", "");
    std::cout << "[PROFILE][RaftEngineTRT] backend=trt"
              << " frames=" << batch_i1.size() << std::endl;
    int max_flow_length = 0;
    envInt("DEGRAF_RAFT_MAX_FLOW_LENGTH", max_flow_length);
    std::cout << "[PROFILE][RaftEngineTRT] max_flow_length="
              << (max_flow_length > 0 ? std::to_string(max_flow_length) : std::string("off"))
              << std::endl;
    std::cout << "[PROFILE][RaftEngineTRT] sample_mode="
              << (pad_aware_sampling ? "pad_aware" : "legacy")
              << std::endl;
    std::cout << "[PROFILE][RaftEngineTRT] require_dst_in_bounds="
              << (require_dst_in_bounds ? "1" : "0")
              << std::endl;
    std::cout << "[PROFILE][RaftEngineTRT] require_sample_in_bounds="
              << (require_sample_in_bounds ? "1" : "0")
              << std::endl;
    std::cout << "[PROFILE][RaftEngineTRT] sample_padding="
              << (sample_zero_padding ? "zeros" : "replicate")
              << std::endl;

    if (engine_path.empty() || !cv::utils::fs::exists(engine_path))
    {
        std::cerr << "[ERROR][RaftEngineTRT] Missing or invalid DEGRAF_RAFT_ENGINE_PATH: "
                  << (engine_path.empty() ? "<empty>" : engine_path) << std::endl;
        if (!allow_lk_fallback)
            return false;
        std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
    }

#if DEGRAF_HAVE_TENSORRT
    // Keep TRT runner alive for process lifetime to avoid teardown-order crashes
    // between CUDA runtime and static object destruction at program exit.
    static RaftTrtRunner *runner = new RaftTrtRunner();
    if (!runner->init(engine_path))
    {
        std::cerr << "[ERROR][RaftEngineTRT] Failed to initialize TensorRT engine: " << engine_path << std::endl;
        if (!allow_lk_fallback)
            return false;
        std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
    }

    batch_matches.clear();
    batch_matches.reserve(batch_i1.size());
    double total_trt_ms = 0.0;
    const std::string dump_dir = envOrDefault("DEGRAF_DEBUG_DUMP_MATCHES_DIR", "");
    if (!dump_dir.empty() && !cv::utils::fs::exists(dump_dir))
    {
        if (!cv::utils::fs::createDirectories(dump_dir))
        {
            std::cerr << "[WARN][RaftEngineTRT] Cannot create DEGRAF_DEBUG_DUMP_MATCHES_DIR: " << dump_dir << std::endl;
        }
    }
    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        const auto &src = batch_points[i];
        SparseFlowMatches matches;
        matches.src_points.reserve(src.size());
        matches.dst_points.reserve(src.size());

        cv::Mat dense_flow;
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!runner->inferDenseFlow(batch_i1[i], batch_i2[i], dense_flow))
        {
            std::cerr << "[ERROR][RaftEngineTRT] TensorRT inference failed on frame " << i << std::endl;
            if (!allow_lk_fallback)
                return false;
            std::cerr << "[WARN][RaftEngineTRT] TensorRT failed; switching whole batch to LK fallback." << std::endl;
            return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double trt_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        total_trt_ms += trt_ms;
        // flow_chw 代表整个光流表，2 层，每层是 dense_flow.rows * dense_flow.cols 个数
        std::vector<float> flow_chw(static_cast<size_t>(2) * dense_flow.rows * dense_flow.cols, 0.0f); // flow_chw 表示 2 层，每层是 dense_flow.rows * dense_flow.cols 个数
        for (int yy = 0; yy < dense_flow.rows; ++yy)
        {
            const cv::Vec2f *row = dense_flow.ptr<cv::Vec2f>(yy);
            for (int xx = 0; xx < dense_flow.cols; ++xx)
            {
                const size_t hw = static_cast<size_t>(yy) * dense_flow.cols + xx;
                flow_chw[hw] = row[xx][0]; // 获取x方向的偏移量，放进第 0 层，第 0 层是x方向的偏移量
                flow_chw[static_cast<size_t>(dense_flow.rows) * dense_flow.cols + hw] = row[xx][1]; // 获取y方向的偏移量，放进第 1 层，第 1 层是y方向的偏移量
            }
        }

        const float sx = static_cast<float>(dense_flow.cols) / static_cast<float>(batch_i1[i].cols);
        const float sy = static_cast<float>(dense_flow.rows) / static_cast<float>(batch_i1[i].rows);
        const float inv_sx = (sx > 1e-6f) ? (1.0f / sx) : 1.0f;
        const float inv_sy = (sy > 1e-6f) ? (1.0f / sy) : 1.0f;
        const int src_w = batch_i1[i].cols;
        const int src_h = batch_i1[i].rows;
        const int pad_w = std::max(0, dense_flow.cols - src_w);
        const int pad_left = pad_w / 2;
        const int pad_top = 0; // InputPadder(mode='kitti') behavior.
        for (const auto &p : src)
        {
            float fx = p.x * sx;
            float fy = p.y * sy;
            if (pad_aware_sampling)
            {
                fx = p.x + static_cast<float>(pad_left);
                fy = p.y + static_cast<float>(pad_top);
            }
            // Python RAFT path uses grid_sample (out-of-range still produces values).
            // Keep this optional filter only for robustness experiments.
            if (require_sample_in_bounds &&
                (fx < 0 || fy < 0 || fx >= dense_flow.cols || fy >= dense_flow.rows))
                continue;
            // bilinearSample 一次只从一层里读出一个数，读哪一层由 channel * h * w 决定：
            const float du = bilinearSample(flow_chw, 0, dense_flow.rows, dense_flow.cols, fy, fx, sample_zero_padding); // 获取x方向的偏移量
            const float dv = bilinearSample(flow_chw, 1, dense_flow.rows, dense_flow.cols, fy, fx, sample_zero_padding); // 获取y方向的偏移量

            // Legacy mode follows existing scale-based mapping.
            // pad_aware mode samples at InputPadder(kitti) coordinates directly.
            const cv::Point2f d(
                p.x + (pad_aware_sampling ? du : (du * inv_sx)),
                p.y + (pad_aware_sampling ? dv : (dv * inv_sy)));
            const float dx = p.x - d.x;
            const float dy = p.y - d.y;
            if (max_flow_length > 0 &&
                std::sqrt(dx * dx + dy * dy) >= static_cast<float>(max_flow_length))
                continue;
            if (require_dst_in_bounds &&
                (d.x < 0 || d.y < 0 || d.x >= batch_i2[i].cols || d.y >= batch_i2[i].rows))
                continue;
            // 将源图像中的特征点和平移后的特征点加入到匹配结果中
            // p 是第 10 帧上的特征点，d 是查表加出来的终点。它们分别放进 matches 的两份名单：
            // src_points[0] 配 dst_points[0]，src_points[1] 配 dst_points[1]。
            // 上面用 continue 跳过的点不会进这两份名单。
            matches.src_points.push_back(p); 
            matches.dst_points.push_back(d);
        }

        std::cout << "[PROFILE][RaftEngineTRT][Frame " << i << "] src_points=" << src.size()
                  << " matched_points=" << matches.dst_points.size()
                  << " trt_ms=" << trt_ms << std::endl;
        if (!dump_dir.empty() && cv::utils::fs::exists(dump_dir))
        {
            const std::string out_path =
                cv::utils::fs::join(dump_dir, cv::format("cpp_trt_matches_frame_%06zu.txt", i));
            if (!dumpMatchesToFile(out_path, matches))
            {
                std::cerr << "[WARN][RaftEngineTRT] Failed to dump matches: " << out_path << std::endl;
            }
            else
            {
                std::cout << "[PROFILE][RaftEngineTRT][Frame " << i << "] dumped_matches=" << out_path << std::endl;
            }
        }
        batch_matches.push_back(std::move(matches));
    }
    std::cout << "[PROFILE][RaftEngineTRT] backend=trt total_trt_ms=" << total_trt_ms << std::endl;
    return true;
#else
    std::cerr << "[ERROR][RaftEngineTRT] This binary was built without TensorRT support (DEGRAF_HAVE_TENSORRT=0)."
              << " Rebuild with TensorRT enabled." << std::endl;
#endif
    if (!allow_lk_fallback)
        return false;
    std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
    return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
}

