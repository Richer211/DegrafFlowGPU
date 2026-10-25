#include "inference/InterpoNetEngineTRT.h"

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
#include "image.h"
#include "variational.h"
#endif

#if DEGRAF_HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#endif

namespace
{
std::string getProjectRoot()
{
    const char *env_path = std::getenv("DEGRAF_PROJECT_ROOT");
    if (env_path && std::string(env_path).size() > 0)
        return std::string(env_path);
    if (cv::utils::fs::exists("/root/autodl-tmp/projects/DegrafFlowGPU"))
        return "/root/autodl-tmp/projects/DegrafFlowGPU";
    return ".";
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

std::string envOrDefault(const char *name, const std::string &fallback)
{
    const char *value = std::getenv(name);
    if (value && std::string(value).size() > 0)
        return std::string(value);
    return fallback;
}

std::string toLower(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(c));
    return s;
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

bool envFloat(const char *name, float &out_value)
{
    const char *value = std::getenv(name);
    if (!value || std::string(value).empty())
        return false;
    try
    {
        out_value = std::stof(value);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

struct VariationalOverride
{
    bool enabled = false;
    int niter_outer = -1;
    float gamma = -1.0f;
    float sigma = -1.0f;
};

void resizeInterpoInput2Ch(const std::vector<float> &src, int src_h, int src_w,
                           std::vector<float> &dst, int dst_h, int dst_w)
{
    dst.assign(static_cast<size_t>(dst_h) * dst_w * 2, 0.0f);
    const int copy_h = std::min(src_h, dst_h);
    const int copy_w = std::min(src_w, dst_w);
    for (int y = 0; y < copy_h; ++y)
    {
        for (int x = 0; x < copy_w; ++x)
        {
            const size_t s = (static_cast<size_t>(y) * src_w + x) * 2;
            const size_t d = (static_cast<size_t>(y) * dst_w + x) * 2;
            dst[d + 0] = src[s + 0];
            dst[d + 1] = src[s + 1];
        }
    }
}

void resizeInterpoInput1Ch(const std::vector<float> &src, int src_h, int src_w,
                           std::vector<float> &dst, int dst_h, int dst_w, float fill_value)
{
    dst.assign(static_cast<size_t>(dst_h) * dst_w, fill_value);
    const int copy_h = std::min(src_h, dst_h);
    const int copy_w = std::min(src_w, dst_w);
    for (int y = 0; y < copy_h; ++y)
    {
        for (int x = 0; x < copy_w; ++x)
        {
            const size_t s = static_cast<size_t>(y) * src_w + x;
            const size_t d = static_cast<size_t>(y) * dst_w + x;
            dst[d] = src[s];
        }
    }
}

bool inferEdgesDatDimsFromElementCount(std::streamoff elements, int target_w, int target_h, int &out_w, int &out_h)
{
    if (elements <= 0 || target_w <= 0 || target_h <= 0)
        return false;
    const double target_ratio = static_cast<double>(target_w) / static_cast<double>(target_h);
    double best_score = std::numeric_limits<double>::max();
    int best_w = -1;
    int best_h = -1;

    for (std::streamoff h = 1; h * h <= elements; ++h)
    {
        if (elements % h != 0)
            continue;
        const std::streamoff w = elements / h;
        const int cand_w1 = static_cast<int>(w);
        const int cand_h1 = static_cast<int>(h);
        const int cand_w2 = static_cast<int>(h);
        const int cand_h2 = static_cast<int>(w);

        auto try_candidate = [&](int cw, int ch) {
            if (cw <= 0 || ch <= 0)
                return;
            const double ratio = static_cast<double>(cw) / static_cast<double>(ch);
            const double score = std::fabs(ratio - target_ratio);
            if (score < best_score)
            {
                best_score = score;
                best_w = cw;
                best_h = ch;
            }
        };

        try_candidate(cand_w1, cand_h1);
        try_candidate(cand_w2, cand_h2);
    }

    if (best_w <= 0 || best_h <= 0)
        return false;
    out_w = best_w;
    out_h = best_h;
    return true;
}

bool loadEdgesDatFile(const std::string &path, int width, int height, cv::Mat &edges_out)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    ifs.seekg(0, std::ios::end);
    const std::streamoff sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (sz <= 0 || (sz % static_cast<std::streamoff>(sizeof(float))) != 0)
        return false;

    const std::streamoff float_count = sz / static_cast<std::streamoff>(sizeof(float));
    int src_w = width;
    int src_h = height;
    if (float_count != static_cast<std::streamoff>(width) * height)
    {
        if (!inferEdgesDatDimsFromElementCount(float_count, width, height, src_w, src_h))
            return false;
    }

    cv::Mat src(src_h, src_w, CV_32FC1);
    ifs.read(reinterpret_cast<char *>(src.data), sz);
    if (!ifs.good())
        return false;

    if (src_w == width && src_h == height)
    {
        edges_out = src;
    }
    else
    {
        cv::resize(src, edges_out, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        static bool logged_edges_resize_once = false;
        if (!logged_edges_resize_once)
        {
            std::cerr << "[WARN][InterpoNetEngineTRT] Resized SED edges.dat from "
                      << src_w << "x" << src_h << " to " << width << "x" << height
                      << " to match current frame size." << std::endl;
            logged_edges_resize_once = true;
        }
    }
    return ifs.good();
}

#if DEGRAF_HAVE_TENSORRT
class TrtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > Severity::kWARNING)
            return;
        std::cerr << "[TensorRT][InterpoNet] " << msg << std::endl;
    }
};

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

class InterpoNetTrtRunner
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

        std::vector<char> data(static_cast<size_t>(size));
        file.read(data.data(), size);
        if (!file.good())
            return false;

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
            return false;
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_)
            return false;
        context_.reset(engine_->createExecutionContext());
        if (!context_)
            return false;

        if (!resolveBindings())
            return false;

        initialized_ = true;
        return true;
    }

    bool inferFlow(
        const std::vector<float> &input_image_nhwc,
        const std::vector<float> &input_mask_nhwc,
        const std::vector<float> &input_edges_nhwc,
        int h, int w,
        cv::Mat &out_flow_hw2)
    {
        if (!initialized_ || h <= 0 || w <= 0)
            return false;
        const size_t img_elems_src = static_cast<size_t>(h) * w * 2;
        const size_t mask_elems_src = static_cast<size_t>(h) * w;
        const size_t edges_elems_src = static_cast<size_t>(h) * w;
        if (input_image_nhwc.size() != img_elems_src ||
            input_mask_nhwc.size() != mask_elems_src ||
            input_edges_nhwc.size() != edges_elems_src)
            return false;

#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
        int run_h = h;
        int run_w = w;
        const nvinfer1::Dims model_img_dims = engine_->getTensorShape(image_name_.c_str());
        if (model_img_dims.nbDims == 4 && model_img_dims.d[1] > 0 && model_img_dims.d[2] > 0)
        {
            run_h = model_img_dims.d[1];
            run_w = model_img_dims.d[2];
        }

        std::vector<float> image_for_trt;
        std::vector<float> mask_for_trt;
        std::vector<float> edges_for_trt;
        const std::vector<float> *image_ptr = &input_image_nhwc;
        const std::vector<float> *mask_ptr = &input_mask_nhwc;
        const std::vector<float> *edges_ptr = &input_edges_nhwc;
        if (run_h != h || run_w != w)
        {
            static bool logged_shape_adapt_once = false;
            if (!logged_shape_adapt_once)
            {
                std::cerr << "[WARN][InterpoNetEngineTRT] Input low-res shape " << h << "x" << w
                          << " mismatches static TRT shape " << run_h << "x" << run_w
                          << ". Adapting by crop/pad to keep TRT path active." << std::endl;
                logged_shape_adapt_once = true;
            }
            resizeInterpoInput2Ch(input_image_nhwc, h, w, image_for_trt, run_h, run_w);
            resizeInterpoInput1Ch(input_mask_nhwc, h, w, mask_for_trt, run_h, run_w, -1.0f);
            resizeInterpoInput1Ch(input_edges_nhwc, h, w, edges_for_trt, run_h, run_w, 0.0f);
            image_ptr = &image_for_trt;
            mask_ptr = &mask_for_trt;
            edges_ptr = &edges_for_trt;
        }

        const size_t img_elems = static_cast<size_t>(run_h) * run_w * 2;
        const size_t mask_elems = static_cast<size_t>(run_h) * run_w;
        const size_t edges_elems = static_cast<size_t>(run_h) * run_w;
        const nvinfer1::Dims4 img_dims(1, run_h, run_w, 2);
        const nvinfer1::Dims4 mask_dims(1, run_h, run_w, 1);
        const nvinfer1::Dims4 edges_dims(1, run_h, run_w, 1);
        if (!context_->setInputShape(image_name_.c_str(), img_dims) ||
            !context_->setInputShape(mask_name_.c_str(), mask_dims) ||
            !context_->setInputShape(edges_name_.c_str(), edges_dims))
            return false;

        const nvinfer1::Dims out_dims = context_->getTensorShape(output_name_.c_str());
        if (out_dims.nbDims != 4)
            return false;
        const int d1 = out_dims.d[1];
        const int d2 = out_dims.d[2];
        const int d3 = out_dims.d[3];
        if (d1 <= 0 || d2 <= 0 || d3 <= 0)
            return false;
        const bool out_nchw = (d1 == 2);
        const bool out_nhwc = (d3 == 2);
        if (!out_nchw && !out_nhwc)
            return false;
        const int out_h = out_nchw ? d2 : d1;
        const int out_w = out_nchw ? d3 : d2;
        const size_t out_elems = static_cast<size_t>(out_h) * out_w * 2;
        const auto out_dtype = engine_->getTensorDataType(output_name_.c_str());
        const size_t out_bytes_per_elem = tensorElementSize(out_dtype);
        if (out_bytes_per_elem == 0)
            return false;

        void *img_dev = nullptr;
        void *mask_dev = nullptr;
        void *edges_dev = nullptr;
        void *out_dev = nullptr;
        if (!allocDeviceByName(image_name_, img_elems * sizeof(float), img_dev) ||
            !allocDeviceByName(mask_name_, mask_elems * sizeof(float), mask_dev) ||
            !allocDeviceByName(edges_name_, edges_elems * sizeof(float), edges_dev) ||
            !allocDeviceByName(output_name_, out_elems * out_bytes_per_elem, out_dev))
            return false;

        if (cudaMemcpy(img_dev, image_ptr->data(), img_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(mask_dev, mask_ptr->data(), mask_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(edges_dev, edges_ptr->data(), edges_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return false;

        if (!context_->setTensorAddress(image_name_.c_str(), img_dev) ||
            !context_->setTensorAddress(mask_name_.c_str(), mask_dev) ||
            !context_->setTensorAddress(edges_name_.c_str(), edges_dev) ||
            !context_->setTensorAddress(output_name_.c_str(), out_dev))
            return false;
        if (!context_->enqueueV3(0))
            return false;

        std::vector<float> out_f32(out_elems, 0.0f);
        if (out_dtype == nvinfer1::DataType::kFLOAT)
        {
            if (cudaMemcpy(out_f32.data(), out_dev, out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
        }
        else if (out_dtype == nvinfer1::DataType::kHALF)
        {
            std::vector<uint16_t> out_half(out_elems, 0);
            if (cudaMemcpy(out_half.data(), out_dev, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
            for (size_t i = 0; i < out_elems; ++i)
                out_f32[i] = halfToFloat(out_half[i]);
        }
        else
        {
            return false;
        }

        out_flow_hw2 = cv::Mat(out_h, out_w, CV_32FC2, cv::Scalar(0, 0));
        for (int y = 0; y < out_h; ++y)
        {
            cv::Vec2f *row = out_flow_hw2.ptr<cv::Vec2f>(y);
            for (int x = 0; x < out_w; ++x)
            {
                if (out_nhwc)
                {
                    const size_t base = (static_cast<size_t>(y) * out_w + x) * 2;
                    row[x][0] = out_f32[base + 0];
                    row[x][1] = out_f32[base + 1];
                }
                else
                {
                    const size_t hw = static_cast<size_t>(y) * out_w + x;
                    row[x][0] = out_f32[hw];
                    row[x][1] = out_f32[static_cast<size_t>(out_h) * out_w + hw];
                }
            }
        }
        return true;
#else
        (void)input_image_nhwc;
        (void)input_mask_nhwc;
        (void)input_edges_nhwc;
        (void)h;
        (void)w;
        (void)out_flow_hw2;
        return false;
#endif
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
        for (auto &kv : device_ptrs_)
        {
            if (kv.second)
                cudaFree(kv.second);
        }
        device_ptrs_.clear();
        alloc_bytes_.clear();
        context_.reset();
        engine_.reset();
        runtime_.reset();
        initialized_ = false;
        image_name_.clear();
        mask_name_.clear();
        edges_name_.clear();
        output_name_.clear();
    }

    bool resolveBindings()
    {
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
        const int nb = static_cast<int>(engine_->getNbIOTensors());
        for (int i = 0; i < nb; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            if (!name)
                continue;
            const std::string low = toLower(name);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                if (low.find("image_ph") != std::string::npos || (image_name_.empty() && low.find("image") != std::string::npos))
                    image_name_ = name;
                else if (low.find("mask_ph") != std::string::npos || (mask_name_.empty() && low.find("mask") != std::string::npos))
                    mask_name_ = name;
                else if (low.find("edges_ph") != std::string::npos || (edges_name_.empty() && low.find("edge") != std::string::npos))
                    edges_name_ = name;
            }
            else
            {
                if (output_name_.empty())
                    output_name_ = name;
            }
        }
        return !image_name_.empty() && !mask_name_.empty() && !edges_name_.empty() && !output_name_.empty();
#else
        return false;
#endif
    }

    bool allocDeviceByName(const std::string &name, size_t bytes, void *&out)
    {
        const auto it = device_ptrs_.find(name);
        if (it != device_ptrs_.end())
        {
            if (alloc_bytes_[name] >= bytes)
            {
                out = it->second;
                return true;
            }
            cudaFree(it->second);
            device_ptrs_.erase(it);
        }
        void *ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
            return false;
        device_ptrs_[name] = ptr;
        alloc_bytes_[name] = bytes;
        out = ptr;
        return true;
    }

private:
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_;
    std::string engine_path_;
    bool initialized_ = false;

    std::string image_name_;
    std::string mask_name_;
    std::string edges_name_;
    std::string output_name_;
    std::map<std::string, void *> device_ptrs_;
    std::map<std::string, size_t> alloc_bytes_;
};
#endif

void buildInterpoNetSparseMapFromMatches(
    const SparseFlowMatches &matches,
    int trim_h,
    int trim_w,
    int downscale,
    int low_h,
    int low_w,
    std::vector<float> &image_nhwc,
    std::vector<float> &mask_nhwc)
{
    image_nhwc.assign(static_cast<size_t>(low_h) * low_w * 2, 0.0f);
    mask_nhwc.assign(static_cast<size_t>(low_h) * low_w, -1.0f);

    // Reproduce Python InterpoNet input semantics:
    // 1) build sparse flow map at full resolution (unknown = NaN, mask=-1)
    // 2) trim to multiples of downscale
    // 3) block nanmean downsample for each channel
    std::vector<float> full_u(static_cast<size_t>(trim_h) * trim_w, 0.0f);
    std::vector<float> full_v(static_cast<size_t>(trim_h) * trim_w, 0.0f);
    std::vector<unsigned char> full_valid(static_cast<size_t>(trim_h) * trim_w, 0);

    const size_t n = std::min(matches.src_points.size(), matches.dst_points.size());
    for (size_t i = 0; i < n; ++i)
    {
        const cv::Point2f &s = matches.src_points[i];
        const cv::Point2f &d = matches.dst_points[i];
        const int x = static_cast<int>(s.x); // numpy.astype(int) semantics used by Python loader
        const int y = static_cast<int>(s.y);
        if (x < 0 || y < 0 || x >= trim_w || y >= trim_h)
            continue;
        const size_t idx = static_cast<size_t>(y) * trim_w + x;
        full_u[idx] = d.x - s.x;
        full_v[idx] = d.y - s.y;
        full_valid[idx] = 1;
    }

    for (int by = 0; by < low_h; ++by)
    {
        for (int bx = 0; bx < low_w; ++bx)
        {
            float sum_u = 0.0f;
            float sum_v = 0.0f;
            int cnt = 0;
            const int y0 = by * downscale;
            const int x0 = bx * downscale;
            for (int yy = y0; yy < y0 + downscale; ++yy)
            {
                for (int xx = x0; xx < x0 + downscale; ++xx)
                {
                    const size_t fi = static_cast<size_t>(yy) * trim_w + xx;
                    if (!full_valid[fi])
                        continue;
                    sum_u += full_u[fi];
                    sum_v += full_v[fi];
                    ++cnt;
                }
            }
            if (cnt <= 0)
                continue;
            const size_t li = static_cast<size_t>(by) * low_w + bx;
            image_nhwc[li * 2 + 0] = sum_u / static_cast<float>(cnt);
            image_nhwc[li * 2 + 1] = sum_v / static_cast<float>(cnt);
            mask_nhwc[li] = 1.0f;
        }
    }
}

void createMeanMapAbBa(
    const std::vector<float> &image_ab,
    const std::vector<float> &mask_ab,
    const std::vector<float> &image_ba,
    const std::vector<float> &mask_ba,
    int low_h,
    int low_w,
    int downscale,
    std::vector<float> &image_out,
    std::vector<float> &mask_out)
{
    image_out.assign(static_cast<size_t>(low_h) * low_w * 2, 0.0f);
    mask_out.assign(static_cast<size_t>(low_h) * low_w, -1.0f);

    std::vector<float> rev_of_map_ba(static_cast<size_t>(low_h) * low_w * 2, 0.0f);
    std::vector<int> map_count(static_cast<size_t>(low_h) * low_w, 0);

    for (int y = 0; y < low_h; ++y)
    {
        for (int x = 0; x < low_w; ++x)
        {
            const size_t idx = static_cast<size_t>(y) * low_w + x;
            if (idx >= mask_ba.size() || mask_ba[idx] <= 0.0f)
                continue;

            const float u = image_ba[idx * 2 + 0];
            const float v = image_ba[idx * 2 + 1];
            const int rev_x = static_cast<int>(std::floor(((static_cast<float>(x) * downscale + u) /
                                                            static_cast<float>(downscale)) + 0.5f));
            const int rev_y = static_cast<int>(std::floor(((static_cast<float>(y) * downscale + v) /
                                                            static_cast<float>(downscale)) + 0.5f));
            // Match original Python implementation, which excludes border coordinates.
            if (!(0 < rev_x && rev_x < low_w && 0 < rev_y && rev_y < low_h))
                continue;

            const size_t rev_idx = static_cast<size_t>(rev_y) * low_w + rev_x;
            map_count[rev_idx] += 1;
            const float count = static_cast<float>(map_count[rev_idx]);
            rev_of_map_ba[rev_idx * 2 + 0] += (-rev_of_map_ba[rev_idx * 2 + 0] - u) / count;
            rev_of_map_ba[rev_idx * 2 + 1] += (-rev_of_map_ba[rev_idx * 2 + 1] - v) / count;
        }
    }

    for (int y = 0; y < low_h; ++y)
    {
        for (int x = 0; x < low_w; ++x)
        {
            const size_t idx = static_cast<size_t>(y) * low_w + x;
            const bool has_ab = idx < mask_ab.size() && mask_ab[idx] > 0.0f;
            const bool has_rev = map_count[idx] > 0;
            if (!has_ab && !has_rev)
                continue;

            mask_out[idx] = 1.0f;
            if (has_ab && has_rev)
            {
                image_out[idx * 2 + 0] = 0.5f * (image_ab[idx * 2 + 0] + rev_of_map_ba[idx * 2 + 0]);
                image_out[idx * 2 + 1] = 0.5f * (image_ab[idx * 2 + 1] + rev_of_map_ba[idx * 2 + 1]);
            }
            else if (has_ab)
            {
                image_out[idx * 2 + 0] = image_ab[idx * 2 + 0];
                image_out[idx * 2 + 1] = image_ab[idx * 2 + 1];
            }
            else
            {
                image_out[idx * 2 + 0] = rev_of_map_ba[idx * 2 + 0];
                image_out[idx * 2 + 1] = rev_of_map_ba[idx * 2 + 1];
            }
        }
    }
}

void buildInterpoNetEdgesInput(
    const cv::Mat &img1,
    int downscale,
    const cv::Mat *precomputed_edges_full,
    const std::string &downscale_mode,
    std::vector<float> &edges_nhwc,
    int &low_h,
    int &low_w)
{
    const int trim_h = img1.rows - (img1.rows % downscale);
    const int trim_w = img1.cols - (img1.cols % downscale);
    low_h = trim_h / downscale;
    low_w = trim_w / downscale;
    edges_nhwc.assign(static_cast<size_t>(low_h) * low_w, 0.0f);

    cv::Mat edges_f;
    if (precomputed_edges_full && !precomputed_edges_full->empty() &&
        precomputed_edges_full->rows == img1.rows && precomputed_edges_full->cols == img1.cols)
    {
        if (precomputed_edges_full->type() == CV_32FC1)
            edges_f = *precomputed_edges_full;
        else
            precomputed_edges_full->convertTo(edges_f, CV_32FC1);
    }
    else
    {
        cv::Mat gray, edges;
        if (img1.channels() == 3)
            cv::cvtColor(img1, gray, cv::COLOR_BGR2GRAY);
        else
            gray = img1;
        cv::Canny(gray, edges, 100, 200);
        edges.convertTo(edges_f, CV_32FC1); // preserve-range semantics (0~255)
    }

    cv::Mat edges_trim = edges_f(cv::Rect(0, 0, trim_w, trim_h));
    cv::Mat edges_low;
    const std::string mode = toLower(downscale_mode);
    if (mode == "python_like")
    {
        // Approximate skimage.transform.rescale(..., preserve_range=True) defaults:
        // bilinear interpolation + anti-aliasing for downsampling.
        const double sigma = std::max(0.0, (static_cast<double>(downscale) - 1.0) * 0.5);
        cv::Mat edges_prefilter = edges_trim;
        if (sigma > 0.0)
        {
            cv::GaussianBlur(edges_trim, edges_prefilter, cv::Size(), sigma, sigma, cv::BORDER_REFLECT_101);
        }
        cv::resize(edges_prefilter, edges_low, cv::Size(low_w, low_h), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        cv::resize(edges_trim, edges_low, cv::Size(low_w, low_h), 0, 0, cv::INTER_AREA);
    }
    for (int y = 0; y < low_h; ++y)
    {
        const float *row = edges_low.ptr<float>(y);
        for (int x = 0; x < low_w; ++x)
            edges_nhwc[static_cast<size_t>(y) * low_w + x] = row[x];
    }
}

void buildInterpoNetInputsFromMatches(
    const cv::Mat &img1,
    const SparseFlowMatches &matches,
    const SparseFlowMatches *matches_ba,
    int downscale,
    const cv::Mat *precomputed_edges_full,
    const std::string &edges_downscale_mode,
    std::vector<float> &image_nhwc,
    std::vector<float> &mask_nhwc,
    std::vector<float> &edges_nhwc,
    int &low_h,
    int &low_w)
{
    const int trim_h = img1.rows - (img1.rows % downscale);
    const int trim_w = img1.cols - (img1.cols % downscale);
    low_h = trim_h / downscale;
    low_w = trim_w / downscale;

    std::vector<float> image_ab, mask_ab;
    buildInterpoNetSparseMapFromMatches(matches, trim_h, trim_w, downscale, low_h, low_w, image_ab, mask_ab);

    if (matches_ba && !matches_ba->src_points.empty() &&
        matches_ba->src_points.size() == matches_ba->dst_points.size())
    {
        std::vector<float> image_ba, mask_ba;
        buildInterpoNetSparseMapFromMatches(*matches_ba, trim_h, trim_w, downscale, low_h, low_w, image_ba, mask_ba);
        createMeanMapAbBa(image_ab, mask_ab, image_ba, mask_ba, low_h, low_w, downscale, image_nhwc, mask_nhwc);
    }
    else
    {
        image_nhwc = std::move(image_ab);
        mask_nhwc = std::move(mask_ab);
    }

    buildInterpoNetEdgesInput(
        img1, downscale, precomputed_edges_full, edges_downscale_mode, edges_nhwc, low_h, low_w);
}

bool densifyWithEpic(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const SparseFlowMatches &matches,
    cv::Mat &dense_flow,
    int k,
    float sigma,
    bool use_post_proc,
    float fgs_lambda,
    float fgs_sigma)
{
    dense_flow = cv::Mat(img1.size(), CV_32FC2, cv::Scalar(0, 0));
    if (matches.src_points.empty() || matches.src_points.size() != matches.dst_points.size())
        return true;
    cv::Ptr<cv::ximgproc::EdgeAwareInterpolator> interpolator =
        cv::ximgproc::createEdgeAwareInterpolator();
    interpolator->setK(k);
    interpolator->setSigma(sigma);
    interpolator->setUsePostProcessing(use_post_proc);
    interpolator->setFGSLambda(fgs_lambda);
    interpolator->setFGSSigma(fgs_sigma);
    interpolator->interpolate(img1, matches.src_points, img2, matches.dst_points, dense_flow);
    return !dense_flow.empty();
}

double computeSparseResidualPx(const cv::Mat &dense_flow, const SparseFlowMatches &matches)
{
    if (dense_flow.empty() || dense_flow.type() != CV_32FC2)
        return -1.0;
    if (matches.src_points.empty() || matches.src_points.size() != matches.dst_points.size())
        return -1.0;

    double sum_epe = 0.0;
    int used = 0;
    for (size_t i = 0; i < matches.src_points.size(); ++i)
    {
        const cv::Point2f &s = matches.src_points[i];
        const cv::Point2f &d = matches.dst_points[i];
        const int x = static_cast<int>(std::round(s.x));
        const int y = static_cast<int>(std::round(s.y));
        if (x < 0 || y < 0 || x >= dense_flow.cols || y >= dense_flow.rows)
            continue;

        const cv::Vec2f &f = dense_flow.at<cv::Vec2f>(y, x);
        const float px = s.x + f[0];
        const float py = s.y + f[1];
        const float ex = px - d.x;
        const float ey = py - d.y;
        sum_epe += std::sqrt(ex * ex + ey * ey);
        ++used;
    }
    if (used == 0)
        return -1.0;
    return sum_epe / static_cast<double>(used);
}

bool writeFlowFile(const std::string &path, const cv::Mat &flow)
{
    if (flow.empty() || flow.type() != CV_32FC2)
        return false;

    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.good())
        return false;

    const float magic = 202021.25f;
    const int width = flow.cols;
    const int height = flow.rows;
    file.write(reinterpret_cast<const char *>(&magic), sizeof(float));
    file.write(reinterpret_cast<const char *>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(&height), sizeof(int));
    file.write(reinterpret_cast<const char *>(flow.data), width * height * 2 * sizeof(float));
    return file.good();
}

bool writeFloatDatFile(const std::string &path, const std::vector<float> &values)
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.good())
        return false;
    if (!values.empty())
    {
        file.write(reinterpret_cast<const char *>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    return file.good();
}

bool dumpInterpoInputTensors(
    const std::string &debug_dump_dir,
    int frame_id,
    const std::vector<float> &image_nhwc,
    const std::vector<float> &mask_nhwc,
    const std::vector<float> &edges_nhwc,
    int low_h,
    int low_w,
    int downscale,
    const std::string &edges_source,
    bool used_ba)
{
    if (debug_dump_dir.empty() || low_h <= 0 || low_w <= 0)
        return false;
    if (image_nhwc.size() != static_cast<size_t>(low_h) * low_w * 2 ||
        mask_nhwc.size() != static_cast<size_t>(low_h) * low_w ||
        edges_nhwc.size() != static_cast<size_t>(low_h) * low_w)
        return false;

    const std::string image_path =
        cv::utils::fs::join(debug_dump_dir, cv::format("%06d_cpp_input_image.flo", frame_id));
    const std::string mask_path =
        cv::utils::fs::join(debug_dump_dir, cv::format("%06d_cpp_input_mask.dat", frame_id));
    const std::string edges_path =
        cv::utils::fs::join(debug_dump_dir, cv::format("%06d_cpp_input_edges.dat", frame_id));
    const std::string meta_path =
        cv::utils::fs::join(debug_dump_dir, cv::format("%06d_cpp_input_meta.txt", frame_id));

    cv::Mat image_mat(low_h, low_w, CV_32FC2, const_cast<float *>(image_nhwc.data()));
    if (!writeFlowFile(image_path, image_mat))
        return false;
    if (!writeFloatDatFile(mask_path, mask_nhwc))
        return false;
    if (!writeFloatDatFile(edges_path, edges_nhwc))
        return false;

    int valid_count = 0;
    for (float v : mask_nhwc)
    {
        if (v > 0.0f)
            ++valid_count;
    }

    std::ofstream meta(meta_path.c_str());
    if (!meta.good())
        return false;
    meta << "low_h=" << low_h << "\n";
    meta << "low_w=" << low_w << "\n";
    meta << "downscale=" << downscale << "\n";
    meta << "mask_valid_count=" << valid_count << "\n";
    meta << "edges_source=" << edges_source << "\n";
    meta << "used_ba=" << (used_ba ? 1 : 0) << "\n";
    return meta.good();
}

cv::Mat readFlowFile(const std::string &path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.good())
        return cv::Mat();

    float magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(float));
    if (magic != 202021.25f)
        return cv::Mat();

    int width = 0, height = 0;
    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));
    if (width <= 0 || height <= 0)
        return cv::Mat();

    cv::Mat flow(height, width, CV_32FC2);
    file.read(reinterpret_cast<char *>(flow.data), width * height * 2 * sizeof(float));
    if (!file.good())
        return cv::Mat();
    return flow;
}

bool runVariationalRefineExternal(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow,
                                  size_t idx, const VariationalOverride *override_params)
{
    const std::string project_root = getProjectRoot();
    const std::string variational_bin =
        envOrDefault("DEGRAF_VARIATIONAL_BIN",
                     project_root + "/external/InterpoNet/SrcVariational/variational_main");
    if (!cv::utils::fs::exists(variational_bin))
        return false;

    const std::string tmp_dir =
        envOrDefault("DEGRAF_VARIATIONAL_TMP_DIR", project_root + "/data/outputs/variational_tmp");
    cv::utils::fs::createDirectories(tmp_dir);

    const std::string dataset = envOrDefault("DEGRAF_VARIATIONAL_DATASET", "kitti");
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string token = std::to_string(static_cast<long long>(stamp)) + "_" + std::to_string(idx);

    const std::string img1_path = tmp_dir + "/" + token + "_a.png";
    const std::string img2_path = tmp_dir + "/" + token + "_b.png";
    const std::string flow_in_path = tmp_dir + "/" + token + "_in.flo";
    const std::string flow_out_path = tmp_dir + "/" + token + "_out.flo";

    if (!cv::imwrite(img1_path, img1) || !cv::imwrite(img2_path, img2))
        return false;
    if (!writeFlowFile(flow_in_path, dense_flow))
        return false;

    std::ostringstream args;
    int niter_outer = 0;
    float alpha = 0.0f, gamma = 0.0f, delta = 0.0f, sigma = 0.0f;
    if (override_params && override_params->enabled && override_params->niter_outer > 0)
        niter_outer = override_params->niter_outer;
    else if (!envInt("DEGRAF_VARIATIONAL_NITER_OUTER", niter_outer))
        niter_outer = 0;
    if (niter_outer > 0)
        args << " -iter " << niter_outer;
    if (envFloat("DEGRAF_VARIATIONAL_ALPHA", alpha))
        args << " -alpha " << alpha;
    if (override_params && override_params->enabled && override_params->gamma >= 0.0f)
        gamma = override_params->gamma;
    else if (!envFloat("DEGRAF_VARIATIONAL_GAMMA", gamma))
        gamma = -1.0f;
    if (gamma >= 0.0f)
        args << " -gamma " << gamma;
    if (envFloat("DEGRAF_VARIATIONAL_DELTA", delta))
        args << " -delta " << delta;
    if (override_params && override_params->enabled && override_params->sigma > 0.0f)
        sigma = override_params->sigma;
    else if (!envFloat("DEGRAF_VARIATIONAL_SIGMA", sigma))
        sigma = -1.0f;
    if (sigma > 0.0f)
        args << " -sigma " << sigma;

    const std::string cmd =
        "\"" + variational_bin + "\" " +
        "\"" + img1_path + "\" " +
        "\"" + img2_path + "\" " +
        "\"" + flow_in_path + "\" " +
        "\"" + flow_out_path + "\" -" + dataset + args.str();

    const int exit_code = std::system(cmd.c_str());
    if (exit_code != 0)
        return false;

    cv::Mat refined = readFlowFile(flow_out_path);
    if (refined.empty() || refined.size() != dense_flow.size() || refined.type() != dense_flow.type())
        return false;

    dense_flow = refined;

    // Keep temp files only when explicitly requested for debugging.
    if (!envEnabled("DEGRAF_KEEP_VARIATIONAL_TMP", false))
    {
        std::remove(img1_path.c_str());
        std::remove(img2_path.c_str());
        std::remove(flow_in_path.c_str());
        std::remove(flow_out_path.c_str());
    }
    return true;
}

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
color_image_t *toVariationalColorImage(const cv::Mat &input)
{
    cv::Mat bgr;
    if (input.channels() == 1)
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    else if (input.channels() == 4)
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
    else
        bgr = input;

    cv::Mat bgr_u8;
    if (bgr.type() != CV_8UC3)
        bgr.convertTo(bgr_u8, CV_8UC3);
    else
        bgr_u8 = bgr;

    color_image_t *out = color_image_new(bgr_u8.cols, bgr_u8.rows);
    for (int y = 0; y < bgr_u8.rows; ++y)
    {
        const cv::Vec3b *row = bgr_u8.ptr<cv::Vec3b>(y);
        const int base = y * out->stride;
        for (int x = 0; x < bgr_u8.cols; ++x)
        {
            // variational code expects RGB layout in c1/c2/c3.
            out->c1[base + x] = static_cast<float>(row[x][2]);
            out->c2[base + x] = static_cast<float>(row[x][1]);
            out->c3[base + x] = static_cast<float>(row[x][0]);
        }
    }
    return out;
}

void flowMatToVariationalImages(const cv::Mat &flow, image_t *wx, image_t *wy)
{
    for (int y = 0; y < flow.rows; ++y)
    {
        const cv::Vec2f *row = flow.ptr<cv::Vec2f>(y);
        const int base = y * wx->stride;
        for (int x = 0; x < flow.cols; ++x)
        {
            wx->data[base + x] = row[x][0];
            wy->data[base + x] = row[x][1];
        }
    }
}

cv::Mat variationalImagesToFlowMat(const image_t *wx, const image_t *wy)
{
    cv::Mat flow(wx->height, wx->width, CV_32FC2, cv::Scalar(0, 0));
    for (int y = 0; y < wx->height; ++y)
    {
        cv::Vec2f *row = flow.ptr<cv::Vec2f>(y);
        const int base = y * wx->stride;
        for (int x = 0; x < wx->width; ++x)
        {
            row[x][0] = wx->data[base + x];
            row[x][1] = wy->data[base + x];
        }
    }
    return flow;
}

void applyPresetFromDataset(variational_params_t &params, const std::string &dataset)
{
    if (dataset == "sintel")
    {
        params.niter_outer = 5;
        params.alpha = 1.0f;
        params.gamma = 0.72f;
        params.delta = 0.0f;
        params.sigma = 1.1f;
    }
    else if (dataset == "middlebury")
    {
        params.niter_outer = 25;
        params.alpha = 1.0f;
        params.gamma = 0.72f;
        params.delta = 0.0f;
        params.sigma = 1.1f;
    }
    else
    {
        // KITTI preset from original variational_main.
        params.niter_outer = 2;
        params.alpha = 1.0f;
        params.gamma = 0.77f;
        params.delta = 0.0f;
        params.sigma = 1.7f;
    }
}

void applyVariationalOverridesFromEnv(variational_params_t &params)
{
    int niter_outer = 0, niter_inner = 0, niter_solver = 0;
    float alpha = 0.0f, gamma = 0.0f, delta = 0.0f, sigma = 0.0f, sor_omega = 0.0f;

    if (envInt("DEGRAF_VARIATIONAL_NITER_OUTER", niter_outer) && niter_outer > 0)
        params.niter_outer = niter_outer;
    if (envInt("DEGRAF_VARIATIONAL_NITER_INNER", niter_inner) && niter_inner > 0)
        params.niter_inner = niter_inner;
    if (envInt("DEGRAF_VARIATIONAL_NITER_SOLVER", niter_solver) && niter_solver > 0)
        params.niter_solver = niter_solver;
    if (envFloat("DEGRAF_VARIATIONAL_ALPHA", alpha) && alpha >= 0.0f)
        params.alpha = alpha;
    if (envFloat("DEGRAF_VARIATIONAL_GAMMA", gamma) && gamma >= 0.0f)
        params.gamma = gamma;
    if (envFloat("DEGRAF_VARIATIONAL_DELTA", delta) && delta >= 0.0f)
        params.delta = delta;
    if (envFloat("DEGRAF_VARIATIONAL_SIGMA", sigma) && sigma > 0.0f)
        params.sigma = sigma;
    if (envFloat("DEGRAF_VARIATIONAL_SOR_OMEGA", sor_omega) && sor_omega > 0.0f)
        params.sor_omega = sor_omega;
}

bool runVariationalRefineInProcess(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow,
                                   const VariationalOverride *override_params)
{
    if (dense_flow.empty() || dense_flow.type() != CV_32FC2)
        return false;

    color_image_t *im1 = toVariationalColorImage(img1);
    color_image_t *im2 = toVariationalColorImage(img2);
    image_t *wx = image_new(dense_flow.cols, dense_flow.rows);
    image_t *wy = image_new(dense_flow.cols, dense_flow.rows);

    flowMatToVariationalImages(dense_flow, wx, wy);

    variational_params_t params;
    variational_params_default(&params);
    applyPresetFromDataset(params, envOrDefault("DEGRAF_VARIATIONAL_DATASET", "kitti"));
    applyVariationalOverridesFromEnv(params);
    if (override_params && override_params->enabled)
    {
        if (override_params->niter_outer > 0)
            params.niter_outer = override_params->niter_outer;
        if (override_params->gamma >= 0.0f)
            params.gamma = override_params->gamma;
        if (override_params->sigma > 0.0f)
            params.sigma = override_params->sigma;
    }
    variational(wx, wy, im1, im2, &params);

    dense_flow = variationalImagesToFlowMat(wx, wy);

    color_image_delete(im1);
    color_image_delete(im2);
    image_delete(wx);
    image_delete(wy);
    return true;
}
#endif

bool runVariationalRefine(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow,
                          size_t idx, const VariationalOverride *override_params)
{
    const std::string mode = envOrDefault("DEGRAF_VARIATIONAL_MODE", "external");
    static bool logged_mode_once = false;
    if (!logged_mode_once)
    {
        std::cout << "[PROFILE][InterpoNetEngineTRT][Variational] requested_mode=" << mode << std::endl;
        logged_mode_once = true;
    }
    if (mode == "external")
    {
        const bool ok = runVariationalRefineExternal(img1, img2, dense_flow, idx, override_params);
        if (!ok)
            std::cerr << "[WARN][InterpoNetEngineTRT][Variational] external refine failed." << std::endl;
        return ok;
    }

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
    if (runVariationalRefineInProcess(img1, img2, dense_flow, override_params))
    {
        static bool logged_inproc_once = false;
        if (!logged_inproc_once)
        {
            std::cout << "[PROFILE][InterpoNetEngineTRT][Variational] using inprocess backend." << std::endl;
            logged_inproc_once = true;
        }
        return true;
    }
    std::cerr << "[WARN][InterpoNetEngineTRT][Variational] inprocess refine failed, fallback to external." << std::endl;
#else
    std::cerr << "[WARN][InterpoNetEngineTRT][Variational] inprocess requested but binary has no inprocess support, fallback to external." << std::endl;
#endif
    const bool ok = runVariationalRefineExternal(img1, img2, dense_flow, idx, override_params);
    if (!ok)
        std::cerr << "[WARN][InterpoNetEngineTRT][Variational] external fallback refine failed." << std::endl;
    return ok;
}
} // namespace

InterpoNetEngineTRT::InterpoNetEngineTRT(int k, float sigma, bool use_post_proc,
                                         float fgs_lambda, float fgs_sigma)
    : k_(k),
      sigma_(sigma),
      use_post_proc_(use_post_proc),
      fgs_lambda_(fgs_lambda),
      fgs_sigma_(fgs_sigma)
{
}

bool InterpoNetEngineTRT::densifyBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<SparseFlowMatches> &batch_matches,
    std::vector<cv::Mat> &batch_flows)
{
    return densifyBatch(batch_i1, batch_i2, batch_matches, nullptr, batch_flows);
}

bool InterpoNetEngineTRT::densifyBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<SparseFlowMatches> &batch_matches,
    const std::vector<SparseFlowMatches> *batch_matches_ba,
    std::vector<cv::Mat> &batch_flows)
{
    const std::string backend = envOrDefault("DEGRAF_INTERPONET_BACKEND", "epic");
    const bool allow_epic_fallback = envEnabled("DEGRAF_ALLOW_EPIC_FALLBACK", true);
    const std::string trt_engine_path = envOrDefault(
        "DEGRAF_INTERPONET_ENGINE_PATH",
        getProjectRoot() + "/external/InterpoNet/models/interponet_kitti_fp32.engine");
    int interponet_downscale = 8;
    envInt("DEGRAF_INTERPONET_DOWNSCALE", interponet_downscale);
    if (interponet_downscale <= 0)
        interponet_downscale = 8;
    const std::string edges_dir = envOrDefault("DEGRAF_INTERPONET_EDGES_DIR", "");
    const bool require_sed_edges = envEnabled("DEGRAF_INTERPONET_REQUIRE_SED", false);
    const std::string edges_downscale_mode =
        envOrDefault("DEGRAF_INTERPONET_EDGES_DOWNSCALE_MODE", "python_like");
    int edges_start_index = 0;
    envInt("DEGRAF_INTERPONET_EDGES_START_INDEX", edges_start_index);
    const std::string debug_dump_dir = envOrDefault("DEGRAF_INTERPONET_DEBUG_DUMP_DIR", "");
    int debug_start_index = 0;
    envInt("DEGRAF_INTERPONET_DEBUG_START_INDEX", debug_start_index);
    const bool debug_dump_lowres = envEnabled("DEGRAF_INTERPONET_DEBUG_DUMP_LOWRES", false);
    const bool debug_dump_inputs = envEnabled("DEGRAF_INTERPONET_DEBUG_DUMP_INPUTS", false);
    if (!debug_dump_dir.empty() && !cv::utils::fs::exists(debug_dump_dir))
        cv::utils::fs::createDirectories(debug_dump_dir);

    const bool enable_variational = envEnabled("DEGRAF_ENABLE_VARIATIONAL", true);
    const bool profile_stages = envEnabled("DEGRAF_PROFILE_STAGES", true);
    const bool residual_gate = envEnabled("DEGRAF_VARIATIONAL_RESIDUAL_GATE", false);
    const bool hard_mode = envEnabled("DEGRAF_VARIATIONAL_HARD_MODE", false);
    float residual_thresh = 3.0f;
    envFloat("DEGRAF_VARIATIONAL_RESIDUAL_THRESH", residual_thresh);
    int hard_min_sparse_points = 4500;
    envInt("DEGRAF_VARIATIONAL_HARD_MIN_POINTS", hard_min_sparse_points);
    float hard_residual_thresh = 6.0f;
    envFloat("DEGRAF_VARIATIONAL_HARD_RESIDUAL_THRESH", hard_residual_thresh);
    int hard_niter_outer = 3;
    envInt("DEGRAF_VARIATIONAL_HARD_NITER_OUTER", hard_niter_outer);
    float hard_gamma = 0.80f;
    envFloat("DEGRAF_VARIATIONAL_HARD_GAMMA", hard_gamma);
    float hard_sigma = 1.9f;
    envFloat("DEGRAF_VARIATIONAL_HARD_SIGMA", hard_sigma);
    if (profile_stages && hard_mode && !residual_gate)
    {
        std::cout << "[PROFILE][InterpoNetEngineTRT][Variational] hard_mode=on but residual_gate=off; "
                  << "all frames may run variational and increase latency." << std::endl;
    }

    batch_flows.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_matches.size())
        return false;
    if (batch_matches_ba && batch_matches_ba->size() != batch_i1.size())
        return false;

    batch_flows.reserve(batch_i1.size());
    double total_interpolate_ms = 0.0;
    double total_variational_ms = 0.0;
    int variational_applied = 0;
    int variational_skipped_by_gate = 0;
    int hard_applied = 0;
    int hard_trigger_by_points = 0;
    int hard_trigger_by_residual = 0;

#if DEGRAF_HAVE_TENSORRT
    // Keep TRT runner alive for process lifetime to avoid teardown-order crashes
    // between CUDA runtime and static/local object destruction at program exit.
    static InterpoNetTrtRunner *trt_runner = new InterpoNetTrtRunner();
    bool trt_ready = false;
    if (backend == "trt")
    {
        trt_ready = trt_runner->init(trt_engine_path);
        if (!trt_ready && !allow_epic_fallback)
        {
            std::cerr << "[ERROR][InterpoNetEngineTRT] Failed to init TRT engine and EPIC fallback disabled: "
                      << trt_engine_path << std::endl;
            return false;
        }
        if (!trt_ready)
        {
            std::cerr << "[WARN][InterpoNetEngineTRT] TRT init failed, fallback to EPIC: "
                      << trt_engine_path << std::endl;
        }
    }
#else
    if (backend == "trt" && !allow_epic_fallback)
    {
        std::cerr << "[ERROR][InterpoNetEngineTRT] Built without TensorRT and EPIC fallback disabled." << std::endl;
        return false;
    }
#endif

    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        cv::Mat dense_flow;

        auto interp_start = std::chrono::high_resolution_clock::now();
        bool interpolate_ok = false;
        bool used_trt = false;
        cv::Mat trt_low_flow_for_dump;
        std::string edges_source = "canny";
        cv::Mat sed_edges;
        std::string sed_try_p1;
        std::string sed_try_p2;
        if (!edges_dir.empty())
        {
            const int frame_id = edges_start_index + static_cast<int>(i);
            sed_try_p1 = cv::format("%s/%06d_edges.dat", edges_dir.c_str(), frame_id);
            sed_try_p2 = cv::format("%s/%06d_10_edges.dat", edges_dir.c_str(), frame_id);
            if (loadEdgesDatFile(sed_try_p1, batch_i1[i].cols, batch_i1[i].rows, sed_edges) ||
                loadEdgesDatFile(sed_try_p2, batch_i1[i].cols, batch_i1[i].rows, sed_edges))
            {
                edges_source = "sed_bin";
            }
        }
        if (require_sed_edges && edges_source != "sed_bin")
        {
            std::cerr << "[ERROR][InterpoNetEngineTRT] DEGRAF_INTERPONET_REQUIRE_SED=1 but SED edges.dat is missing/invalid"
                      << " for frame " << i
                      << " (searched: " << (sed_try_p1.empty() ? "<no_dir>" : sed_try_p1)
                      << ", " << (sed_try_p2.empty() ? "<no_dir>" : sed_try_p2) << ")"
                      << ". Refusing Canny fallback." << std::endl;
            return false;
        }
        if (backend == "trt")
        {
#if DEGRAF_HAVE_TENSORRT
            std::vector<float> image_nhwc, mask_nhwc, edges_nhwc;
            int low_h = 0, low_w = 0;
            const SparseFlowMatches *matches_ba = nullptr;
            if (batch_matches_ba)
                matches_ba = &((*batch_matches_ba)[i]);
            buildInterpoNetInputsFromMatches(
                batch_i1[i],
                batch_matches[i],
                matches_ba,
                interponet_downscale,
                sed_edges.empty() ? nullptr : &sed_edges,
                edges_downscale_mode,
                image_nhwc,
                mask_nhwc,
                edges_nhwc,
                low_h,
                low_w);
            if (debug_dump_inputs && !debug_dump_dir.empty())
            {
                const int frame_id = debug_start_index + static_cast<int>(i);
                if (dumpInterpoInputTensors(
                        debug_dump_dir,
                        frame_id,
                        image_nhwc,
                        mask_nhwc,
                        edges_nhwc,
                        low_h,
                        low_w,
                        interponet_downscale,
                        edges_source,
                        matches_ba != nullptr))
                {
                    std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i
                              << "] dumped_input_tensors=" << debug_dump_dir << std::endl;
                }
            }
            cv::Mat low_flow;
            if (trt_ready && trt_runner->inferFlow(image_nhwc, mask_nhwc, edges_nhwc, low_h, low_w, low_flow))
            {
                cv::resize(low_flow, dense_flow, batch_i1[i].size(), 0, 0, cv::INTER_CUBIC);
                interpolate_ok = !dense_flow.empty();
                used_trt = interpolate_ok;
                if (interpolate_ok)
                    trt_low_flow_for_dump = low_flow;
            }
#endif
            if (!interpolate_ok && allow_epic_fallback)
            {
                interpolate_ok = densifyWithEpic(
                    batch_i1[i], batch_i2[i], batch_matches[i], dense_flow,
                    k_, sigma_, use_post_proc_, fgs_lambda_, fgs_sigma_);
            }
        }
        else
        {
            interpolate_ok = densifyWithEpic(
                batch_i1[i], batch_i2[i], batch_matches[i], dense_flow,
                k_, sigma_, use_post_proc_, fgs_lambda_, fgs_sigma_);
        }
        if (!interpolate_ok)
        {
            std::cerr << "[ERROR][InterpoNetEngineTRT] Dense interpolation failed on frame " << i
                      << " backend=" << backend << std::endl;
            return false;
        }
        if (profile_stages)
        {
            std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i << "] backend="
                      << ((backend == "trt" && used_trt) ? "trt" : "epic")
                      << " downscale=" << interponet_downscale
                      << " edges_source=" << edges_source
                      << " sparse_points=" << batch_matches[i].src_points.size()
                      << " ba_sparse_points=" << ((batch_matches_ba && i < batch_matches_ba->size())
                                                     ? (*batch_matches_ba)[i].src_points.size()
                                                     : 0)
                      << std::endl;
        }

        if (!debug_dump_dir.empty() && cv::utils::fs::exists(debug_dump_dir))
        {
            const int frame_id = debug_start_index + static_cast<int>(i);
            const std::string dump_path =
                cv::utils::fs::join(debug_dump_dir, cv::format("%06d_trt_dense.flo", frame_id));
            if (writeFlowFile(dump_path, dense_flow))
            {
                std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i
                          << "] dumped_dense_flow=" << dump_path << std::endl;
            }
            if (debug_dump_lowres && used_trt && !trt_low_flow_for_dump.empty())
            {
                const std::string low_path =
                    cv::utils::fs::join(debug_dump_dir, cv::format("%06d_trt_low.flo", frame_id));
                if (writeFlowFile(low_path, trt_low_flow_for_dump))
                {
                    std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i
                              << "] dumped_lowres_flow=" << low_path << std::endl;
                }
            }
        }
        auto interp_end = std::chrono::high_resolution_clock::now();
        total_interpolate_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(interp_end - interp_start).count();

        bool should_run_variational = enable_variational && !dense_flow.empty();
        double sparse_residual_px = -1.0;
        if (should_run_variational && (residual_gate || hard_mode))
        {
            sparse_residual_px = computeSparseResidualPx(dense_flow, batch_matches[i]);
        }
        if (should_run_variational && residual_gate)
        {
            if (sparse_residual_px >= 0.0 && sparse_residual_px < residual_thresh)
            {
                should_run_variational = false;
                ++variational_skipped_by_gate;
            }
        }

        bool use_hard_variational = false;
        if (should_run_variational && hard_mode)
        {
            const bool hard_by_points =
                static_cast<int>(batch_matches[i].src_points.size()) < hard_min_sparse_points;
            const bool hard_by_residual =
                (sparse_residual_px >= 0.0 && sparse_residual_px > hard_residual_thresh);
            use_hard_variational = hard_by_points || hard_by_residual;
            if (use_hard_variational)
            {
                ++hard_applied;
                if (hard_by_points)
                    ++hard_trigger_by_points;
                if (hard_by_residual)
                    ++hard_trigger_by_residual;
            }
        }

        if (should_run_variational)
        {
            auto vari_start = std::chrono::high_resolution_clock::now();
            VariationalOverride override_params;
            if (use_hard_variational)
            {
                override_params.enabled = true;
                override_params.niter_outer = hard_niter_outer;
                override_params.gamma = hard_gamma;
                override_params.sigma = hard_sigma;
            }
            runVariationalRefine(batch_i1[i], batch_i2[i], dense_flow, i, &override_params);
            auto vari_end = std::chrono::high_resolution_clock::now();
            total_variational_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(vari_end - vari_start).count();
            ++variational_applied;
        }

        if (profile_stages && (residual_gate || hard_mode))
        {
            std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i << "] residual_px=" << sparse_residual_px
                      << " gate_thresh=" << residual_thresh
                      << " hard_mode=" << (hard_mode ? "on" : "off")
                      << " hard=" << (use_hard_variational ? "on" : "off")
                      << " hard_point_thresh=" << hard_min_sparse_points
                      << " hard_residual_thresh=" << hard_residual_thresh
                      << " variational=" << (should_run_variational ? "on" : "skip")
                      << std::endl;
        }

        batch_flows.push_back(dense_flow);
    }

    if (profile_stages)
    {
        std::cout << "[PROFILE][InterpoNetEngineTRT] frames=" << batch_i1.size()
                  << " interpolate_ms=" << total_interpolate_ms
                  << " variational_ms=" << total_variational_ms
                  << " variational_frames=" << variational_applied
                  << " gate_skipped_frames=" << variational_skipped_by_gate
                  << " hard_frames=" << hard_applied
                  << " hard_by_points=" << hard_trigger_by_points
                  << " hard_by_residual=" << hard_trigger_by_residual
                  << " total_ms=" << (total_interpolate_ms + total_variational_ms)
                  << std::endl;
    }

    return true;
}

