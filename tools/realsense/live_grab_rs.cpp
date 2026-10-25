// Continuous RealSense grabber for Mac live status UI.
// Overwrites a slot directory with the latest aligned color+depth (+calib).
//
// Build (same flags as record_png_seq):
//   c++ -O2 -std=c++14 live_grab_rs.cpp -o live_grab_rs \
//     -I"$HOME/librealsense/include" \
//     -I"$HOME/librealsense/third-party" \
//     -I"$HOME/librealsense/third-party/stb" \
//     -L"$HOME/librealsense/build/Release" \
//     -lrealsense2 -Wl,-rpath,"$HOME/librealsense/build/Release" -lz
//
// Run:
//   sudo ./live_grab_rs --out /tmp/degraf_live
// Then in another terminal:
//   python3 realtime_status.py --mode live --live-dir /tmp/degraf_live

#include <librealsense2/rs.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <sys/stat.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

std::atomic<bool> g_stop{false};

void on_sigint(int) { g_stop = true; }

bool mkdir_p(const std::string& path)
{
    if (path.empty()) return false;
    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) return true;
    return false;
}

void atomic_write_bytes(const std::string& path, const void* data, size_t n)
{
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    f.close();
    std::rename(tmp.c_str(), path.c_str());
}

void atomic_write_text(const std::string& path, const std::string& text)
{
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp);
    f << text;
    f.close();
    std::rename(tmp.c_str(), path.c_str());
}

struct Args {
    std::string out = "/tmp/degraf_live";
    double warmup = 2.0;
};

Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--out") a.out = need("--out");
        else if (k == "--warmup") a.warmup = std::atof(need("--warmup").c_str());
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: live_grab_rs --out DIR [--warmup 2.0]\n";
            std::exit(0);
        }
    }
    return a;
}

rs2::pipeline_profile start_pipe(rs2::pipeline& pipe)
{
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);
    return pipe.start(cfg);
}

} // namespace

int main(int argc, char** argv) try
{
    Args args = parse_args(argc, argv);
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    if (!mkdir_p(args.out)) {
        std::cerr << "[ERR] cannot create " << args.out << "\n";
        return 1;
    }

    rs2::pipeline pipe;
    rs2::pipeline_profile profile;
    rs2::device device;
    float depth_scale = 0.001f;
    int warm_ok = 0;

    auto warmup = [&](const char* tag) -> bool {
        std::cout << "[INFO] " << tag << " starting...\n";
        profile = start_pipe(pipe);
        device = profile.get_device();
        std::cout << "[INFO] Device: " << device.get_info(RS2_CAMERA_INFO_NAME)
                  << " usb=" << device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) << "\n";
        depth_scale = device.first<rs2::depth_sensor>().get_depth_scale();
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(args.warmup);
        warm_ok = 0;
        while (std::chrono::steady_clock::now() < t_end && !g_stop) {
            try {
                pipe.wait_for_frames(2000);
                ++warm_ok;
            } catch (...) {
            }
        }
        std::cout << "[INFO] Warmup frames=" << warm_ok << "\n";
        return warm_ok > 0;
    };

    if (!warmup("attempt-1")) {
        try {
            device.hardware_reset();
        } catch (...) {
        }
        try {
            pipe.stop();
        } catch (...) {
        }
        std::this_thread::sleep_for(std::chrono::seconds(4));
        if (!warmup("attempt-2")) {
            std::cerr << "[ERR] warmup failed. Unplug camera, sudo, quit Viewer.\n";
            return 1;
        }
    }

    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    const rs2_intrinsics color_intr = color_stream.get_intrinsics();
    float baseline_m = 0.05f;
    try {
        auto left = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
        auto right = profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
        baseline_m = std::fabs(left.get_extrinsics_to(right).translation[0]);
    } catch (...) {
    }

    {
        char buf[1024];
        std::snprintf(
            buf,
            sizeof(buf),
            "{\n"
            "  \"fx\": %.6f, \"fy\": %.6f, \"cx\": %.6f, \"cy\": %.6f,\n"
            "  \"baseline_m\": %.6f, \"depth_scale\": %.8f,\n"
            "  \"width\": %d, \"height\": %d\n"
            "}\n",
            color_intr.fx,
            color_intr.fy,
            color_intr.ppx,
            color_intr.ppy,
            baseline_m,
            depth_scale,
            color_intr.width,
            color_intr.height);
        atomic_write_text(args.out + "/calib.json", buf);
    }

    std::cout << "[OK] Writing live frames to " << args.out << "  (Ctrl+C to stop)\n";
    std::cout << "[OK] Next: python3 realtime_status.py --mode live --live-dir " << args.out << "\n";

    int frame_id = 0;
    std::vector<uint16_t> depth_mm;

    while (!g_stop) {
        rs2::frameset frames = pipe.wait_for_frames(15000);
        rs2::frameset aligned = align_to_color.process(frames);
        rs2::video_frame color = aligned.get_color_frame();
        rs2::depth_frame depth = aligned.get_depth_frame();
        if (!color || !depth) continue;

        const int w = color.get_width();
        const int h = color.get_height();
        if (depth.get_width() != w || depth.get_height() != h) continue;

        // RealSense color is RGB8; stb_image_write expects RGB for JPEG.
        const auto* rgb = reinterpret_cast<const unsigned char*>(color.get_data());

        const auto* z16 = reinterpret_cast<const uint16_t*>(depth.get_data());
        depth_mm.resize(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < depth_mm.size(); ++i) {
            float mm = static_cast<float>(z16[i]) * depth_scale * 1000.f;
            depth_mm[i] = static_cast<uint16_t>(mm < 0 ? 0 : (mm > 65535.f ? 65535.f : mm + 0.5f));
        }

        // color JPG (fast) + depth raw uint16 LE
        const std::string ctmp = args.out + "/color.jpg.tmp";
        const std::string cpath = args.out + "/color.jpg";
        if (!stbi_write_jpg(ctmp.c_str(), w, h, 3, rgb, 80)) {
            std::cerr << "[WARN] color write failed\n";
            continue;
        }
        std::rename(ctmp.c_str(), cpath.c_str());

        atomic_write_bytes(args.out + "/depth_mm.u16", depth_mm.data(), depth_mm.size() * sizeof(uint16_t));

        char idbuf[64];
        std::snprintf(idbuf, sizeof(idbuf), "%d\n%d %d\n", frame_id, w, h);
        atomic_write_text(args.out + "/frame_id.txt", idbuf);

        ++frame_id;
        if (frame_id % 30 == 0) {
            std::cout << "[INFO] live frames=" << frame_id << "\n" << std::flush;
        }
    }

    pipe.stop();
    std::cout << "[OK] stopped after " << frame_id << " frames\n";
    return 0;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error: " << e.what() << "\n";
    return 1;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << "\n";
    return 1;
}
