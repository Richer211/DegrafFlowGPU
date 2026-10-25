// Continuous RealSense color+depth recorder for Mac (no pyrealsense2).
// Output matches record_realsense_seq.py (v0):
//   <out>/<name>/{color,depth}/%06d.png  timestamps.csv  meta.json  calib.json
//
// Build:
//   c++ -O2 -std=c++14 record_png_seq.cpp -o record_png_seq \
//     -I"$HOME/librealsense/include" \
//     -I"$HOME/librealsense/third-party" \
//     -L"$HOME/librealsense/build/Release" \
//     -lrealsense2 -Wl,-rpath,"$HOME/librealsense/build/Release" -lz
//
// Run:
//   sudo ./record_png_seq --out ~/realsense_seq --name seq_box_slide_002 --seconds 15

#include <librealsense2/rs.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <zlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

std::atomic<bool> g_stop{false};

void on_sigint(int) { g_stop = true; }

bool mkdir_p(const std::string& path)
{
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void be32(unsigned char* p, uint32_t v)
{
    p[0] = static_cast<unsigned char>((v >> 24) & 0xff);
    p[1] = static_cast<unsigned char>((v >> 16) & 0xff);
    p[2] = static_cast<unsigned char>((v >> 8) & 0xff);
    p[3] = static_cast<unsigned char>(v & 0xff);
}

void write_be32(std::ofstream& f, uint32_t v)
{
    unsigned char b[4];
    be32(b, v);
    f.write(reinterpret_cast<char*>(b), 4);
}

void write_chunk(std::ofstream& f, const char* type, const unsigned char* data, size_t n)
{
    write_be32(f, static_cast<uint32_t>(n));
    f.write(type, 4);
    if (n) f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    uint32_t crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(type), 4);
    if (n) crc = crc32(crc, data, static_cast<uInt>(n));
    write_be32(f, crc);
}

bool write_png_u16(const std::string& path, int w, int h, const uint16_t* src)
{
    const size_t row_bytes = static_cast<size_t>(w) * 2 + 1;
    std::vector<unsigned char> raw(row_bytes * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        unsigned char* row = raw.data() + static_cast<size_t>(y) * row_bytes;
        row[0] = 0;
        for (int x = 0; x < w; ++x) {
            const uint16_t v = src[y * w + x];
            row[1 + 2 * x] = static_cast<unsigned char>((v >> 8) & 0xff);
            row[2 + 2 * x] = static_cast<unsigned char>(v & 0xff);
        }
    }

    uLongf bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> compressed(bound);
    if (compress2(compressed.data(), &bound, raw.data(),
                  static_cast<uLong>(raw.size()), Z_BEST_SPEED) != Z_OK) {
        return false;
    }
    compressed.resize(bound);

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write(reinterpret_cast<const char*>(sig), 8);

    unsigned char ihdr[13];
    be32(ihdr + 0, static_cast<uint32_t>(w));
    be32(ihdr + 4, static_cast<uint32_t>(h));
    ihdr[8] = 16;
    ihdr[9] = 0;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    write_chunk(f, "IHDR", ihdr, 13);
    write_chunk(f, "IDAT", compressed.data(), compressed.size());
    write_chunk(f, "IEND", nullptr, 0);
    return static_cast<bool>(f);
}

struct Args {
    std::string out = ".";
    std::string name = "seq";
    double seconds = 15.0;
    double warmup = 1.5;
    int width = 0;   // 0 = auto (let librealsense pick a working profile)
    int height = 0;
    int fps = 30;
};

void usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0
              << " --out DIR --name SEQ [--seconds 15] [--warmup 1.5]"
              << " [--width W --height H] [--fps 30]\n"
              << "  Omit --width/--height to auto-pick Depth+Color profiles.\n";
}

Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--out") a.out = need("--out");
        else if (k == "--name") a.name = need("--name");
        else if (k == "--seconds") a.seconds = std::atof(need("--seconds").c_str());
        else if (k == "--warmup") a.warmup = std::atof(need("--warmup").c_str());
        else if (k == "--width") a.width = std::atoi(need("--width").c_str());
        else if (k == "--height") a.height = std::atoi(need("--height").c_str());
        else if (k == "--fps") a.fps = std::atoi(need("--fps").c_str());
        else if (k == "-h" || k == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << k << "\n";
            usage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

rs2::pipeline_profile start_with_fallbacks(rs2::pipeline& pipe, const Args& args)
{
    struct Attempt {
        const char* label;
        int w, h, fps;
        bool sized;
    };
    std::vector<Attempt> attempts;
    if (args.width > 0 && args.height > 0) {
        attempts.push_back({"requested", args.width, args.height, args.fps, true});
    }
    attempts.push_back({"auto-default", 0, 0, 0, false});
    attempts.push_back({"848x480@30", 848, 480, 30, true});
    attempts.push_back({"640x480@30", 640, 480, 30, true});
    attempts.push_back({"640x480@15", 640, 480, 15, true});
    attempts.push_back({"424x240@30", 424, 240, 30, true});

    std::string last_err;
    for (const auto& a : attempts) {
        rs2::config cfg;
        if (a.sized) {
            cfg.enable_stream(RS2_STREAM_DEPTH, a.w, a.h, RS2_FORMAT_Z16, a.fps);
            cfg.enable_stream(RS2_STREAM_COLOR, a.w, a.h, RS2_FORMAT_RGB8, a.fps);
            std::cout << "[INFO] Trying " << a.label << " depth+color\n";
        } else {
            // Let librealsense pick any mutually compatible Depth+Color profiles.
            cfg.enable_stream(RS2_STREAM_DEPTH);
            cfg.enable_stream(RS2_STREAM_COLOR);
            std::cout << "[INFO] Trying " << a.label << " (device picks resolution)\n";
        }
        try {
            return pipe.start(cfg);
        } catch (const rs2::error& e) {
            last_err = e.what();
            std::cerr << "[WARN] " << a.label << " failed: " << e.what() << "\n";
            try {
                pipe.stop();
            } catch (...) {
            }
        }
    }
    throw std::runtime_error("No Depth+Color profile worked. Last error: " + last_err
                             + "\nCheck USB3 (Name=D435I, Usb Type=3.x), unplug 10s, retry.");
}

std::string json_escape(const std::string& s)
{
    std::ostringstream o;
    for (char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else o << c;
    }
    return o.str();
}

} // namespace

int main(int argc, char** argv) try
{
    const Args args = parse_args(argc, argv);
    signal(SIGINT, on_sigint);

    const std::string seq = args.out + "/" + args.name;
    const std::string color_dir = seq + "/color";
    const std::string depth_dir = seq + "/depth";
    if (!mkdir_p(args.out) || !mkdir_p(seq) || !mkdir_p(color_dir) || !mkdir_p(depth_dir)) {
        std::cerr << "Failed to create output dirs under " << seq << "\n";
        return 1;
    }

    rs2::pipeline pipe;
    rs2::pipeline_profile profile;
    rs2::device device;
    rs2::video_stream_profile depth_vp;
    rs2::video_stream_profile color_vp;
    float depth_scale = 0.001f;
    int warm_ok = 0;

    auto do_start_and_warmup = [&](const char* tag) -> bool {
        std::cout << "[INFO] " << tag << ": starting pipeline...\n";
        profile = start_with_fallbacks(pipe, args);
        device = profile.get_device();
        const std::string usb = device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
        std::cout << "[INFO] Device: " << device.get_info(RS2_CAMERA_INFO_NAME)
                  << " usb=" << usb << "\n";
        if (!usb.empty() && usb[0] == '2') {
            std::cerr << "[WARN] USB2 detected. Prefer a USB3 port/cable; bandwidth will drop frames.\n";
        }
        depth_vp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        color_vp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        std::cout << "[INFO] Active depth " << depth_vp.width() << "x" << depth_vp.height()
                  << "@" << depth_vp.fps() << " color " << color_vp.width() << "x" << color_vp.height()
                  << "@" << color_vp.fps() << "\n";
        rs2::depth_sensor depth_sensor = device.first<rs2::depth_sensor>();
        depth_scale = depth_sensor.get_depth_scale();

        // Give Mac USB stack a moment after exclusive open.
        std::this_thread::sleep_for(std::chrono::milliseconds(800));

        const auto t_warm_end = std::chrono::steady_clock::now()
            + std::chrono::duration<double>(args.warmup);
        warm_ok = 0;
        while (std::chrono::steady_clock::now() < t_warm_end && !g_stop) {
            rs2::frameset fs;
            bool got = false;
            try {
                got = pipe.try_wait_for_frames(&fs, 2000);
            } catch (...) {
                got = false;
            }
            if (!got) {
                try {
                    fs = pipe.wait_for_frames(2000);
                    got = true;
                } catch (...) {
                    got = false;
                }
            }
            if (got) {
                ++warm_ok;
            }
        }
        std::cout << "[INFO] Warmup frames=" << warm_ok << "\n";
        return warm_ok > 0;
    };

    if (!do_start_and_warmup("attempt-1")) {
        std::cerr << "[WARN] Warmup empty — hardware_reset + retry...\n";
        try {
            device.hardware_reset();
        } catch (...) {
        }
        try {
            pipe.stop();
        } catch (...) {
        }
        std::this_thread::sleep_for(std::chrono::seconds(4));
        if (!do_start_and_warmup("attempt-2")) {
            std::cerr << "[ERR] No frames during warmup.\n"
                      << "  1) Quit rs-capture/Viewer; unplug camera 15s; replug USB3\n"
                      << "  2) Test: sudo ~/librealsense/build/Release/rs-capture (need live video)\n"
                      << "  3) Then retry record_png_seq\n";
            try {
                pipe.stop();
            } catch (...) {
            }
            return 1;
        }
    }

    rs2::align align_to_color(RS2_STREAM_COLOR);

    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    const rs2_intrinsics depth_intr = depth_stream.get_intrinsics();
    const rs2_intrinsics color_intr = color_stream.get_intrinsics();
    const rs2_extrinsics extr = depth_stream.get_extrinsics_to(color_stream);

    float baseline_m = 0.05f;
    try {
        auto left = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
        auto right = profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
        auto ir_ext = left.get_extrinsics_to(right);
        baseline_m = std::fabs(ir_ext.translation[0]);
    } catch (...) {
        // IR not enabled; keep D435-ish default.
    }

    {
        std::ofstream meta(seq + "/meta.json");
        meta << "{\n"
             << "  \"device_name\": \"" << json_escape(device.get_info(RS2_CAMERA_INFO_NAME)) << "\",\n"
             << "  \"serial\": \"" << json_escape(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) << "\",\n"
             << "  \"firmware\": \"" << json_escape(device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION)) << "\",\n"
             << "  \"usb_type\": \"" << json_escape(device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)) << "\",\n"
             << "  \"color_size\": [" << color_intr.width << ", " << color_intr.height << "],\n"
             << "  \"depth_size\": [" << depth_intr.width << ", " << depth_intr.height << "],\n"
             << "  \"fps\": " << color_vp.fps() << ",\n"
             << "  \"align\": \"depth_to_color\",\n"
             << "  \"depth_unit\": \"millimeter_uint16_png\",\n"
             << "  \"note\": \"color RGB8 PNG; depth uint16 mm aligned to color\"\n"
             << "}\n";
    }
    {
        std::ofstream calib(seq + "/calib.json");
        calib << "{\n"
              << "  \"color\": {\n"
              << "    \"width\": " << color_intr.width << ", \"height\": " << color_intr.height << ",\n"
              << "    \"fx\": " << color_intr.fx << ", \"fy\": " << color_intr.fy << ",\n"
              << "    \"cx\": " << color_intr.ppx << ", \"cy\": " << color_intr.ppy << "\n"
              << "  },\n"
              << "  \"depth_before_align\": {\n"
              << "    \"width\": " << depth_intr.width << ", \"height\": " << depth_intr.height << ",\n"
              << "    \"fx\": " << depth_intr.fx << ", \"fy\": " << depth_intr.fy << ",\n"
              << "    \"cx\": " << depth_intr.ppx << ", \"cy\": " << depth_intr.ppy << "\n"
              << "  },\n"
              << "  \"depth_scale_m_per_unit\": " << depth_scale << ",\n"
              << "  \"stereo_baseline_m\": " << baseline_m << ",\n"
              << "  \"depth_to_color_extrinsics\": {\n"
              << "    \"translation_m\": [" << extr.translation[0] << ", "
              << extr.translation[1] << ", " << extr.translation[2] << "]\n"
              << "  },\n"
              << "  \"aligned\": {\n"
              << "    \"fx\": " << color_intr.fx << ", \"fy\": " << color_intr.fy << ",\n"
              << "    \"cx\": " << color_intr.ppx << ", \"cy\": " << color_intr.ppy << ",\n"
              << "    \"width\": " << color_intr.width << ", \"height\": " << color_intr.height << ",\n"
              << "    \"baseline_m\": " << baseline_m << ",\n"
              << "    \"depth_scale_m_per_unit\": " << depth_scale << "\n"
              << "  }\n"
              << "}\n";
    }

    std::ofstream ts(seq + "/timestamps.csv");
    ts << "frame_id,color_t_ns,depth_t_ns\n";

    std::cout << "[INFO] Recording to " << seq << " for " << args.seconds << "s (Ctrl+C to stop)\n";
    std::cout << "[INFO] " << device.get_info(RS2_CAMERA_INFO_NAME)
              << " usb=" << device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) << "\n";

    const auto t_end = std::chrono::steady_clock::now()
        + std::chrono::duration<double>(args.seconds);
    int frame_id = 0;
    std::vector<uint16_t> depth_mm;
    std::vector<unsigned char> color_rgb;

    while (!g_stop && std::chrono::steady_clock::now() < t_end) {
        rs2::frameset frames = pipe.wait_for_frames(15000);
        rs2::frameset aligned = align_to_color.process(frames);
        rs2::depth_frame depth = aligned.get_depth_frame();
        rs2::video_frame color = aligned.get_color_frame();
        if (!depth || !color) continue;

        const int w = color.get_width();
        const int h = color.get_height();
        const int dw = depth.get_width();
        const int dh = depth.get_height();
        if (w != dw || h != dh) {
            std::cerr << "[WARN] size mismatch color=" << w << "x" << h
                      << " depth=" << dw << "x" << dh << "\n";
            continue;
        }

        color_rgb.resize(static_cast<size_t>(w) * h * 3);
        std::memcpy(color_rgb.data(), color.get_data(), color_rgb.size());

        const auto* z16 = reinterpret_cast<const uint16_t*>(depth.get_data());
        depth_mm.resize(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < depth_mm.size(); ++i) {
            const float mm = static_cast<float>(z16[i]) * depth_scale * 1000.f;
            depth_mm[i] = static_cast<uint16_t>(mm < 0.f ? 0.f : (mm > 65535.f ? 65535.f : mm + 0.5f));
        }

        char stem[32];
        std::snprintf(stem, sizeof(stem), "%06d", frame_id);
        const std::string cpath = color_dir + "/" + stem + ".png";
        const std::string dpath = depth_dir + "/" + stem + ".png";

        if (!stbi_write_png(cpath.c_str(), w, h, 3, color_rgb.data(), w * 3)) {
            std::cerr << "[ERR] write color failed: " << cpath << "\n";
            break;
        }
        if (!write_png_u16(dpath, w, h, depth_mm.data())) {
            std::cerr << "[ERR] write depth failed: " << dpath << "\n";
            break;
        }

        // rs2 timestamps are milliseconds (double); store integer nanoseconds.
        const long long c_ns = static_cast<long long>(color.get_timestamp() * 1e6);
        const long long d_ns = static_cast<long long>(depth.get_timestamp() * 1e6);
        ts << frame_id << "," << c_ns << "," << d_ns << "\n";

        ++frame_id;
        if (frame_id % args.fps == 0) {
            std::cout << "[INFO] frames=" << frame_id << "\n" << std::flush;
        }
    }

    pipe.stop();
    ts.flush();
    std::cout << "[OK] Saved " << frame_id << " frames to " << seq << "\n";
    if (frame_id < 2) {
        std::cerr << "[ERR] Too few frames; check USB3 / sudo / camera not used by another app\n";
        return 1;
    }
    std::cout << "[OK] Check: ls " << color_dir << " | wc -l ; ls " << depth_dir << " | wc -l\n";
    return 0;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function()
              << "(" << e.get_failed_args() << "):\n    " << e.what() << "\n";
    return 1;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << "\n";
    return 1;
}
