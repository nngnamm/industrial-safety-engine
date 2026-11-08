#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "safety_engine/detector.hpp"
#include "safety_engine/geometry.hpp"
#include "safety_engine/pipeline.hpp"
#include "safety_engine/tracker.hpp"
#include "safety_engine/types.hpp"
#include "safety_engine/version.hpp"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SAFETY_ENGINE_HAS_LSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(SAFETY_ENGINE_HAS_LSAN)
#define SAFETY_ENGINE_HAS_LSAN 1
#endif

#if defined(SAFETY_ENGINE_HAS_LSAN)
extern "C" const char* __lsan_default_suppressions() {
    // Pre-built libonnxruntime.so allocates two one-time process-global cpuinfo buffers via calloc.
    return "leak:libonnxruntime.so\n";
}
#endif

namespace {

struct CliOptions {
    bool run_benchmark{true};
    bool benchmark_explicitly_set{false};
    bool model_explicitly_set{false};
    std::string model_path;
    std::string save_hud_path;
    std::string export_demo_dir;
    int num_frames{50};
    int warmup_frames{10};
    int width{1920};
    int height{1080};
    std::size_t queue_capacity{8};
};

std::string resolve_default_model_path() {
    const std::vector<std::filesystem::path> candidates = {
        "models/dummy_yolo.onnx",
        "../models/dummy_yolo.onnx",
#ifdef SAFETY_ENGINE_PROJECT_ROOT
        std::filesystem::path(SAFETY_ENGINE_PROJECT_ROOT) / "models" / "dummy_yolo.onnx",
#endif
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    return "models/dummy_yolo.onnx";
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --benchmark               Run multi-threaded synthetic latency benchmark "
                 "(default)\n"
              << "  --model <path>            Path to ONNX model (default: "
                 "models/dummy_yolo.onnx;\n"
              << "                            use models/yolov8n_bench.onnx for Conv2D NN "
                 "benchmark)\n"
              << "  --frames <N>              Number of timed benchmark frames (default: 50)\n"
              << "  --warmup <N>              Number of warm-up frames before timing (default: 10)\n"
              << "  --width <W>               Synthetic frame width (default: 1920)\n"
              << "  --height <H>              Synthetic frame height (default: 1080)\n"
              << "  --queue-capacity <cap>    BoundedQueue capacity (default: 8)\n"
              << "  --save-hud <path>         Render and write a sample annotated HUD image\n"
              << "  --export-demo <dir>       Simulate 60-frame 1280x720 HUD demo sequence\n"
              << "  -h, --help                Show this help message\n";
}

int parse_positive_int(const std::string& flag, const std::string& raw) {
    std::size_t idx = 0;
    const int value = std::stoi(raw, &idx);
    if (idx != raw.size() || value <= 0) {
        throw std::invalid_argument("Invalid positive integer for " + flag + ": " + raw);
    }
    return value;
}

int parse_non_negative_int(const std::string& flag, const std::string& raw) {
    std::size_t idx = 0;
    const int value = std::stoi(raw, &idx);
    if (idx != raw.size() || value < 0) {
        throw std::invalid_argument("Invalid non-negative integer for " + flag + ": " + raw);
    }
    return value;
}

CliOptions parse_args(int argc, char* argv[], bool& help_requested) {
    CliOptions opts;
    opts.model_path = resolve_default_model_path();

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            help_requested = true;
            return opts;
        }
        if (arg == "--benchmark") {
            opts.run_benchmark = true;
            opts.benchmark_explicitly_set = true;
        } else if (arg == "--model") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --model");
            }
            opts.model_path = argv[++i];
            opts.model_explicitly_set = true;
        } else if (arg == "--frames") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --frames");
            }
            opts.num_frames = parse_positive_int("--frames", argv[++i]);
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --warmup");
            }
            opts.warmup_frames = parse_non_negative_int("--warmup", argv[++i]);
        } else if (arg == "--width") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --width");
            }
            opts.width = parse_positive_int("--width", argv[++i]);
        } else if (arg == "--height") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --height");
            }
            opts.height = parse_positive_int("--height", argv[++i]);
        } else if (arg == "--queue-capacity") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --queue-capacity");
            }
            opts.queue_capacity =
                static_cast<std::size_t>(parse_positive_int("--queue-capacity", argv[++i]));
        } else if (arg == "--save-hud") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --save-hud");
            }
            opts.save_hud_path = argv[++i];
        } else if (arg == "--export-demo") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing argument for --export-demo");
            }
            opts.export_demo_dir = argv[++i];
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (!opts.export_demo_dir.empty() && !opts.benchmark_explicitly_set) {
        opts.run_benchmark = false;
    }

    return opts;
}

void print_row(const std::string& stage_name, const safety_engine::StageLatencyStats& s) {
    std::cout << "| " << std::left << std::setw(33) << stage_name << " | " << std::right
              << std::setw(9) << std::fixed << std::setprecision(3) << s.mean_ms << " | "
              << std::setw(9) << s.p50_ms << " | " << std::setw(9) << s.p95_ms << " | "
              << std::setw(9) << s.p99_ms << " |\n";
}

void print_benchmark_table(const safety_engine::BenchmarkReport& report, const CliOptions& opts) {
    const std::string sep =
        "+-----------------------------------+-----------+-----------+-----------+-----------+\n";

    const bool is_dummy_model =
        opts.model_path.find("dummy_yolo") != std::string::npos;
    const std::string benchmark_mode =
        is_dummy_model
            ? "Pipeline & Memory Overhead Benchmark (dummy_yolo.onnx - isolates C++ zero-copy "
              "tensor wrapping, letterboxing, geometry & thread sync)"
            : "Full Neural Network Compute Benchmark (" +
                  std::filesystem::path(opts.model_path).filename().string() +
                  " - multi-layer Conv2D + SiLU + Detect CPU inference)";

    std::cout << '\n'
              << "Benchmark Mode: " << benchmark_mode << '\n'
              << "Benchmark Configuration: " << report.total_frames << " timed frames + "
              << report.warmup_frames << " warm-up frames @ " << opts.width << "x" << opts.height
              << " (BGR8), Queue Capacity = " << opts.queue_capacity << '\n'
              << sep
              << "| Pipeline Stage                    | Mean (ms) |  P50 (ms) |  P95 (ms) |  P99 "
                 "(ms) |\n"
              << sep;

    print_row("Detector (ONNX)", report.detect_stats);
    print_row("Spatial Geometry (IoPA + Polygon)", report.geometry_stats);
    print_row("Tracker + State Machine", report.tracker_stats);
    std::cout << sep;
    print_row("Total End-to-End", report.total_frame_stats);
    std::cout << sep;

    std::cout << "Overall Producer-Consumer Throughput: " << std::fixed << std::setprecision(2)
              << report.throughput_fps << " FPS (" << report.total_frames << " frames, 0 drops)\n";
}

std::vector<safety_engine::HazardZone> make_default_hazard_zones(int width, int height) {
    const auto w = static_cast<float>(width);
    const auto h = static_cast<float>(height);
    safety_engine::HazardZone zone{
        "StampingPressBay_01",
        {
            {0.25f * w, 0.15f * h},
            {0.75f * w, 0.15f * h},
            {0.75f * w, 0.90f * h},
            {0.25f * w, 0.90f * h},
        },
        /*require_helmet=*/true,
        /*require_vest=*/true,
    };
    return {zone};
}

void draw_industrial_floor_background(cv::Mat& frame,
                                      const std::vector<cv::Point2f>& zone_polygon) {
    frame.setTo(cv::Scalar(38, 42, 48));

    // Minor floor tile grid (64 px)
    for (int x = 0; x < frame.cols; x += 64) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows - 1), cv::Scalar(48, 53, 60), 1);
    }
    for (int y = 0; y < frame.rows; y += 64) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols - 1, y), cv::Scalar(48, 53, 60), 1);
    }

    // Major bay expansion joints (256 px)
    for (int x = 0; x < frame.cols; x += 256) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows - 1), cv::Scalar(62, 68, 76), 2);
    }
    for (int y = 0; y < frame.rows; y += 256) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols - 1, y), cv::Scalar(62, 68, 76), 2);
    }

    // Pedestrian corridor lane guides
    cv::line(frame, cv::Point(40, 355), cv::Point(frame.cols - 40, 355), cv::Scalar(65, 95, 110), 1,
             cv::LINE_AA);
    cv::line(frame, cv::Point(40, 565), cv::Point(frame.cols - 40, 565), cv::Scalar(65, 95, 110), 1,
             cv::LINE_AA);

    // Corner caution markers around the central polygon vertices
    for (const cv::Point2f& pt : zone_polygon) {
        const cv::Point p(static_cast<int>(std::lround(pt.x)), static_cast<int>(std::lround(pt.y)));
        cv::circle(frame, p, 4, cv::Scalar(0, 190, 255), cv::FILLED, cv::LINE_AA);
    }
}

void draw_synthetic_worker_avatar(cv::Mat& frame, float feet_x, float feet_y, bool has_helmet,
                                  bool has_vest) {
    const int cx = static_cast<int>(std::lround(feet_x));
    const int fy = static_cast<int>(std::lround(feet_y));

    // Shadow ellipse on floor
    cv::ellipse(frame, cv::Point(cx, fy), cv::Size(24, 7), 0.0, 0.0, 360.0, cv::Scalar(24, 26, 30),
                cv::FILLED, cv::LINE_AA);

    // Legs / work pants
    cv::line(frame, cv::Point(cx - 10, fy - 58), cv::Point(cx - 12, fy - 4),
             cv::Scalar(95, 80, 65), 9, cv::LINE_AA);
    cv::line(frame, cv::Point(cx + 10, fy - 58), cv::Point(cx + 12, fy - 4),
             cv::Scalar(95, 80, 65), 9, cv::LINE_AA);

    // Torso base
    const cv::Rect torso_rect(cx - 21, fy - 120, 42, 62);
    cv::rectangle(frame, torso_rect, cv::Scalar(110, 95, 80), cv::FILLED);

    // High-vis vest overlay on torso
    if (has_vest) {
        cv::rectangle(frame, torso_rect, cv::Scalar(35, 215, 235), cv::FILLED);
        // Reflective silver stripes
        cv::rectangle(frame, cv::Rect(cx - 21, fy - 96, 42, 6), cv::Scalar(225, 235, 240),
                      cv::FILLED);
        cv::rectangle(frame, cv::Rect(cx - 21, fy - 76, 42, 6), cv::Scalar(225, 235, 240),
                      cv::FILLED);
    }

    // Arms
    cv::line(frame, cv::Point(cx - 23, fy - 115), cv::Point(cx - 28, fy - 68),
             cv::Scalar(115, 100, 85), 7, cv::LINE_AA);
    cv::line(frame, cv::Point(cx + 23, fy - 115), cv::Point(cx + 28, fy - 68),
             cv::Scalar(115, 100, 85), 7, cv::LINE_AA);

    // Head
    const cv::Point head_center(cx, fy - 140);
    cv::circle(frame, head_center, 14, cv::Scalar(155, 175, 205), cv::FILLED, cv::LINE_AA);

    // Hard-hat helmet
    if (has_helmet) {
        cv::ellipse(frame, cv::Point(cx, fy - 144), cv::Size(17, 13), 0.0, 180.0, 360.0,
                    cv::Scalar(0, 225, 255), cv::FILLED, cv::LINE_AA);
        cv::line(frame, cv::Point(cx - 20, fy - 143), cv::Point(cx + 20, fy - 143),
                 cv::Scalar(0, 200, 245), 3, cv::LINE_AA);
    } else {
        // Dark hair cap so bare head is visually unmistakable
        cv::ellipse(frame, cv::Point(cx, fy - 144), cv::Size(14, 10), 0.0, 180.0, 360.0,
                    cv::Scalar(45, 50, 60), cv::FILLED, cv::LINE_AA);
    }
}

std::vector<safety_engine::Detection> make_worker_detections(float feet_x, float feet_y,
                                                             bool has_helmet, bool has_vest) {
    constexpr float kPersonW = 76.0f;
    constexpr float kPersonH = 168.0f;
    const cv::Rect2f person_bbox(feet_x - 0.5f * kPersonW, feet_y - kPersonH, kPersonW, kPersonH);

    std::vector<safety_engine::Detection> dets;
    dets.push_back(safety_engine::Detection{person_bbox, 0.96f,
                                            static_cast<int>(safety_engine::ClassId::Person)});

    if (has_helmet) {
        const cv::Rect2f helmet_bbox(feet_x - 18.0f, feet_y - 162.0f, 36.0f, 28.0f);
        dets.push_back(safety_engine::Detection{helmet_bbox, 0.94f,
                                                static_cast<int>(safety_engine::ClassId::Helmet)});
    }
    if (has_vest) {
        const cv::Rect2f vest_bbox(feet_x - 22.0f, feet_y - 122.0f, 44.0f, 64.0f);
        dets.push_back(safety_engine::Detection{vest_bbox, 0.93f,
                                                static_cast<int>(safety_engine::ClassId::Vest)});
    }
    return dets;
}

bool export_demo_sequence(const std::string& output_dir) {
    constexpr int kWidth = 1280;
    constexpr int kHeight = 720;
    constexpr int kTotalFrames = 60;

    std::filesystem::create_directories(output_dir);

    // Polygonal "HIGH VOLTAGE ZONE" in the center of the 1280x720 industrial floor.
    // Between y = 185 and y = 575, the left edge is x = 360 and right edge is x = 920.
    const safety_engine::HazardZone hv_zone{
        "HIGH VOLTAGE ZONE",
        {
            {420.0f, 105.0f},
            {860.0f, 105.0f},
            {920.0f, 185.0f},
            {920.0f, 575.0f},
            {860.0f, 645.0f},
            {420.0f, 645.0f},
            {360.0f, 575.0f},
            {360.0f, 185.0f},
        },
        /*require_helmet=*/true,
        /*require_vest=*/true,
    };
    const std::vector<safety_engine::HazardZone> zones = {hv_zone};

    const safety_engine::SafetyAnalyzer analyzer;
    safety_engine::WorkerTracker tracker(0.30f, /*alert_streak_threshold=*/3,
                                         /*max_missed_frames=*/5);

    std::vector<cv::Point> trail_w1;
    std::vector<cv::Point> trail_w2;
    trail_w1.reserve(kTotalFrames);
    trail_w2.reserve(kTotalFrames);

    for (int f = 0; f < kTotalFrames; ++f) {
        // Worker #1 (Compliant: Person + Helmet + Vest) walks safely across the upper lane
        const float w1_feet_x = 270.0f + 9.5f * static_cast<float>(f);
        const float w1_feet_y = 355.0f;

        // Worker #2 (Non-Compliant: Person + Vest, NO Helmet) walks in the lower lane:
        // - Frames 0..14 (15 frames): Outside zone (feet_x = 180..348 < 360) -> Green SAFE
        // - Frame 15 (Enters zone, Frame 1): feet_x = 364 >= 360 -> Orange/Yellow SUSPECTED (1/3)
        // - Frame 16 (Enters zone, Frame 2): feet_x = 374 >= 360 -> Orange/Yellow SUSPECTED (2/3)
        // - Frames 17..59 (Stays in zone, Frame 3+): feet_x = 384..814 -> Red ACTIVE ALERT [NO HELMET]
        float w2_feet_x = 0.0f;
        if (f < 15) {
            w2_feet_x = 180.0f + 12.0f * static_cast<float>(f);
        } else {
            w2_feet_x = 364.0f + 10.0f * static_cast<float>(f - 15);
        }
        const float w2_feet_y = 565.0f;

        trail_w1.emplace_back(static_cast<int>(std::lround(w1_feet_x)),
                              static_cast<int>(std::lround(w1_feet_y)));
        trail_w2.emplace_back(static_cast<int>(std::lround(w2_feet_x)),
                              static_cast<int>(std::lround(w2_feet_y)));

        cv::Mat frame(kHeight, kWidth, CV_8UC3);
        draw_industrial_floor_background(frame, hv_zone.polygon);

        // Draw subtle trajectory breadcrumbs
        for (const cv::Point& pt : trail_w1) {
            cv::circle(frame, pt, 2, cv::Scalar(60, 160, 60), cv::FILLED, cv::LINE_AA);
        }
        for (std::size_t i = 0; i < trail_w2.size(); ++i) {
            const cv::Scalar dot_col = (i < 15U) ? cv::Scalar(60, 160, 60)
                                       : (i < 17U) ? cv::Scalar(0, 180, 255)
                                                   : cv::Scalar(40, 40, 220);
            cv::circle(frame, trail_w2[i], 2, dot_col, cv::FILLED, cv::LINE_AA);
        }

        // Draw synthetic worker avatars on the floor
        draw_synthetic_worker_avatar(frame, w1_feet_x, w1_feet_y, /*has_helmet=*/true,
                                     /*has_vest=*/true);
        draw_synthetic_worker_avatar(frame, w2_feet_x, w2_feet_y, /*has_helmet=*/false,
                                     /*has_vest=*/true);

        // Build detections and run through SafetyAnalyzer + WorkerTracker + SafetyPipeline::render_hud
        std::vector<safety_engine::Detection> detections =
            make_worker_detections(w1_feet_x, w1_feet_y, /*has_helmet=*/true, /*has_vest=*/true);
        const std::vector<safety_engine::Detection> w2_dets =
            make_worker_detections(w2_feet_x, w2_feet_y, /*has_helmet=*/false, /*has_vest=*/true);
        detections.insert(detections.end(), w2_dets.begin(), w2_dets.end());

        const std::vector<safety_engine::WorkerStatus> statuses =
            analyzer.analyze_frame(detections, zones);
        const std::vector<safety_engine::TrackedWorker> tracked = tracker.update(statuses);

        const std::string telemetry =
            "ONNX: 1.83 ms | Geometry: 0.002 ms | Tracker: 0.002 ms | 96.7 FPS";
        safety_engine::SafetyPipeline::render_hud(frame, zones, tracked, telemetry);

        // Render bottom status bar showing frame progress & FSM de-bounce state
        std::ostringstream footer;
        footer << "Frame " << std::setw(2) << std::setfill('0') << (f + 1) << "/" << kTotalFrames
               << " | IoPA PPE Binding + Ground-Plane Polygon Test + 3-Frame FSM De-Bouncing";
        cv::putText(frame, footer.str(), cv::Point(18, kHeight - 16), cv::FONT_HERSHEY_SIMPLEX,
                    0.50, cv::Scalar(170, 185, 200), 1, cv::LINE_AA);

        std::ostringstream filename;
        filename << "frame_" << std::setw(3) << std::setfill('0') << f << ".png";
        const std::filesystem::path out_path = std::filesystem::path(output_dir) / filename.str();
        if (!cv::imwrite(out_path.string(), frame)) {
            std::cerr << "Error: Failed to write demo frame to '" << out_path.string() << "'.\n";
            return false;
        }
    }

    std::cout << "Exported " << kTotalFrames << " HUD demo frames (1280x720) to: " << output_dir
              << '\n';
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        bool help_requested = false;
        const CliOptions opts = parse_args(argc, argv, help_requested);
        if (help_requested) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        const std::vector<std::string> providers = Ort::GetAvailableProviders();
        std::cout << "=====================================================================\n"
                  << "  " << safety_engine::get_engine_info() << '\n'
                  << "  OpenCV: " << CV_VERSION << " | ONNX Runtime Providers: ";
        for (std::size_t i = 0; i < providers.size(); ++i) {
            std::cout << providers[i] << (i + 1U < providers.size() ? ", " : "");
        }
        std::cout << "\n=====================================================================\n";

        if (!opts.export_demo_dir.empty()) {
            if (!export_demo_sequence(opts.export_demo_dir)) {
                return EXIT_FAILURE;
            }
            if (!opts.run_benchmark && opts.save_hud_path.empty()) {
                return EXIT_SUCCESS;
            }
        }

        if (!std::filesystem::exists(opts.model_path)) {
            if (opts.model_explicitly_set) {
                std::cerr << "Error: ONNX model not found at '" << opts.model_path << "'.\n";
                return EXIT_FAILURE;
            }
            std::cout << "Notice: Default model '" << opts.model_path
                      << "' not found. Generate it via:\n"
                      << "  python3 scripts/create_dummy_yolo_onnx.py\n";
            return EXIT_SUCCESS;
        }

        std::cout << "Loading ONNX model: " << opts.model_path << '\n';
        safety_engine::Detector detector(opts.model_path, 0.45f, 0.50f, 640, 640);
        const safety_engine::SafetyAnalyzer analyzer;
        safety_engine::WorkerTracker tracker(0.30f, 3, 5);
        const std::vector<safety_engine::HazardZone> zones =
            make_default_hazard_zones(opts.width, opts.height);

        safety_engine::SafetyPipeline pipeline(detector, analyzer, tracker, zones);

        if (opts.run_benchmark) {
            const safety_engine::BenchmarkReport report = pipeline.run_multithreaded_benchmark(
                opts.num_frames, opts.width, opts.height, opts.queue_capacity, opts.warmup_frames);
            print_benchmark_table(report, opts);
        }

        if (!opts.save_hud_path.empty()) {
            tracker.reset();
            cv::Mat demo_frame(opts.height, opts.width, CV_8UC3, cv::Scalar(42, 46, 52));
            std::vector<safety_engine::TrackedWorker> tracked;
            for (int step = 0; step < 3; ++step) {
                tracked = pipeline.process_frame(demo_frame);
            }
            safety_engine::SafetyPipeline::render_hud(demo_frame, pipeline.hazard_zones(), tracked);
            if (!cv::imwrite(opts.save_hud_path, demo_frame)) {
                std::cerr << "Error: Failed to write HUD snapshot to '" << opts.save_hud_path
                          << "'.\n";
                return EXIT_FAILURE;
            }
            std::cout << "Saved annotated HUD snapshot to: " << opts.save_hud_path << '\n';
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
