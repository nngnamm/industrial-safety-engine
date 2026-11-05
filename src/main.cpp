#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
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
    bool model_explicitly_set{false};
    std::string model_path;
    std::string save_hud_path;
    int num_frames{50};
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
                 "models/dummy_yolo.onnx)\n"
              << "  --frames <N>              Number of benchmark frames (default: 50)\n"
              << "  --width <W>               Synthetic frame width (default: 1920)\n"
              << "  --height <H>              Synthetic frame height (default: 1080)\n"
              << "  --queue-capacity <cap>    BoundedQueue capacity (default: 8)\n"
              << "  --save-hud <path>         Render and write a sample annotated HUD image\n"
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
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
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

    std::cout << '\n'
              << "Benchmark Configuration: " << report.total_frames << " frames @ " << opts.width
              << "x" << opts.height << " (BGR8), Queue Capacity = " << opts.queue_capacity << '\n'
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
                opts.num_frames, opts.width, opts.height, opts.queue_capacity);
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
