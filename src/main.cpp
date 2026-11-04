#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "safety_engine/version.hpp"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SafetyEngineSmokeTest");

    cv::Mat frame = cv::Mat::zeros(640, 640, CV_8UC3);
    const std::size_t frame_bytes = frame.total() * frame.elemSize();

    const std::vector<std::string> providers = Ort::GetAvailableProviders();

    std::cout << "Engine: " << safety_engine::get_engine_info() << '\n';
    std::cout << "OpenCV Version: " << CV_VERSION << '\n';
    std::cout << "ONNX Runtime Execution Providers (" << providers.size() << "): ";
    for (std::size_t i = 0; i < providers.size(); ++i) {
        std::cout << providers[i] << (i + 1 < providers.size() ? ", " : "");
    }
    std::cout << '\n';
    std::cout << "Dummy Frame Buffer Size (" << frame.cols << "x" << frame.rows << "x"
              << frame.channels() << "): " << frame_bytes << " bytes\n";

    return 0;
}
