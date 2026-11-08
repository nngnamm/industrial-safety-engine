cmake_minimum_required(VERSION 3.16)

project(IndustrialSafetyEngine
    VERSION 0.1.0
    DESCRIPTION "Real-Time Industrial Worker Safety & Hazard Zone Analytics Engine"
    LANGUAGES CXX
)

# Strictly enforce C++17 without compiler-specific extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd / IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# AddressSanitizer & UndefinedBehaviorSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "AddressSanitizer & UndefinedBehaviorSanitizer enabled (-fsanitize=address,undefined)")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    else()
        message(WARNING "ENABLE_ASAN is only supported with GCC or Clang compilers.")
    endif()
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

# 1. Threads (C++17 concurrency primitives)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# 2. OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

# 3. ONNX Runtime (supports -DONNXRUNTIME_ROOT_DIR=... or ENV ONNXRUNTIME_ROOT_DIR)
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to ONNX Runtime installation root directory")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT_DIR}
        ENV ONNXRUNTIME_ROOT_DIR
    PATH_SUFFIXES
        include
        include/onnxruntime
        include/onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT_DIR}
        ENV ONNXRUNTIME_ROOT_DIR
    PATH_SUFFIXES
        lib
        lib64
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Please specify its root location via "
        "-DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime "
        "(ONNXRUNTIME_INCLUDE_DIR='${ONNXRUNTIME_INCLUDE_DIR}', "
        "ONNXRUNTIME_LIB='${ONNXRUNTIME_LIB}')."
    )
endif()

message(STATUS "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")

get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib")
list(APPEND CMAKE_BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

# ------------------------------------------------------------------------------
# Core Library Target
# ------------------------------------------------------------------------------
add_library(safety_engine_core STATIC
    src/version.cpp
    src/detector.cpp
    src/geometry.cpp
    src/tracker.cpp
    src/pipeline.cpp
)

target_include_directories(safety_engine_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(safety_engine_core
    PUBLIC
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        Threads::Threads
)

target_compile_options(safety_engine_core
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# ------------------------------------------------------------------------------
# Main Application Target
# ------------------------------------------------------------------------------
add_executable(safety_app
    src/main.cpp
)

target_compile_definitions(safety_app
    PRIVATE
        SAFETY_ENGINE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(safety_app
    PRIVATE
        safety_engine_core
)

target_compile_options(safety_app
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# ------------------------------------------------------------------------------
# Unit & Smoke Tests
# ------------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
