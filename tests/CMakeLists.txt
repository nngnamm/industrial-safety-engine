if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    # Prevent overriding the parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(safety_tests
    test_smoke.cpp
    test_detector.cpp
    test_geometry.cpp
    test_tracker_and_queue.cpp
)

target_compile_definitions(safety_tests
    PRIVATE
        SAFETY_ENGINE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(safety_tests
    PRIVATE
        safety_engine_core
        GTest::gtest_main
)

target_compile_options(safety_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

include(GoogleTest)
gtest_discover_tests(safety_tests)
