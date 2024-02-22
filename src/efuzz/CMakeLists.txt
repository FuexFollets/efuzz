set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/neural_network")

add_library(efuzz
    STATIC
        efuzz.cpp
)

target_precompile_headers(efuzz
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/efuzz.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/encode.hpp"
)

target_link_libraries(efuzz
    PUBLIC
        efuzz_neural_network
)

target_include_directories(efuzz
    PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/thirdparty/eigen"
        "${PROJECT_SOURCE_DIR}/thirdparty/rapidfuzz-cpp"
)

target_compile_features(efuzz
    PUBLIC
        cxx_std_20
)
