set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(efuzz_neural_network
    STATIC
        neural_network.cpp neural_network_diff.cpp
)

target_include_directories(efuzz_neural_network
    PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/thirdparty/eigen"
)
