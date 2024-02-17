set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/neural_network")

add_library(efuzz
    STATIC
        efuzz.cpp
)

target_link_libraries(efuzz
    PUBLIC
        efuzz_neural_network
)

target_include_directories(efuzz
    PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${CMAKE_PROJECT_SOURCE_DIR}/thirdparty/eigen"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rapidfuzz-cpp"
)
