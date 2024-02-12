set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(lexocraft_neural_network neural_network.cpp neural_network_diff.cpp)

target_include_directories(lexocraft_neural_network PUBLIC ${CMAKE_PROJECT_SOURCE_DIR}/thirdparty/eigen})
