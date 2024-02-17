cmake_minimum_required(VERSION 3.15...3.25 FATAL_ERROR)

project(Efuzz)

option(COMPILE_TESTS "Compile tests" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/annoy)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/efuzz)

set(public_headers
    efuzz/efuzz.hpp
    efuzz/encode.hpp
    efuzz/neural_network/neural_network.hpp
)

foreach (HEADER ${public_headers})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER}" "${CMAKE_CURRENT_BINARY_DIR}/include/${HEADER}" COPYONLY)
endforeach()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
