cmake_minimum_required(VERSION 3.15...3.25 FATAL_ERROR)

project(Efuzz)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/annoy/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cereal/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/efuzz)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/annoy)

set(public_headers
    efuzz/efuzz.hpp
)

foreach (HEADER ${public_headers})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER}" "${CMAKE_CURRENT_BINARY_DIR}/include/${HEADER}" COPYONLY)
endforeach()
