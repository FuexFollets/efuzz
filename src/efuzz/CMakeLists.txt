set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${CMAKE_PROJECT_SOURCE_DIR}/src
    ${CMAKE_PROJECT_SOURCE_DIR}/thirdparty/eigen}
)

add_library(efuzz STATIC
    efuzz.cpp
)

target_include_directories(efuzz
    PRIVATE
        ${CMAKE_PROJECT_SOURCE_DIR}/src
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/neural_network")
