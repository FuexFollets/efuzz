set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TEST_FILES
    encode
    train_encoder
)

if(COMPILE_TESTS)
    function(add_test_executable name)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)
        add_executable(${name} ${PROJECT_SOURCE_DIR}/tests/${name}.cpp)
        target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${name} efuzz)
        target_compile_options(${name} PUBLIC "-g")
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    enable_testing()

    foreach(test_file ${TEST_FILES})
        add_test_executable(${test_file})
    endforeach()
endif()
