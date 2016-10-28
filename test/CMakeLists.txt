cmake_minimum_required(VERSION 3.5)

project(${TEST_PROJECT_NAME})

set(LIBRARY_TESTS_SOURCE DTWTests.cpp vector_a.h vector_b.h)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if (USE_GPU)
    cuda_add_executable(${TEST_PROJECT_NAME} ${LIBRARY_TESTS_SOURCE})
else (USE_GPU)
    add_executable(${TEST_PROJECT_NAME} ${LIBRARY_TESTS_SOURCE})
endif (USE_GPU)

target_link_libraries(${TEST_PROJECT_NAME} gmock)
target_link_libraries(${TEST_PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${TEST_PROJECT_NAME} pthread)
target_link_libraries(${TEST_PROJECT_NAME} ${LIBRARY_NAME}) # NOTE: This is defined from project above

add_test(${TEST_PROJECT_NAME} ${TEST_PROJECT_NAME})

add_custom_command(TARGET ${TEST_PROJECT_NAME}
        COMMENT "Run Tests"
        POST_BUILD
        COMMAND ctest -V -C $<CONFIGURATION> --output-on-failure)