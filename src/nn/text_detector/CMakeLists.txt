cmake_minimum_required(VERSION 3.0)

set(SRCS TextDetector.cpp)
embed_ftlite_model(${CMAKE_CURRENT_SOURCE_DIR}/TextDetector.onnx)

add_library(text-detector STATIC ${SRCS})
target_link_libraries(text-detector stb onnxruntime pthread)

add_executable(text_detector_demo demo.cpp)
target_link_libraries(text_detector_demo text-detector)

file(GLOB_RECURSE TEST_SRCS tests/*.cpp)
add_executable(text_detector_tests tests.cpp ${TEST_SRCS})
target_link_libraries(text_detector_tests text-detector)

add_custom_target(run-text-detector-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/text_detector_tests --durations yes
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
	DEPENDS text_detector_tests
)
