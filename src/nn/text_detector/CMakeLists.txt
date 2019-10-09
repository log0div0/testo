cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
# list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
embed_ftlite_model(${CMAKE_CURRENT_SOURCE_DIR}/TextDetector.tflite)

add_library(text-detector STATIC ${SRCS})
target_link_libraries(text-detector stb tensorflow-lite)
# add_executable(text_detector_demo main.cpp)
# target_link_libraries(text_detector_demo text-detector)
