cmake_minimum_required(VERSION 3.0)

if(NOT TENSORFLOW_DIR)
	message(FATAL_ERROR "TENSORFLOW_DIR is not set")
endif()

include_directories(
	${TENSORFLOW_DIR}
	${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

if(APPLE)
link_directories(
	${TENSORFLOW_DIR}/tensorflow/lite/tools/make/gen/osx_x86_64/lib/
)
else()
link_directories(
	${TENSORFLOW_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib/
)
endif()

list(APPEND SRC_DIRS "*.cpp")

list(APPEND SRC_DIRS "backends/*.cpp")
list(APPEND SRC_DIRS "backends/dummy/*.cpp")
list(APPEND SRC_DIRS "backends/vbox/*.cpp")
if (WIN32)
list(APPEND SRC_DIRS "backends/hyperv/*.cpp")
endif()
if (UNIX AND NOT APPLE)
list(APPEND SRC_DIRS "backends/qemu/*.cpp")
endif()

list(APPEND SRC_DIRS "nn/*.cpp")
list(APPEND SRC_DIRS "nn/**/*.cpp")

file(GLOB SRCS ${SRC_DIRS})
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

macro(embed_ftlite_model MODEL_PATH)
	get_filename_component(MODEL_NAME ${MODEL_PATH} NAME)
	get_filename_component(MODEL_DIRECTORY ${MODEL_PATH} DIRECTORY)
	add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp
		WORKING_DIRECTORY ${MODEL_DIRECTORY}
		COMMAND xxd -i ${MODEL_NAME} ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp
		DEPENDS ${MODEL_PATH}
	)
	list(APPEND SRCS ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp)
endmacro(embed_ftlite_model)

embed_ftlite_model(${CMAKE_CURRENT_SOURCE_DIR}/nn/text_detector/TextDetector.tflite)

add_library(testo-core STATIC ${SRCS})
target_link_libraries(testo-core coro vbox pugixml tensorflow-lite)

if(WIN32)
	target_link_libraries(testo-core hyperv msft virtdisk++)
endif()

if (UNIX AND NOT APPLE)
	target_link_libraries(testo-core qemu)
endif()

if(UNIX)
	target_link_libraries(testo-core stdc++fs pthread)
endif()

add_executable(testo main.cpp)
add_dependencies(testo testo-core)
target_link_libraries(testo testo-core)
