cmake_minimum_required(VERSION 3.0)

add_definitions(-std=gnu++14 -DSDL_MAIN_HANDLED)

file(GLOB_RECURSE SRCS *.cpp)

include_directories(${CMAKE_BINARY_DIR}/out/include)
link_directories(${CMAKE_BINARY_DIR}/out/lib)

add_executable(testo ${SRCS})
add_dependencies(testo sdl2)

target_link_libraries(testo vboxcapi)
if (WIN32)
	target_link_libraries(testo
		debug SDL2d
		optimized SDL2
		Version
		Imm32
		Winmm
	)
else()
	target_link_libraries(testo dl pthread SDL2)
	if (APPLE)
		target_link_libraries(testo
			"-framework AudioToolbox"
			"-framework CoreAudio"
			"-framework CoreVideo"
			"-framework ForceFeedback"
			"-framework IOKit"
			"-framework Cocoa"
			"-framework Carbon"
			iconv
		)
	endif()
endif()
