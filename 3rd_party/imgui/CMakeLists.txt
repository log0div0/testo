cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SRCS
	imgui.cpp
	imgui_draw.cpp
	imgui_widgets.cpp
)
if(WIN32)
	set(IMPL_SRCS
		imgui_impl_dx11.cpp
		imgui_impl_win32.cpp
	)
endif()
add_library(imgui STATIC ${SRCS} ${IMPL_SRCS})
