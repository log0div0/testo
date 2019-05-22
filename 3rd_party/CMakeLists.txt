cmake_minimum_required(VERSION 3.0)

add_subdirectory(coro)
add_subdirectory(VBoxCAPI)
add_subdirectory(imgui)
add_subdirectory(pugixml)

if(UNIX AND NOT APPLE)
add_subdirectory(gl3w)
endif()

add_subdirectory(CLBlast)