
#pragma once

#ifdef WIN32
#include "windows/Texture.hpp"
#elif __APPLE__
#error "Implement me"
#elif __linux__
#include "linux/Texture.hpp"
#else
#error "Unknown OS"
#endif
