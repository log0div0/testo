
#pragma once

#ifdef WIN32
#include "windows/Texture.hpp"
#elif __APPLE__
#include "osx/Texture.hpp"
#elif __linux__
#include "linux/Texture.hpp"
#else
#error "Unknown OS"
#endif
