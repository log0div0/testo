
#pragma once

#include <SDL2/SDL.h>

namespace sdl {

struct API {
	API(uint32_t flags);
	~API();

	API(const API&) = delete;
	API& operator=(const API&) = delete;
};

}
