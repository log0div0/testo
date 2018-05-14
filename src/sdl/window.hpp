
#pragma once

#include <SDL2/SDL.h>

namespace sdl {

struct Window {
	Window(const char* title, int x, int y, int w, int h, uint32_t flags);
	~Window();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	SDL_Window* handle = nullptr;
};

}
