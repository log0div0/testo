
#pragma once

#include "renderer.hpp"

namespace sdl {

struct Window {
	Window() = default;
	Window(const char* title, int x, int y, int w, int h, uint32_t flags);
	~Window();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;
	Window(Window&& other);
	Window& operator=(Window&& other);

	operator bool() const;

	Renderer create_renderer(int index = -1, uint32_t flags = 0);

	void set_size(int w, int h);

	SDL_Window* handle = nullptr;
};

}
