
#include "window.hpp"
#include <stdexcept>

namespace sdl {

Window::Window(const char* title, int x, int y, int w, int h, uint32_t flags) {
	handle = SDL_CreateWindow(title, x, y, w, h, flags);
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Window::~Window() {
	if (handle) {
		SDL_DestroyWindow(handle);
	}
}

}
