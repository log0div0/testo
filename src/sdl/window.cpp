
#include "window.hpp"
#include <stdexcept>

namespace sdl {

Window::Window(const char* title, int x, int y, int w, int h, uint32_t flags) {
	try {
		handle = SDL_CreateWindow(title, x, y, w, h, flags);
		if (!handle) {
			throw std::runtime_error(SDL_GetError());
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Window::~Window() {
	if (handle) {
		SDL_DestroyWindow(handle);
	}
}

Renderer Window::create_renderer(int index, uint32_t flags) {
	try {
		SDL_Renderer* result = SDL_CreateRenderer(handle, index, flags);
		if (!result) {
			throw std::runtime_error(SDL_GetError());
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
