
#include "renderer.hpp"
#include <stdexcept>

namespace sdl {

Renderer::Renderer(SDL_Renderer* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Renderer::~Renderer() {
	if (handle) {
		SDL_DestroyRenderer(handle);
	}
}

Renderer::Renderer(Renderer&& other): handle(other.handle) {
	other.handle = nullptr;
}

Renderer& Renderer::operator=(Renderer&& other) {
	std::swap(handle, other.handle);
	return *this;
}

}
