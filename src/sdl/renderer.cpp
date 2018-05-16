
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

Texture Renderer::create_texture(uint32_t format, int access, int w, int h) {
	try {
		SDL_Texture* result = SDL_CreateTexture(handle, format, access, w, h);
		if (!result) {
			throw std::runtime_error(SDL_GetError());
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Renderer::copy(const Texture& texture, const SDL_Rect* src, const SDL_Rect* dst) {
	try {
		int error_code = SDL_RenderCopy(handle, texture.handle, src, dst);
		if (error_code) {
			throw std::runtime_error(SDL_GetError());
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Renderer::present() {
	SDL_RenderPresent(handle);
}

}
