
#include "texture.hpp"
#include <stdexcept>

namespace sdl {

Texture::Texture(SDL_Texture* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Texture::~Texture() {
	if (handle) {
		SDL_DestroyTexture(handle);
	}
}

Texture::Texture(Texture&& other): handle(other.handle) {
	other.handle = nullptr;
}

Texture& Texture::operator=(Texture&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void Texture::update(const void* pixels, int pitch, const SDL_Rect* rect) {
	try {
		int error_code = SDL_UpdateTexture(handle, rect, pixels, pitch);
		if (error_code) {
			throw std::runtime_error(SDL_GetError());
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
