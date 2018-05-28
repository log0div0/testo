
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

void Texture::update(const SDL_Rect* rect, const void* pixels, int pitch) {
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

void Texture::lock(const SDL_Rect* rect, void** pixels, int* pitch) {
	try {
		int error_code = SDL_LockTexture(handle, rect, pixels, pitch);
		if (error_code) {
			throw std::runtime_error(SDL_GetError());
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Texture::unlock() {
	SDL_UnlockTexture(handle);
}

}
