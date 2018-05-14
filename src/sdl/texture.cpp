
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

}
