
#pragma once

#include <SDL2/SDL.h>

namespace sdl {

struct Texture {
	Texture() = default;
	Texture(SDL_Texture* handle);
	~Texture();

	Texture(const Texture&) = delete;
	Texture& operator=(const Texture&) = delete;
	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	void update(const void* pixels, int pitch, const SDL_Rect* rect = nullptr);

	SDL_Texture* handle = nullptr;
};

}
