
#pragma once

#include <SDL2/SDL.h>

namespace sdl {

struct Texture {
	Texture(SDL_Texture* handle);
	~Texture();

	Texture(const Texture&) = delete;
	Texture& operator=(const Texture&) = delete;
	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	SDL_Texture* handle = nullptr;
};

}
