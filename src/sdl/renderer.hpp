
#pragma once

#include <SDL2/SDL.h>

namespace sdl {

struct Renderer {
	Renderer(SDL_Renderer* handle);
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(Renderer&& other);
	Renderer& operator=(Renderer&& other);

	SDL_Renderer* handle = nullptr;
};

}
