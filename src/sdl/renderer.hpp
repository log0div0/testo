
#pragma once

#include "texture.hpp"

namespace sdl {

struct Renderer {
	Renderer() = default;
	Renderer(SDL_Renderer* handle);
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(Renderer&& other);
	Renderer& operator=(Renderer&& other);

	Texture create_texture(uint32_t format, int access, int w, int h);
	void copy(const Texture& texture, const SDL_Rect* src = nullptr, const SDL_Rect* dst = nullptr);
	void present();

	SDL_Renderer* handle = nullptr;
};

}
