
#pragma once

#include <cstddef>
#include <stdint.h>
#include <GL/gl3w.h>

struct Texture
{
	Texture() = default;
	Texture(size_t width, size_t height);
	~Texture();

	Texture(const Texture&) = delete;
	Texture& operator=(const Texture&) = delete;

	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	void* handle() const {
		return (void*)(intptr_t)_handle;
	}

	void write(const uint8_t* data, size_t size);

private:
	GLuint _handle = 0;
	size_t _width, _height;
};
