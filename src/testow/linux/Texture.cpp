
#include "Texture.hpp"
#include <stdexcept>
#include <iostream>

Texture::Texture(size_t width, size_t height): _width(width), _height(height) {
	glGenTextures(1, &_handle);
	if (glGetError() != GL_NO_ERROR) {
		throw std::runtime_error("glGenTextures failed");
	}
	glBindTexture(GL_TEXTURE_2D, _handle);
	if (glGetError() != GL_NO_ERROR) {
		throw std::runtime_error("glBindTexture failed");
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

Texture::~Texture() {
	if (_handle) {
		glDeleteTextures(1, &_handle);
	}
}

Texture::Texture(Texture&& other):
	_handle(other._handle),
	_width(other._width),
	_height(other._height)
{
	other._handle = 0;
	other._width = 0;
	other._height = 0;
}

Texture& Texture::operator=(Texture&& other) {
	std::swap(_handle, other._handle);
	std::swap(_width, other._width);
	std::swap(_height, other._height);
	return *this;
}

void Texture::write(const uint8_t* data, size_t size) {
	if (!size) {
		return;
	}
	glBindTexture(GL_TEXTURE_2D, _handle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	if (glGetError() != GL_NO_ERROR) {
		throw std::runtime_error("OpenGL error");
	}
}

