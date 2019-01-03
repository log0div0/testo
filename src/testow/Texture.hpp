
#pragma once

#include <d3d11.h>
#include <stdint.h>

struct Texture
{
	Texture() = default;
	Texture(size_t width, size_t height);
	~Texture();

	Texture(const Texture&) = delete;
	Texture& operator=(const Texture&) = delete;

	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	ID3D11ShaderResourceView* handle() const {
		return _view;
	}

	size_t width() const {
		return _width;
	}
	size_t height() const {
		return _height;
	}

	void write(uint8_t* data, size_t size);
	void clear();

private:
	ID3D11Texture2D* _texture = nullptr;
	ID3D11ShaderResourceView* _view = nullptr;
	size_t _width, _height;
};
