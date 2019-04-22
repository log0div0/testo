
#pragma once

struct Texture
{
	Texture() = default;
	Texture(size_t width, size_t height);
	~Texture();

	Texture(const Texture&) = delete;
	Texture& operator=(const Texture&) = delete;

	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	void* handle() const;

	void write(const uint8_t* data, size_t size);
};
