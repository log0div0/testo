
#include "Texture.hpp"
#include <stdexcept>

extern ID3D11Device* g_pd3dDevice;
extern ID3D11DeviceContext* g_pd3dDeviceContext;

Texture::Texture(size_t width, size_t height):
	_width(width), _height(height)
{
	D3D11_TEXTURE2D_DESC desc = {};
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	desc.MiscFlags = 0;

	auto result = g_pd3dDevice->CreateTexture2D(&desc, NULL, &_texture);
	if (FAILED(result)) {
		throw std::runtime_error("CreateTexture2D failed");
	}

	result = g_pd3dDevice->CreateShaderResourceView(_texture, NULL, &_view);
	if (FAILED(result)) {
		throw std::runtime_error("CreateShaderResourceView failed");
	}
}

Texture::~Texture() {
	if (_view) {
		_view->Release();
	}
	if (_texture) {
		_texture->Release();
	}
}

Texture::Texture(Texture&& other):
	_width(other._width),
	_height(other._height),
	_texture(other._texture),
	_view(other._view)
{
	other._width = 0;
	other._height = 0;
	other._texture = nullptr;
	other._view = nullptr;
}

Texture& Texture::operator=(Texture&& other) {
	std::swap(_width, other._width);
	std::swap(_height, other._height);
	std::swap(_texture, other._texture);
	std::swap(_view, other._view);
	return *this;
}

void Texture::write(const uint8_t* data, size_t size) {
	D3D11_MAPPED_SUBRESOURCE mappedResource = {};
	auto result = g_pd3dDeviceContext->Map(_texture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result)) {
		throw std::runtime_error("Texture write failed");
	}
	for(size_t h = 0; h < _height; ++h){
		for(size_t w = 0; w < _width; ++w){
			for(size_t c = 0; c < 3; ++c){
				size_t src_index = h*_width*3 + w*3 + c;
				size_t dst_index = h*_width*4 + w*4 + c;
				((uint8_t*)mappedResource.pData)[dst_index] = data[src_index];
			}
			size_t dst_index = h*_width*4 + w*4 + 3;
			((uint8_t*)mappedResource.pData)[dst_index] = 255;
		}
	}
	// memcpy(mappedResource.pData, data, size);
	g_pd3dDeviceContext->Unmap(_texture, 0);
}
