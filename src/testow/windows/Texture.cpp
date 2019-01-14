
#include "Texture.hpp"
#include <stdexcept>

extern ID3D11Device* g_pd3dDevice;
extern ID3D11DeviceContext* g_pd3dDeviceContext;

Texture::Texture(size_t width, size_t height) {
	D3D11_TEXTURE2D_DESC desc = {};
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
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
	_texture(other._texture),
	_view(other._view)
{
	other._texture = nullptr;
	other._view = nullptr;
}

Texture& Texture::operator=(Texture&& other) {
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
	memcpy(mappedResource.pData, data, size);
	g_pd3dDeviceContext->Unmap(_texture, 0);
}
