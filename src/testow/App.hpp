
#pragma once

#include <qemu/Host.hpp>
#include <darknet/Network.hpp>
#include "VM.hpp"
#include "Texture.hpp"
#include <map>

struct App {
	App();
	void render();

	vir::Connect qemu_connect;
	std::map<std::string, vir::Domain> domains;

	// darknet::Network net;
	std::shared_ptr<VM> vm;
	Texture texture1;
	Texture texture2;
	size_t width = 0;
	size_t height = 0;
};

extern App* app;
