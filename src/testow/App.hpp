
#pragma once

#include <qemu/Host.hpp>
#include "VM.hpp"
#include "Texture.hpp"
#include <map>

struct App {
	App();
	void render();

	vir::Connect qemu_connect;
	std::map<std::string, vir::Domain> domains;
	char query[128] = {};

	std::unique_ptr<VM> vm;
	Texture texture;
};

extern App* app;
