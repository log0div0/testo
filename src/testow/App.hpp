
#pragma once

#include "Hypervisor.hpp"
#include "VM.hpp"
#include "Texture.hpp"
#include <map>

struct App {
	App(std::shared_ptr<Host> host);
	void render();

	std::vector<std::shared_ptr<Guest>> guests;
	char query[128] = {};

	std::unique_ptr<VM> vm;
	Texture texture;
};

extern App* app;
extern std::ostream& operator<<(std::ostream& stream, const std::exception& error);
