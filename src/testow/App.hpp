
#pragma once

#include "Hypervisor.hpp"
#include "VM.hpp"
#include "Texture.hpp"
#include <map>

struct App {
	App(int argc, char** argv);
	void render();

	std::shared_ptr<Hypervisor> hypervisor;
	std::vector<std::shared_ptr<Guest>> guests;
	char query[128] = {};

	std::unique_ptr<VM> vm;
	Texture texture;
	int foreground = -1;
	int background = -1;
};

extern App* app;
extern std::ostream& operator<<(std::ostream& stream, const std::exception& error);
