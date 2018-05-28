
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vbox/machine.hpp"
#include "sdl/window.hpp"

struct Application {
	Application(const std::string& vm_name);
	void run();

	void update();
	void resize(size_t w, size_t h);

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session session;
	vbox::Machine machine;
	vbox::Console console;
	vbox::Display display;
	size_t width = 0;
	size_t height = 0;
	sdl::Window window;
	sdl::Renderer renderer;
	sdl::Texture texture;
};
