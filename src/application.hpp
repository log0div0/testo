
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vbox/machine.hpp"
#include "sdl/window.hpp"

struct Application {
	Application(const std::string& vm_name);
	void run();

	void update();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Machine machine;
	vbox::Session session;
	sdl::Window window;
};
