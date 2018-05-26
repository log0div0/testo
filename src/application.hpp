
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "sdl/window.hpp"

struct Application {
	Application();
	void run();

	void step_0();
	void step_1();
	void step_2();
	void set_up();
	void event_loop();
	void tear_down();

	void update_window(int width, int height, void* data);

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session session;

	sdl::Window window;
	sdl::Renderer renderer;
	sdl::Texture texture;
};
