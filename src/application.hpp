
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "sdl/window.hpp"

struct ARGB {
	static ARGB blue() {
		return {0xff, 0, 0, 0};
	}
	static ARGB green() {
		return {0, 0xff, 0, 0};
	}
	static ARGB red() {
		return {0, 0, 0xff, 0};
	}
	uint8_t b, g, r, a;
};

struct Application {
	Application();
	void run();

	void step_0();
	void step_1();
	void step_2();
	void set_up();
	void event_loop();
	void tear_down();

	void update_window(int width, int height, ARGB* data);

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session session;

	sdl::Window window;
	sdl::Renderer renderer;
	sdl::Texture texture;
};
