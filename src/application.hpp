
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"

struct Application {
	Application();
	void run();

	void step_0();
	void step_1();
	void step_2();
	void set_up();
	void gui();
	void tear_down();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session session;
};
