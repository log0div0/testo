
#pragma once

#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vm.hpp"

struct Application {
	Application();
	void run();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;

	VM vm;
};
