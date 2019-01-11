
#pragma once

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>
#include "VM.hpp"
#include "Texture.hpp"

struct App {
	App();
	void render();

	vbox::API api;
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;

	std::shared_ptr<VM> vm;
	Texture texture;
};

extern App* app;
