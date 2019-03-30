
#pragma once

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>
#include <darknet/Network.hpp>
#include "VM.hpp"
#include "Texture.hpp"

struct App {
	App();
	void render();

	vbox::API api;
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;

	darknet::Network net;
	std::shared_ptr<VM> vm;
	Texture texture1;
	Texture texture2;
	size_t width = 0;
	size_t height = 0;
};

extern App* app;
