
#pragma once

#include "VM.hpp"
#include "Texture.hpp"
#include <mutex>
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct App {
	App();
	void render();

	vbox::API api;
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;

	Texture texture;
	std::mutex mutex;
	std::shared_ptr<VM> vm;
};

extern App* app;