
#pragma once

#include <vbox/machine.hpp>
#include <vbox/session.hpp>
#include <vbox/display.hpp>
#include <thread>
#include <shared_mutex>
#include <darknet/Image.hpp>

struct VM {
	VM(vbox::Machine machine);
	~VM();

	vbox::Machine machine;
	vbox::Session session;
	vbox::Display display;

	std::shared_mutex mutex;
	vbox::ArrayOut texture1;
	std::vector<uint8_t> texture2;
	size_t width = 0;
	size_t height = 0;

private:
	std::thread thread;
	void run();
	bool running = false;
};
