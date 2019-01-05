
#pragma once

#include <vbox/machine.hpp>
#include <vbox/session.hpp>
#include <vbox/display.hpp>
#include <thread>
#include <shared_mutex>
#include "Texture.hpp"

struct VM {
	VM(vbox::Machine machine);
	~VM();

	vbox::Machine machine;
	vbox::Session session;
	vbox::Display display;

	Texture texture;
	std::shared_mutex mutex;

private:
	std::thread thread;
	void run();
	bool running = false;
};
