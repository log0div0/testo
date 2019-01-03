
#pragma once

#include <vbox/machine.hpp>
#include <vbox/session.hpp>
#include <vbox/display.hpp>
#include <thread>

struct VM {
	VM(vbox::Machine machine);
	~VM();

	const vbox::Machine& machine() const {
		return _machine;
	}

private:
	vbox::Machine _machine;
	vbox::Session session;
	vbox::Display display;

private:
	std::thread thread;
	void run();
	bool running = false;
};
