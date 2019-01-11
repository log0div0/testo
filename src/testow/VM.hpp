
#pragma once

#include <vbox/machine.hpp>
#include <vbox/session.hpp>
#include <vbox/display.hpp>
#include <thread>
#include <shared_mutex>

struct Screen {
	Screen(vbox::Display display): _display(std::move(display)) {

	}

	void update();

	size_t width() const {
		return _width;
	}
	size_t height() const {
		return _height;
	}
	const uint8_t* data() const {
		return _pixels.data;
	}
	size_t data_size() const {
		return _pixels.data_size;
	}

private:
	vbox::Display _display;
	vbox::ArrayOut _pixels;
	size_t _width = 0;
	size_t _height = 0;
};

struct VM {
	VM(vbox::Machine machine);
	~VM();

	vbox::Machine machine;
	vbox::Session session;

	std::unique_ptr<Screen> screen;
	std::shared_mutex mutex;

private:
	std::thread thread;
	void run();
	bool running = false;
};
