
#include "VM.hpp"
#include "App.hpp"
#include <iostream>
#include <math.h>
#include <stb_image.h>

using namespace std::chrono_literals;

struct BGRA {
	static BGRA blue() {
		return {0xff, 0, 0, 0xff};
	}
	static BGRA green() {
		return {0, 0xff, 0, 0xff};
	}
	static BGRA red() {
		return {0, 0, 0xff, 0xff};
	}
	uint8_t b, g, r, a;
};

VM::VM(std::shared_ptr<Guest> guest_): guest(guest_) {
	running = true;
	thread = std::thread([=] {
		run();
	});
}

VM::~VM() {
	running = false;
	thread.join();
}

void VM::run() {
	auto interval = 200ms;
	auto previous = std::chrono::high_resolution_clock::now();
	while (running) {
		auto current = std::chrono::high_resolution_clock::now();
		auto diff = current - previous;
		if (diff < interval) {
			std::this_thread::sleep_for(interval - diff);
		}
		previous = current;

		stb::Image screenshot = guest->screenshot();
		if (!screenshot.data()) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			view = {};
			continue;
		}

		std::string query_copy;
		{
			std::lock_guard<std::shared_mutex> lock(mutex);
			query_copy = query;
		}

		shit.stink_even_stronger(screenshot, query_copy);

		std::lock_guard<std::shared_mutex> lock(mutex);
		std::swap(view, screenshot);
	}
}
