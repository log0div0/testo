
#include "application.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono_literals;

Application::Application(const std::string& vm_name) {
	virtual_box = virtual_box_client.virtual_box();
	session = virtual_box_client.session();
	machine = virtual_box.find_machine(vm_name);

	window = sdl::Window(
		"testo",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		500, 500,
		SDL_WINDOW_SHOWN
	);
}

void Application::run() {
	auto max_time = 40ms;
	SDL_Event event;
	while (true) {
		auto t0 = std::chrono::high_resolution_clock::now();
		update();
		auto t1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> time = t1 - t0;
		if (time > max_time) {
			std::cout << "Slow! " << time.count() << " ms" << std::endl;
			SDL_WaitEventTimeout(&event, 0);
		} else {
			SDL_WaitEventTimeout(&event, std::chrono::duration_cast<std::chrono::duration<int, std::milli>>(max_time - time).count());
		}
		switch (event.type) {
			case SDL_QUIT:
				return;
		}
	}
}

void Application::update() {
}
