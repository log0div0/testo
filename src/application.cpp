
#include "application.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono_literals;

Application::Application() {
	virtual_box = virtual_box_client.virtual_box();
}

void Application::run() {
	SDL_Event event;
	while (true) {
		while (SDL_WaitEventTimeout(&event, 1000 / 60)) {
			switch (event.type) {
				case SDL_QUIT:
					return;
			}
		}
	}
}
