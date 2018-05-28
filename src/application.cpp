
#include "application.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono_literals;

Application::Application(const std::string& vm_name) {
	virtual_box = virtual_box_client.virtual_box();
	session = virtual_box_client.session();
	virtual_box.find_machine(vm_name).lock_machine(session, LockType_Shared);
	machine = session.machine();
	console = session.console();
	display = console.display();

	window = sdl::Window(
		"testo",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		width, height,
		SDL_WINDOW_SHOWN
	);
	renderer = window.create_renderer();
}

void Application::run() {
#ifdef WIN32
	auto max_time = 40ms;
#else
	auto max_time = 80ms;
#endif
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

struct ARGB {
	static ARGB blue() {
		return {0xff, 0, 0, 0};
	}
	static ARGB green() {
		return {0, 0xff, 0, 0};
	}
	static ARGB red() {
		return {0, 0, 0xff, 0};
	}
	uint8_t b, g, r, a;
};

struct ImageView {
	ImageView(ARGB* p, size_t w, size_t h):
		pixels(p), width(w), height(h) {}

	ARGB& operator()(size_t x, size_t y) {
		return pixels[width * y + x];
	}

	ARGB* pixels;
	size_t width;
	size_t height;
};

void Application::update() {
	ULONG width = 0;
	ULONG height = 0;
	ULONG bits_per_pixel = 0;
	LONG x_origin = 0;
	LONG y_origin = 0;
	GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

	display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

	if (!width || !height) {
		return;
	}

	resize(width, height);

	vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
	vbox::ArrayOut array_out = safe_array.copy_out(VT_UI1);
	ImageView src((ARGB*)array_out.data, width, height);

	void* pixels = nullptr;
	int pitch = 0;
	texture.lock(nullptr, &pixels, &pitch);
	ImageView dst((ARGB*)pixels, width, height);
	for (size_t x = 0; x < width; ++x) {
		for (size_t y = 0; y < height; ++y) {
			dst(x, y) = src(x, y);
		}
	}
	texture.unlock();

	renderer.copy(texture);
	renderer.present();
}

void Application::resize(size_t w, size_t h) {
	if ((width == w) && (height == h)) {
		return;
	}
	width = w;
	height = h;
	window.set_size(width, height);
	texture = renderer.create_texture(SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
}
