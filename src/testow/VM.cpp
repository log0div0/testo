
#include "VM.hpp"
#include "App.hpp"
#include <iostream>
#include <math.h>

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

VM::VM(vbox::Machine machine_): machine(std::move(machine_)) {
	session = ::app->virtual_box_client.session();
	machine.lock_machine(session, LockType_Shared);
	running = true;
	thread = std::thread([=] {
		run();
	});
}

VM::~VM() {
	running = false;
	thread.join();
	session.unlock_machine();
}

static int entry_index(layer l, int location, int entry)
{
    return entry*l.w*l.h + location;
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

void VM::run() {
	std::vector<uint8_t> texture2;
	darknet::Image image;

	auto interval = 200ms;
	auto previous = std::chrono::high_resolution_clock::now();
	while (running) {
		auto current = std::chrono::high_resolution_clock::now();
		auto diff = current - previous;
		if (diff < interval) {
			std::this_thread::sleep_for(interval - diff);
		}
		previous = current;

		if (machine.state() == MachineState_Running) {
			if (!display) {
				display = session.console().display();
			}
		} else {
			if (display) {
				display = {};
				std::lock_guard<std::shared_mutex> lock(mutex);
				width = 0;
				height = 0;
			}
		}

		if (!display) {
			continue;
		}

		ULONG width = 0;
		ULONG height = 0;
		ULONG bits_per_pixel = 0;
		LONG x_origin = 0;
		LONG y_origin = 0;
		GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

		display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

		if (!width || !height) {
			continue;
		}

		vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
		vbox::ArrayOut texture1 = safe_array.copy_out(VT_UI1);
		if (this->texture1 == texture1) {
			continue;
		}

		texture2.resize(width * height * sizeof(BGRA));
		std::fill(texture2.begin(), texture2.end(), 0);

		if ((image.width() != width) || (image.height() != height)) {
			image = darknet::Image(width, height, 3);
		}

		size_t channels = 4;

		for(size_t c = 0; c < image.channels(); ++c){
			for(size_t h = 0; h < image.height(); ++h){
				for(size_t w = 0; w < image.width(); ++w){
					size_t src_index = c + channels*w + channels*width*h;
					size_t dst_index = w + image.width()*h + image.width()*image.height()*c;
					image[dst_index] = ((float)texture1[src_index])/255.;
				}
			}
		}

		throw std::runtime_error("TODO: predict");

		std::lock_guard<std::shared_mutex> lock(mutex);
		std::swap(this->texture1, texture1);
		std::swap(this->texture2, texture2);
		this->width = width;
		this->height = height;
	}
}
