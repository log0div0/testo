
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

		float* predictions = ::app->net.predict(image);

		// const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";

		const auto& l = ::app->net.back();

		int netw = ::app->net.width();
		int neth = ::app->net.height();
		int w = image.width();
		int h = image.height();
		int new_w=0;
		int new_h=0;
		if (((float)netw/w) < ((float)neth/h)) {
		    new_w = netw;
		    new_h = (h * netw)/w;
		} else {
		    new_h = neth;
		    new_w = (w * neth)/h;
		}

		for (int y = 0; y < l.h; ++y) {
			for (int x = 0; x < l.w; ++x) {
				int i = y * l.w + x;
				int box_index  = entry_index(l, i, 0);
				int obj_index  = entry_index(l, i, 4);
				float objectness = predictions[obj_index];
				if (objectness < 0.25) {
					continue;
				}

				// int class_index = -1;
				// float max_class_prob = 0;
				// for (int j = 0; j < l.classes; ++j){
				// 	int class_index = entry_index(l, i, 4 + 1 + j);
				// 	float class_prob = objectness*predictions[class_index];
				// 	if (class_prob > max_class_prob){
				// 		class_index = j;
				// 		max_class_prob = class_prob;
				// 	}
				// }
				box b = get_yolo_box(predictions, l.biases, l.mask[0], box_index, x, y, l.w, l.h, netw, neth, l.w*l.h);

			    b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
			    b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
			    b.w *= (float)netw/new_w;
			    b.h *= (float)neth/new_h;

				int left  = (b.x-b.w/2.)*image.width();
				int right = (b.x+b.w/2.)*image.width();
				int top   = (b.y-b.h/2.)*image.height();
				int bot   = (b.y+b.h/2.)*image.height();

				if (left < 0) {
					left = 0;
				}
				if (right > (image.width()-1)) {
					right = (image.width()-1);
				}
				if (top < 0) {
					top = 0;
				}
				if (bot > (image.height()-1)) {
					bot = (image.height()-1);
				}

				BGRA color = BGRA::red();
				BGRA* p = (BGRA*)texture2.data();

				for (size_t x = left; x <= right; ++x) {
					p[width * top + x] = color;
					p[width * bot + x] = color;
				}
				for (size_t y = top; y <= bot; ++y) {
					p[width * y + left] = color;
					p[width * y + right] = color;
				}
			}
		}

		std::lock_guard<std::shared_mutex> lock(mutex);
		std::swap(this->texture1, texture1);
		std::swap(this->texture2, texture2);
		this->width = width;
		this->height = height;
	}
}
