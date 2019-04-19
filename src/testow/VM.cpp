
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

VM::VM(vir::Connect& qemu_connect, vir::Domain domain): qemu_connect(qemu_connect), domain(std::move(domain)) {
	running = true;
	buffer.reserve(10000000); //10 Mb
	thread = std::thread([=] {
		run();
	});
}

VM::~VM() {
	running = false;
	thread.join();
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
		auto stream = qemu_connect.new_stream();
		auto mime = domain.screenshot(stream);
		size_t bytes = stream.recv_all(buffer.data(), buffer.capacity());
		stream.finish();

		std::string format(""), str_width(""), str_height("");
		size_t current_pos = 0;
		while(buffer[current_pos] != '\n') {
			format += buffer[current_pos];
			current_pos++;
		}
		current_pos++;

		while(buffer[current_pos] != ' ') {
			str_width += buffer[current_pos];
			current_pos++;
		}

		current_pos++;

		while(buffer[current_pos] != '\n') {
			str_height += buffer[current_pos];
			current_pos++;
		}

		current_pos++;

		while(buffer[current_pos] != '\n') {
			current_pos++;
		}

		current_pos++;

		width = std::stoi(str_width);
		height = std::stoi(str_height);

		if (!width || !height) {
			continue;
		}

		std::vector<uint8_t> texture1(buffer.begin() + current_pos, buffer.begin() + bytes);

		texture2.resize(width * height * 3);
		std::fill(texture2.begin(), texture2.end(), 0);

		if (this->texture1 == texture1) {
			continue;
		}


		if ((image.width() != width) || (image.height() != height)) {
			image = darknet::Image(width, height, 3);
		}

		size_t channels = 3;

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
