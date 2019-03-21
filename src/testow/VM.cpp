
#include "VM.hpp"
#include "App.hpp"
#include <iostream>

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

		float* predictions = ::app->net.predict(image);

		const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";
		const std::string pattern = "Select a language";

		const auto& l = ::app->net.back();
		for (int y = 0; y < l.h; ++y) {
			std::cout << y << ": ";
			for (int x = 0; x < l.w; ++x) {
				int i = y * l.w + x;
				int index = -1;
				float max_prob = 0;
				for (int j = 0; j < l.classes; ++j){
					char ch = chars[j];
					if (pattern.find(ch) != std::string::npos) {
						int obj_index  = entry_index(l, i, 4);
						float objectness = predictions[obj_index];
						int class_index = entry_index(l, i, 4 + 1 + j);
						float prob = objectness*predictions[class_index];
						if (prob > max_prob){
							index = j;
							max_prob = prob;
						}
					}
				}
				if ((max_prob > 0.01) && (index >=0)) {
					std::cout << chars[index];
					// box b = detection.bbox;

					// int left  = (b.x-b.w/2.)*image.width();
					// int right = (b.x+b.w/2.)*image.width();
					// int top   = (b.y-b.h/2.)*image.height();
					// int bot   = (b.y+b.h/2.)*image.height();

					// if (left < 0) {
					// 	left = 0;
					// }
					// if (right > (image.width()-1)) {
					// 	right = (image.width()-1);
					// }
					// if (top < 0) {
					// 	top = 0;
					// }
					// if (bot > (image.height()-1)) {
					// 	bot = (image.height()-1);
					// }

					// BGRA color = BGRA::red();
					// BGRA* p = (BGRA*)texture2.data();

					// for (size_t x = left; x <= right; ++x) {
					// 	p[width * top + x] = color;
					// 	p[width * bot + x] = color;
					// }
					// for (size_t y = top; y <= bot; ++y) {
					// 	p[width * y + left] = color;
					// 	p[width * y + right] = color;
					// }
				}
			}
			std::cout << std::endl;
		}

		std::lock_guard<std::shared_mutex> lock(mutex);
		std::swap(this->texture1, texture1);
		std::swap(this->texture2, texture2);
		this->width = width;
		this->height = height;
	}
}
