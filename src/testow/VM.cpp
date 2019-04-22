
#include "VM.hpp"
#include "App.hpp"
#include <iostream>
#include <math.h>
#include <stb_image.h>
#include <darknet/YOLO.hpp>

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

VM::VM(vir::Connect& qemu_connect, vir::Domain& domain): qemu_connect(qemu_connect), domain(domain) {
	domain_name = domain.name();
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
	std::vector<uint8_t> buffer(10'000'000);

	auto interval = 200ms;
	auto previous = std::chrono::high_resolution_clock::now();
	while (running) {
		auto current = std::chrono::high_resolution_clock::now();
		auto diff = current - previous;
		if (diff < interval) {
			std::this_thread::sleep_for(interval - diff);
		}
		previous = current;
		if (!domain.is_active()) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			view = {};
			continue;
		}
		auto stream = qemu_connect.new_stream();
		auto mime = domain.screenshot(stream);
		size_t bytes = stream.recv_all(buffer.data(), buffer.size());
		stream.finish();

		stb::Image screenshot(buffer.data(), bytes);

		if (network) {
			if ((network->w != screenshot.width) ||
				(network->h != screenshot.height) ||
				(network->c != screenshot.channels)) {
				network.reset();
			}
		}
		if (!network) {
			use_gpu = true;
			network = std::make_unique<darknet::Network>("/home/log0div0/work/testo/nn/testo.network",
				1, screenshot.width, screenshot.height, screenshot.channels);
			network->load_weights("/home/log0div0/work/testo/nn/testo.weights");
		}

		std::string query_copy;
		{
			std::lock_guard<std::shared_mutex> lock(mutex);
			query_copy = query;
		}

		yolo::predict(*network, screenshot, query_copy);

		std::lock_guard<std::shared_mutex> lock(mutex);
		std::swap(view, screenshot);
	}
}
