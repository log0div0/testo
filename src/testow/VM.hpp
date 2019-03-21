
#pragma once

#include <qemu/Domain.hpp>
#include <qemu/Host.hpp>
#include <thread>
#include <shared_mutex>
#include <darknet/Image.hpp>

struct VM {
	VM(vir::Connect& qemu_connect, vir::Domain domain);
	~VM();

	vir::Domain domain;

	std::shared_mutex mutex;
	std::vector<uint8_t> texture1;
	std::vector<uint8_t> texture2;
	size_t width = 0;
	size_t height = 0;
	std::vector<uint8_t> buffer;

private:
	vir::Connect& qemu_connect;
	std::thread thread;
	void run();
	bool running = false;
};
