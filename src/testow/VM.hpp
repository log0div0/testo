
#pragma once

#include <qemu/Domain.hpp>
#include <qemu/Host.hpp>
#include <thread>
#include <shared_mutex>
#include <darknet/Image.hpp>

struct VM {
	VM(vir::Connect& qemu_connect, vir::Domain& domain);
	~VM();

	vir::Connect& qemu_connect;
	std::string domain_name;
	vir::Domain& domain;

	std::shared_mutex mutex;
	stb::Image texture1;
	stb::Image texture2;
	size_t width = 0;
	size_t height = 0;

private:
	std::thread thread;
	void run();
	bool running = false;
};
