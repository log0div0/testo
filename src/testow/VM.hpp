
#pragma once

#include <qemu/Domain.hpp>
#include <qemu/Host.hpp>
#include <thread>
#include <shared_mutex>
#include <darknet/Image.hpp>
#include <darknet/Network.hpp>

struct VM {
	VM(vir::Connect& qemu_connect, vir::Domain& domain);
	~VM();

	vir::Connect& qemu_connect;
	std::string domain_name;
	vir::Domain& domain;

	std::shared_mutex mutex;
	stb::Image view;
	std::string query;

private:
	std::unique_ptr<darknet::Network> network;
	std::thread thread;
	void run();
	bool running = false;
};
