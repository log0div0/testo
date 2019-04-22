
#pragma once

#include <qemu/Domain.hpp>
#include <qemu/Host.hpp>
#include <thread>
#include <shared_mutex>
#include <testo/StinkingPileOfShit.hpp>

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
	StinkingPileOfShit shit;
	std::thread thread;
	void run();
	bool running = false;
};
