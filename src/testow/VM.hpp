
#pragma once

#include "Hypervisor.hpp"
#include <thread>
#include <shared_mutex>
#include <testo/StinkingPileOfShit.hpp>

struct VM {
	VM(std::shared_ptr<Guest> guest);
	~VM();

	std::shared_ptr<Guest> guest;

	std::shared_mutex mutex;
	stb::Image view;
	std::string query;

private:
	StinkingPileOfShit shit;
	std::thread thread;
	void run();
	bool running = false;
};
