
#pragma once

#include "Hypervisor.hpp"
#include <thread>
#include <shared_mutex>
#include <testo/nn/text_detector/TextDetector.hpp>

struct VM {
	VM(std::shared_ptr<Guest> guest);
	~VM();

	std::shared_ptr<Guest> guest;

	std::shared_mutex mutex;
	Image view;
	std::string query;
	std::string foreground, background;

private:
	TextDetector text_detector;
	std::thread thread;
	void run();
	bool running = false;
};
