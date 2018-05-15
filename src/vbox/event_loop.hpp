
#pragma once

#include <thread>

namespace vbox {

struct EventLoop {
	EventLoop();
	~EventLoop();

	EventLoop(const EventLoop&) = delete;
	EventLoop& operator=(const EventLoop&) = delete;
	EventLoop(EventLoop&& other);
	EventLoop& operator=(EventLoop&& other);

private:
	std::thread _thread;
};

}
