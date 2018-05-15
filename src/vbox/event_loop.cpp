
#include "event_loop.hpp"
#include "api.hpp"

namespace vbox {

EventLoop::EventLoop() {
	_thread = std::thread([] {
		vbox::api->pfnProcessEventQueue(-1);
	});
}

EventLoop::~EventLoop() {
	vbox::api->pfnInterruptEventQueueProcessing();
	_thread.join();
}

}
