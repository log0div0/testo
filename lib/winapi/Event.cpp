
#include "Event.hpp"
#include <stdexcept>

namespace winapi {

Event::Event(BOOL manual_reset, BOOL is_signaled) {
	handle = CreateEvent(NULL, manual_reset, is_signaled, NULL);
	if (!handle) {
		throw std::runtime_error("CreateEvent failed");
	}
}

Event::~Event() {
	CloseHandle(handle);
}

void Event::set() {
	SetEvent(handle);
}

DWORD Event::wait(DWORD ms) {
	return WaitForSingleObject(handle, ms);
}

}